from __future__ import annotations
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import List

# Paths
data_dir  = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\test")  # test or val
weights   = Path("stage_d_tsm_attn.pth")
class_txt = Path("stage_d_classes.txt")

from stage_D_new import TSM_Attn, ASLSeqDataset, NUM_FRAMES, DEVICE

# Metrics and display tools
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate


#EVALUATION FUNCTION
def eval_full(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module = None,
    class_names: List[str] = None,
    desc: str = "Eval",
):
    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    loss_sum = 0.0

    y_true = []
    y_pred = []

    bar = tqdm(total=len(loader.dataset), unit="vid", desc=desc, ncols=80)

    with torch.no_grad():
        for clip, lab in loader:
            clip = clip.to(DEVICE)
            lab = lab.to(DEVICE)

            out = model(clip)

            if criterion is not None:
                loss = criterion(out, lab)
                loss_sum += loss.item() * lab.size(0)

            total += lab.size(0)

            _, pred_top5 = out.topk(5, dim=1, largest=True, sorted=True)
            correct = pred_top5.t().eq(lab.view(1, -1).expand_as(pred_top5.t()))

            top1 += correct[:1].flatten().sum().item()
            top5 += correct.flatten().sum().item()

            y_true.extend(lab.cpu().tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())

            bar.update(lab.size(0))

    bar.close()

    avg_loss = loss_sum / total if criterion is not None else None

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True  # for extracting macro/weighted avg
    )

    conf_mat = confusion_matrix(y_true, y_pred)

    return {
        "loss": avg_loss,
        "top1": 100 * top1 / total,
        "top5": 100 * top5 / total,
        "report": report_dict,
        "conf_mat": conf_mat
    }


def main() -> None:
    split_name = data_dir.parts[-1]

    # Load class names
    if class_txt.exists():
        with class_txt.open(encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = None

    ds = ASLSeqDataset(data_dir, split="val", T=NUM_FRAMES)
    if len(ds) == 0:
        raise RuntimeError(f"No videos found in {data_dir}. Did you extract frames?")

    print(f"{split_name.capitalize()} set : {len(ds)} videos   {class_names} classes\n")

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = TSM_Attn(len(class_names)).to(DEVICE)
    weights_loaded = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(weights_loaded, strict=False)

    metrics = eval_full(model, loader, nn.CrossEntropyLoss(), class_names, desc=f"Scoring {split_name}")

    # Print summary
    if metrics["loss"] is not None:
        print(f"Average loss    : {metrics['loss']:.4f}")
    print(f"Top-1 accuracy      : {metrics['top1']:6.2f}%")
    print(f"Top-5 accuracy      : {metrics['top5']:6.2f}%\n")

    # Show macro and weighted average metrics only
    report = metrics["report"]
    print("Precision / Recall / F1 (Macro & Weighted Avg):")
    print(f"Macro  Avg Precision: {report['macro avg']['precision']:.3f}   "
          f"Recall: {report['macro avg']['recall']:.3f}   "
          f"F1-score: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg  Precision: {report['weighted avg']['precision']:.3f}   "
          f"Recall: {report['weighted avg']['recall']:.3f}   "
          f"F1-score: {report['weighted avg']['f1-score']:.3f}")

    # Print confusion matrix (optional if <30 classes)
    if class_names is not None and len(class_names) <= 30:
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(tabulate(metrics["conf_mat"], headers=class_names, showindex=class_names,
                       tablefmt="grid", floatfmt="d"))


#ENTRY
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # For Windows compatibility
    main()
