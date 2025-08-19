#evaluation_C.py
from __future__ import annotations
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


ROOT      = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code")
DATA_DIR  = ROOT / "data" / "images" / "test"
WEIGHTS   = ROOT / "src" / "cnn+tsm" / "stage_c_adv_cnn_attn.pth"
CLASS_TXT = ROOT / "src" / "cnn+tsm" / "stage_c_classes.txt"


#
from stage_C_new import TemporalAttn, ASLSeqDataset, num_frames, DEVICE

# EVALUATION HELPER
def eval_topk(model, loader, criterion: nn.Module | None = None):
    model.eval(); top1=top5=tot=loss=0.
    bar = tqdm(total=len(loader.dataset), unit="vid",
               desc=f"Testing {split_name}", ncols=80)
    with torch.no_grad():
        for clip, lab in loader:
            clip, lab = clip.to(DEVICE), lab.to(DEVICE)
            out = model(clip)
            if criterion:
                loss += criterion(out, lab).item() * lab.size(0)
            tot += lab.size(0)

            _, pred = out.topk(5, 1, True, True)
            corr    = pred.t().eq(lab.view(1, -1).expand_as(pred.t()))
            top1   += corr[:1].flatten().sum().item()
            top5   += corr[:5].flatten().sum().item()

            bar.update(lab.size(0))
    bar.close()
    return ((loss/tot) if criterion else None, 100*top1/tot, 100*top5/tot)

# MAIN
def main() -> None:
    global split_name
    split_name = DATA_DIR.parts[-1]           # "val" or "test"

    #  class list / output size
    if CLASS_TXT.exists():
        cls_list = [l.strip() for l in CLASS_TXT.open(encoding="utf‑8") if l.strip()]
        n_cls = len(cls_list)
    else:
        n_cls = torch.load(WEIGHTS, map_location="cpu")["fc.weight"].shape[0]

    #  dataset & loader
    ds = ASLSeqDataset(DATA_DIR, split="val", T=num_frames)
    if len(ds) == 0:
        raise RuntimeError(f"No clips found in {DATA_DIR}. "
                           "Did you extract frames for this split?")
    print(f"{split_name.capitalize()} set : {len(ds)} videos   "
          f"{len(ds.cls)}/{n_cls} classes")

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=4, pin_memory=True)

    #  model
    model = TemporalAttn(n_cls).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE), strict=True)

    # evaluate
    loss, top1, top5 = eval_topk(model, loader, nn.CrossEntropyLoss())
    print(f"\nTop‑1 accuracy : {top1:6.2f}%")
    print(f"Top‑5 accuracy : {top5:6.2f}%")
    print(f"{split_name.capitalize()} loss : {loss:.4f}")

# ───────── ENTRY POINT ─────────
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # Windows
    main()