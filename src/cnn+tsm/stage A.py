import numpy as np, torch, torch.nn as nn
from pathlib import Path
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
import functools

# Paths
train_dir = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\train"
val_dir   = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\val"

save_w = "mnet_frame_pretrain_correct.pth"
save_h = "hist_mnet_frame_correct.npy"

# Hyper-params
batch_size  = 64
epoch    = 10
LR        = 1e-4
val_every = 2
img_size  = 160
device    = "cuda" if torch.cuda.is_available() else "cpu"

#Transforms
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)
tx_train = Compose([
    Resize((img_size, img_size)), RandomHorizontalFlip(), ColorJitter(0.2, 0.2, 0.2, 0.1),
    ToTensor(), Normalize(imagenet_mean, imagenet_std)
])
tx_eval = Compose([Resize((img_size, img_size)), ToTensor(), Normalize(imagenet_mean, imagenet_std)])

# Corrected MediaPipe-based Cropping
mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=True)
bbox_cache: dict[Path, tuple[int,int,int,int]] = {}

@functools.lru_cache(maxsize=4096)
def detect_bbox(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    res = mp_holistic.process(np.array(img)[:, :, ::-1])

    xs, ys, detected = [], [], False
    for lm_set in [res.face_landmarks, res.left_hand_landmarks, res.right_hand_landmarks]:
        if lm_set:
            detected = True
            for lm in lm_set.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

    if not detected:
        return (0, 0, w, h)

    x1, y1 = max(min(xs)-40, 0), max(min(ys)-40, 0)
    x2, y2 = min(max(xs)+40, w), min(max(ys)+40, h)

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    l, t = int(max(cx-half, 0)), int(max(cy-half, 0))
    r, b = int(min(cx+half, w)), int(min(cy+half, h))
    return l, t, r, b

def crop(img_path: Path) -> Image.Image:
    if img_path.parent not in bbox_cache:
        bbox_cache[img_path.parent] = detect_bbox(img_path)
    l, t, r, b = bbox_cache[img_path.parent]
    return Image.open(img_path).convert("RGB").crop((l, t, r, b))

# Dataset Class
class FrameDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split
        self.cls_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.cls2idx = {c: i for i,c in enumerate(self.cls_names)}
        self.samples = [(f, self.cls2idx[cls]) for cls in self.cls_names for vid in (self.root/cls).iterdir() for f in vid.glob("*.jpg")]
        self.tx = tx_train if split == "train" else tx_eval

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return self.tx(crop(img_path)), label

# model
class MobileNetFrame(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        m = mobilenet_v2(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(1280, n_cls)
        self.net = m
    def forward(self, x): return self.net(x)

# eval Function
@torch.no_grad()
def eval_top1(model, loader):
    model.eval(); correct=total=0
    for x,y in loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred == y.to(device)).sum().item(); total+=y.size(0)
    return 100*correct/total

# Main
if __name__ == "__main__":
    train_ds, val_ds = FrameDataset(train_dir), FrameDataset(val_dir, split="val")
    train_ld = DataLoader(train_ds, batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size)

    model = MobileNetFrame(len(train_ds.cls_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {k: [] for k in ["epoch","loss","train_acc","val_acc"]}

    import time

    for ep in range(1, epoch + 1):
        t0 = time.time()
        model.train(); tloss = correct = total = 0

        for x,y in tqdm(train_ld, desc=f"E{ep:02d}"):
            logits = model(x.to(device))
            loss = ce(logits, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item()*y.size(0); correct += (logits.argmax(1) == y.to(device)).sum().item(); total+=y.size(0)

        tr_acc, tr_loss = 100*correct/total, tloss/total
        val_acc = eval_top1(model,val_ld) if ep % val_every == 0 else float('nan')

        lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {ep:02d} | LR: {lr:.2e} | Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s")

        hist["epoch"].append(ep); hist["loss"].append(tr_loss); hist["train_acc"].append(tr_acc); hist["val_acc"].append(val_acc)

    torch.save(model.state_dict(), save_w)
    np.save(save_h, hist)
