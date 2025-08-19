# stage C
from __future__ import annotations
import mediapipe as mp
import math, time, functools, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2
from torchvision.transforms import (
    Compose,Resize, RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, GaussianBlur, ToTensor, Normalize,
    RandomErasing)

################# PATHS
root = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images")
train_dir = root / "train"
val_dir   = root / "val"
backbone_checkpoint      = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_b_cnn_attn_final.pth"
save_model_w     = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_adv_cnn_attn.pth"
save_performance_h     = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_adv_hist.npy"

###############    HYPER-PARAMS
num_frames    = 32
img_size      = 160
batch_size      = 4
max_epochs    = 50
LR_BACKBONE   = 1e-4
LR_FULL       = 5e-4
FREEZE_EPOCHS = 5
MIXUP_ALPHA   = 0.1
PATIENCE      = 7                  # increased patience for better training
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


################# AUGMENTATIONS
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)
tr_train = Compose([
    RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(0.2,0.2,0.2,0.1),
    GaussianBlur(kernel_size=3),
    ToTensor(), Normalize(imagenet_mean, imagenet_std),
    RandomErasing(p=0.1)
])
tr_eval = Compose([
    Resize((img_size, img_size)),   # deterministic, keeps aspect inside bbox
    ToTensor(),
    Normalize(imagenet_mean, imagenet_std)
])

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=False)

@functools.lru_cache(maxsize=4096)
def detect_bbox(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    ##Landmark base cropping
    res = mp_holistic.process(np.array(img)[:, :, ::-1])

    def extract(lms):
        return [(lm.x * w, lm.y * h) for lm in lms.landmark] if lms else []

    hand_points = extract(res.left_hand_landmarks) + extract(res.right_hand_landmarks)
    face_points = extract(res.face_landmarks)

    if hand_points:
        xs, ys = zip(*hand_points)
    elif face_points:
        xs, ys = zip(*face_points)
    else:
        return (0, 0, w, h)

    padding = 60
    x1, y1 = max(min(xs)-padding, 0), max(min(ys)-padding, 0)
    x2, y2 = min(max(xs)+padding, w), min(max(ys)+padding, h)

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    l, t = int(max(cx-half, 0)), int(max(cy-half, 0))
    r, b = int(min(cx+half, w)), int(min(cy+half, h))
    return l, t, r, b

# Center-Crop helper (placeholder)
def crop(img_path: Path) -> Image.Image:
    l,t,r,b = detect_bbox(img_path)
    return Image.open(img_path).convert("RGB").crop((l,t,r,b))

# Dataset
class ASLSeqDataset(Dataset):
    def __init__(self, root: Path, split="train", T=32):
        self.root  = Path(root)
        self.split = split
        self.T     = T
        self.cls   = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.idx   = {c:i for i,c in enumerate(self.cls)}
        self.samples = []
        for c in self.cls:
            for vid in (self.root/c).iterdir():
                frames = sorted(vid.glob("*.jpg"))
                if len(frames) >= 2:
                    self.samples.append((frames, self.idx[c]))
        self.tx = tr_train if split == "train" else tr_eval

    def __len__(self): return len(self.samples)

    def select(self, L:int):
        base = np.linspace(0, L-1, self.T, dtype=int)
        if self.split=="train":
            base = np.clip(base + np.random.randint(-1,2,self.T), 0, L-1)
        return base

    def __getitem__(self, i):
        frames,label = self.samples[i]
        idxs = self.select(len(frames))
        clip = torch.stack([self.tx(crop(frames[j])) for j in idxs], dim=1)
        return clip, label

#Model
#building a new neuralnetwork block called TemporalAttn. It inherits from PyTorch’s basic “Module” so it behaves like any other layer or model.
class TemporalAttn(nn.Module):
    def __init__(self,n_cls,emb_dim=256):              # in this constructor we built a network that will be using
        super().__init__()
        m = mobilenet_v2(weights=None)                   # grab a readymade MobileNetV2 network but without its pretrained weights here. because i will load our own weight later which got from stage B.
        self.backbone = m.features                         # keeping only the feature-extractor part of MobileNet (all the convolution layers). This will turn each frame into a 1280-length feature vector.
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Linear(1280,emb_dim,bias=False) #A small linear (fully-connected) layer that shrinks the 1280-length vector down to 256 numbers (our embedding).
        self.bn       = nn.BatchNorm1d(emb_dim)               #BatchNorm: keeps those 256 numbers well-scaled and stable during training.
        self.dropout  = nn.Dropout(0.5)                  #Randomly turns off half of the 256 numbers while training – a cheap trick to stop over-fitting.
        self.attn     = nn.Linear(emb_dim,1)  #A tiny layer that will output one score per frame. Later we turn those scores into attention weights (how important each frame is).
        self.fc       = nn.Linear(emb_dim,n_cls)         #Final layer that converts the 256-number summary of a whole video into 100 class scores.

    def forward(self,x):
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        f = self.pool(self.backbone(x)).flatten(1)
        f = F.relu(self.bn(self.proj(f)))
        f = self.dropout(f).view(B,T,-1)
        w = F.softmax(self.attn(f).squeeze(-1),dim=1).unsqueeze(-1)
        emb = (w*f).sum(1)
        return self.fc(emb)

#  MixUp and Loss
def mixup(x,y,alpha=MIXUP_ALPHA):
    if alpha<=0: return x,y,None
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], (y,y[idx],lam), idx

def ce_mix(ce, logits, target):
    if isinstance(target,tuple):
        y1,y2,lam = target
        return lam*ce(logits,y1)+(1-lam)*ce(logits,y2)
    return ce(logits,target)

# Evaluation
@torch.no_grad()
def evaluate(model,loader,n_cls):
    model.eval(); ce=nn.CrossEntropyLoss()
    tot=correct=0; loss=0.
    class_corr = np.zeros(n_cls,int)
    class_tot  = np.zeros(n_cls,int)
    for clip,lab in loader:
        clip,lab = clip.to(DEVICE), lab.to(DEVICE)
        out = model(clip)
        loss += ce(out,lab).item()*lab.size(0)
        pred = out.argmax(1).cpu().numpy()
        labs = lab.cpu().numpy()
        for p,t in zip(pred,labs):
            class_tot[t]+=1
            class_corr[t]+= int(p==t)
        correct += (pred==labs).sum(); tot += labs.size
    per_cls = class_corr/class_tot.clip(1)
    return loss/tot, 100*correct/tot, per_cls

# Main
def main():
    torch.backends.cudnn.benchmark = True

    # — data loaders —
    train_ds = ASLSeqDataset(train_dir, "train", num_frames)
    val_ds   = ASLSeqDataset(val_dir, "val", num_frames)
    labels   = np.array([l for _,l in train_ds.samples])
    wts      = 1.0/(np.bincount(labels,minlength=len(train_ds.cls))+1e-6)
    samp_wts = wts[labels]
    NUM_WORKERS = 6

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(samp_wts, len(samp_wts), replacement=True),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    # — model & optimizer —
    model = TemporalAttn(len(train_ds.cls)).to(DEVICE)
    ckpt  = torch.load(backbone_checkpoint, map_location="cpu")
    st    = {k.replace("backbone.",""):v for k,v in ckpt.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(st,strict=False)
    for p in model.backbone.parameters(): p.requires_grad=False

    # Optimizer
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.proj.parameters()) +
                   list(model.bn.parameters()) +
                   list(model.attn.parameters()) +
                   list(model.fc.parameters()), "lr": LR_FULL}
    ], weight_decay=1e-4)

    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=1e-6
    )

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {"epoch":[], "train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_loss, epochs_no_improve = math.inf, 0

#######training loop
    for ep in range(1, max_epochs + 1):
        model.train()
        tl, tc, tot = 0, 0, 0
        t0 = time.time()

        if ep == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f" Backbone unfrozen at epoch {ep}")

        for clip, lab in tqdm(train_ld, desc=f"Ep{ep:02d}"):
            clip, lab = clip.to(DEVICE), lab.to(DEVICE)
            clip, mix_tgt, _ = mixup(clip, lab)
            out = model(clip)
            loss = ce_mix(ce, out, mix_tgt if mix_tgt else lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = lab.size(0)
            tl += loss.item() * bs
            tc += int((out.argmax(1) == lab).sum())
            tot += bs

        scheduler.step()  # Cosine LR step each epoch

        tr_loss, tr_acc = tl / tot, 100 * tc / tot
        v_loss, v_acc, per_cls = evaluate(model, val_ld, len(train_ds.cls))

        # Early stopping logic
        if v_loss < best_loss:
            best_loss, epochs_no_improve = v_loss, 0
            torch.save(model.state_dict(), save_model_w)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Logging clearly
        lrs = ", ".join(f"{g['lr']:.2e}" for g in opt.param_groups)
        print(f"Ep{ep:02d}|LRs[{lrs}]|Train {tr_loss:.3f}/{tr_acc:.1f}% "
              f"| Val {v_loss:.3f}/{v_acc:.1f}% | {(time.time() - t0) / 60:.1f}m")

        worst5 = np.argsort(per_cls)[:5]
        print("Worst classes:", [train_ds.cls[i] for i in worst5], per_cls[worst5])

        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)

    np.save(save_performance_h, hist)
    print("Training complete.")

if __name__=="__main__":
    main()
