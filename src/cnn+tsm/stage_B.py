#stage_B
from __future__ import annotations
import math, time, functools, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                     RandomHorizontalFlip, ColorJitter)

################ PATHS
root        = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images")
train_dir   = root / "train"
val_dir     = root / "val"
backbone_checkpoint = "mnet_frame_pretrain_correct.pth"

save_model_w = "stage_b_cnn_attn_final.pth"
save_performance_h = "stage_b_hist_cnn_attn.npy"

##################### HYPER-PARAMS
num_frames   = 32
img_size     = 160
batch_sz     = 4
epoch        = 30
LR_of_Frozen_Backbone_layers = 1e-4
LR_after_Unfroze      = 5e-4
freeze_epochs = 5
val_every    = 1
mixup_alpha  = 0.1   # slightly lowered
device       = "cuda" if torch.cuda.is_available() else "cpu"


############## Transforms
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)
tx_train = Compose([
    Resize((img_size, img_size)),
    RandomHorizontalFlip(),
    ColorJitter(0.2,0.2,0.2,0.1),
    ToTensor(), Normalize(imagenet_mean, imagenet_std)
])
tx_eval  = Compose([
    Resize((img_size, img_size)),
    ToTensor(), Normalize(imagenet_mean, imagenet_std)
])

################ MediaPipe crop helpers
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=0,refine_face_landmarks=False, enable_segmentation=False)
except ModuleNotFoundError:
    mp_holistic = None  # fall back to centre-crop

box_cache: dict[Path, tuple[int,int,int,int]] = {}
@functools.lru_cache(maxsize=4096)
def detect_bbox(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    res = mp_holistic.process(np.array(img)[:, :, ::-1])

    def extract(lms):
        return [(lm.x * w, lm.y * h) for lm in lms.landmark] if lms else []

    hand_points = extract(res.left_hand_landmarks) + extract(res.right_hand_landmarks)
    face_points = extract(res.face_landmarks)

    # PRIORITIZE HANDS (Crucial for ASL)
    if hand_points:
        xs, ys = zip(*hand_points)
    elif face_points:
        xs, ys = zip(*face_points)
    else:
        return (0, 0, w, h)

    padding = 60  # generous padding around detected landmarks
    x1, y1 = max(min(xs)-padding, 0), max(min(ys)-padding, 0)
    x2, y2 = min(max(xs)+padding, w), min(max(ys)+padding, h)

    # Square bbox
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    l, t = int(max(cx-half, 0)), int(max(cy-half, 0))
    r, b = int(min(cx+half, w)), int(min(cy+half, h))
    return l, t, r, b


def crop(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if img_path.parent not in box_cache:
        box_cache[img_path.parent] = detect_bbox(img_path)
    l,t,r,b = box_cache[img_path.parent]
    if r != -1:
        img = img.crop((l,t,r,b))
    else:  # centre-crop fallback
        w,h = img.size; side = min(w,h)
        img = img.crop(((w-side)//2, (h-side)//2,
                        (w+side)//2, (h+side)//2))
    return img

############## Dataset
class ASLSeqDataset(Dataset):
    def __init__(self, root: Path, split="train", T=32):
        self.root  = Path(root)
        self.split = split.lower()
        self.T     = T
        self.cls_names = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.cls2idx   = {c:i for i,c in enumerate(self.cls_names)}
        self.samples   = []
        for cls in self.cls_names:
            for vid in (self.root/cls).iterdir():
                if vid.is_dir():
                    frames = sorted(vid.glob("*.jpg"))
                    if len(frames) >= 2:
                        self.samples.append((frames, self.cls2idx[cls]))
        self.tx = tx_train if split == "train" else tx_eval

    def __len__(self): return len(self.samples)

    def select(self, L):
        base = np.linspace(0,L-1,self.T,dtype=int)
        if self.split=="train":
            base = np.clip(base + np.random.randint(-1,2,size=self.T), 0, L-1)
        return base

    def __getitem__(self, idx):
        frames,label = self.samples[idx]
        idxs = self.select(len(frames))
        clip = torch.stack([self.tx(crop(frames[i])) for i in idxs], dim=1)  # (3,T,H,W)
        return clip, label

################# Model
class TemporalAttn(nn.Module):
    #MobileNet-V2 per-frame -> soft attention -> logits
    def __init__(self, n_cls, emb_dim=256):
        super().__init__()
        m = mobilenet_v2(weights=None)             ## grab a readymade MobileNetV2 network but without its pretrained weights here. because i will load our own weight later which got from stage B.
        self.backbone = m.features                ## keeping only the feature-extractor part of MobileNet (all the convolution layers). This will turn each frame into a 1280-length feature vector.
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Linear(1280, emb_dim, bias=False) #A small linear (fully-connected) layer that shrinks the 1280-length vector down to 256 numbers (our embedding).
        self.bn    = nn.BatchNorm1d(emb_dim)                #BatchNorm: keeps those 256 numbers well-scaled and stable during training.
        self.attn  = nn.Linear(emb_dim, 1)      #A tiny layer that will output one score per frame. Later we turn those scores into attention weights (how important each frame is).
        self.fc    = nn.Linear(emb_dim, n_cls)         ##Final layer that converts the 256-number summary of a whole video into 100 class scores.

    def forward(self,x):                                    # x (B,3,T,H,W)
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        feat = self.pool(self.backbone(x)).flatten(1)       # (B·T,1280)
        feat = F.relu(self.bn(self.proj(feat)))             # (B·T,emb)
        feat = feat.view(B,T,-1)                            # (B,T,emb)
        attn_w = F.softmax(self.attn(feat).squeeze(-1),dim=1) # (B,T)
        emb = (attn_w.unsqueeze(-1)*feat).sum(1)            # (B,emb)
        return self.fc(emb)

############### Mix-Up helpers
def mixup(x, y, alpha=mixup_alpha):
    if alpha<=0: return x,y,None
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], (y, y[idx], lam), idx

def ce_mix(ce, logits, target):
    if isinstance(target, tuple):
        y_a,y_b,lam = target
        return lam*ce(logits,y_a)+(1-lam)*ce(logits,y_b)
    return ce(logits,target)

############### Evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ce=nn.CrossEntropyLoss()
    tot=correct=loss=0.
    for clip,lab in loader:
        clip,lab = clip.to(device), lab.to(device)
        logit = model(clip)
        loss += ce(logit,lab).item()*lab.size(0)
        correct += (logit.argmax(1)==lab).sum().item()
        tot += lab.size(0)
    return loss/tot, 100*correct/tot

################### MAIN
def main():
    torch.backends.cudnn.benchmark = True

    # -------- datasets --------
    train_ds = ASLSeqDataset(train_dir, "train", num_frames)
    val_ds   = ASLSeqDataset(val_dir, "val", num_frames)

    # ----------------Data balancing------------
    labels = np.array([lbl for _,lbl in train_ds.samples])
    counts = np.bincount(labels, minlength=len(train_ds.cls_names))
    weights = (1.0/(counts+1e-6))[labels]
    sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    ########---------------

    train_ld = DataLoader(train_ds, batch_sz, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_sz, shuffle=False,
                          num_workers=0, pin_memory=True)

    # -------- model --------
    model = TemporalAttn(len(train_ds.cls_names)).to(device)

    # ---- load StageA MobileNet backbone ----
    ckpt = torch.load(backbone_checkpoint, map_location="cpu")
    bb_state = {k.replace("net.features.",""): v
                for k,v in ckpt.items() if k.startswith("net.features.")}
    missing,_ = model.backbone.load_state_dict(bb_state, strict=False)
    print(f"Loaded Stage-A backbone → {len(bb_state)-len(missing)}/{len(bb_state)} layers")

    # freeze backbone initially
    for p in model.backbone.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_of_Frozen_Backbone_layers},
        {"params": list(model.pool.parameters()) +
                   list(model.proj.parameters()) +
                   list(model.bn.parameters()) +
                   list(model.attn.parameters()) +
                   list(model.fc.parameters()), "lr": LR_after_Unfroze}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {k:[] for k in ["epoch","loss","train_acc","val_loss","val_acc"]}

    print(f"Train clips {len(train_ds)} | Val clips {len(val_ds)} | Classes {len(train_ds.cls_names)}\n")

    # -------- training loop --------
    for ep in range(1, epoch + 1):
        model.train();
        tloss = correct = total = 0
        if ep == freeze_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt.param_groups[0]["lr"] = LR_after_Unfroze * 0.2  # conservative lr change

        for x, y in tqdm(train_ld, desc=f"E{ep:02d}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad();
            loss.backward();
            opt.step()
            tloss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        sched.step()  # cosine LR scheduler step

        tr_acc = 100 * correct / total
        tr_loss = tloss / total
        val_acc = evaluate(model, val_ld) if ep % val_every == 0 else float('nan')

        print(f"Epoch {ep:02d} | LR: {sched.get_last_lr()[0]:.2e} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Loss: {val_acc[0]:.4f} | Val Acc: {val_acc[1]:.2f}%")

        hist["epoch"].append(ep)
        hist["loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

    torch.save(model.state_dict(), save_model_w)
    np.save(save_performance_h, hist)
    print(f"\nDone!  Weights to {save_model_w} | History to {save_performance_h}")

# entry-point
if __name__ == "__main__":
    main()

