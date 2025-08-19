
from __future__ import annotations
import cv2, torch, numpy as np, mediapipe as mp
from pathlib import Path
from collections import deque
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

#paths
WEIGHTS = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_d_tsm_attn.pth")
CLASSES = Path("stage_d_classes.txt")
assert WEIGHTS.exists() and CLASSES.exists(), "Weights or class‑list file missing"

# ────────── constants ──────────
NUM_FRAMES = 64
IMG_SIZE   = 160
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ────────── model definition (must match Stage‑D) ──────────
class TSM_Attn(nn.Module):
    def __init__(self, n_cls: int, emb_dim: int = 256):
        super().__init__()
        m = mobilenet_v2(weights=None)
        self.backbone = m.features
        self.shift_div = 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, emb_dim, bias=False)
        self.bn   = nn.BatchNorm1d(emb_dim)
        self.attn = nn.Linear(emb_dim, 1)
        self.fc   = nn.Linear(emb_dim, n_cls)

    def tsm(self, x: torch.Tensor):
        # x shape (B·T, C, H, W)  → temporal shift
        B_T, C, H, W = x.shape
        B = B_T // NUM_FRAMES
        x = x.view(B, NUM_FRAMES, C, H, W)
        fold = C // self.shift_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold]       = x[:, 1:, :fold]          # ← shift left
        out[:, 1:, fold:2*fold]  = x[:, :-1, fold:2*fold]   # → shift right
        out[:, :, 2*fold:]       = x[:, :, 2*fold:]         # stay
        return out.view(B_T, C, H, W)

    def forward(self, x: torch.Tensor):
        # x (B, 3, T, H, W)
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        x = self.tsm(x)
        f = self.pool(self.backbone(x)).flatten(1)
        f = F.relu(self.bn(self.proj(f))).view(B, T, -1)
        w = F.softmax(self.attn(f).squeeze(-1), dim=1).unsqueeze(-1)
        emb = (w * f).sum(1)
        return self.fc(emb)

#load classes and model
with open(CLASSES, "r") as f:
    cls_names = [ln.strip() for ln in f if ln.strip()]
model = TSM_Attn(len(cls_names)).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"Loaded Stage‑D weights from: {WEIGHTS}")

#torchvision transform
transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

#MediaPipe Holistic for cropping
mp_hol = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    refine_face_landmarks=True
)

def _detect_bbox_np(rgb: np.ndarray):
    """Return (l,t,r,b) bounding box in the numpy image (H,W,3) space."""
    h, w, _ = rgb.shape
    res = mp_hol.process(rgb)
    # try hands first
    pts = []
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            pts += [(int(lm.x*w), int(lm.y*h)) for lm in hand.landmark]
    # fallback to face
    if not pts and res.face_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in res.face_landmarks.landmark]
    # if still nothing use full frame
    if not pts:
        return 0,0,w,h
    xs, ys = zip(*pts)
    pad = 60
    x1,y1 = max(min(xs)-pad,0), max(min(ys)-pad,0)
    x2,y2 = min(max(xs)+pad,w), min(max(ys)+pad,h)
    side = max(x2-x1, y2-y1)
    cx,cy = (x1+x2)//2, (y1+y2)//2
    half  = side//2
    l = max(cx-half, 0); t = max(cy-half, 0)
    r = min(cx+half, w); b = min(cy+half, h)
    return l,t,r,b

def crop_and_resize(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    l,t,r,b = _detect_bbox_np(rgb)
    cropped = Image.fromarray(rgb[t:b, l:r])
    return transform(cropped)

#webcam loop
buf = deque(maxlen=NUM_FRAMES)
cap = cv2.VideoCapture(0)
print("Press  'q'  to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tensor_frame = crop_and_resize(frame)
    buf.append(tensor_frame)

    # draw prediction if we have enough frames
    disp = frame.copy()
    if len(buf) == NUM_FRAMES:
        clip = torch.stack(list(buf), dim=1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(clip)
            prob   = torch.softmax(logits, dim=1)[0]
            pred   = prob.argmax().item()
            conf   = prob[pred].item()*100
            label  = cls_names[pred].upper()
        cv2.putText(disp, f"{label} ({conf:.1f}%)",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Live Stage D", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
