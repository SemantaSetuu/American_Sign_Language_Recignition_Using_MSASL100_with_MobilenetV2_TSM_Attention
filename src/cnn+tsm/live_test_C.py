from __future__ import annotations
import cv2, torch, numpy as np, mediapipe as mp
from pathlib import Path
from collections import deque
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from stage_C_new import TemporalAttn

#  paths
WEIGHTS = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_adv_cnn_attn.pth")
CLASSES = Path("stage_d_classes.txt")
assert WEIGHTS.exists() and CLASSES.exists(), "Weights or class-list file missing"


NUM_FRAMES = 32
IMG_SIZE   = 160
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# load classes & model
with open(CLASSES, "r") as f:
    cls_names = [ln.strip() for ln in f if ln.strip()]
    model = TemporalAttn(len(cls_names)).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
print(f"Loaded Stage-C weights from: {WEIGHTS}")

#torchvision transform
transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#MediaPipe Holistic crop helper
mp_hol = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    refine_face_landmarks=True
)

def detect_bbox(rgb: np.ndarray):
    h, w, _ = rgb.shape
    res = mp_hol.process(rgb)
    pts = []
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            pts += [(int(lm.x*w), int(lm.y*h)) for lm in hand.landmark]
    if not pts and res.face_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in res.face_landmarks.landmark]
    if not pts:
        return 0, 0, w, h                       # fallback = full frame

    xs, ys = zip(*pts)
    pad = 60
    x1, y1 = max(min(xs)-pad, 0), max(min(ys)-pad, 0)
    x2, y2 = min(max(xs)+pad, w), min(max(ys)+pad, h)
    side   = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half   = side//2
    l = max(cx-half, 0); t = max(cy-half, 0)
    r = min(cx+half, w); b = min(cy+half, h)
    return l, t, r, b

def crop_and_resize(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    l, t, r, b = detect_bbox(rgb)
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
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Live Stage C", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
