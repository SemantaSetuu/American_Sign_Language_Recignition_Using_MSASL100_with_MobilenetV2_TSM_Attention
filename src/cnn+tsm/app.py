#app.py
import cv2, tempfile, pathlib, numpy as np, streamlit as st
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import mediapipe as mp
from stage_C_new import TemporalAttn
from stage_D_new import TSM_Attn
# ----------------------------- paths
CKPT_C = pathlib.Path(r"stage_c_adv_cnn_attn.pth")
CKPT_D = pathlib.Path(r"stage_d_tsm_attn.pth")
CLASSES = pathlib.Path(r"stage_d_classes.txt")

# --------------------------- constants
IMG_SIZE   = 160
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM  = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# --------------------- MediaPipe crop helper
mp_hol = mp.solutions.holistic.Holistic(static_image_mode=True,
                                        model_complexity=1,
                                        refine_face_landmarks=True)

def detect_bbox(rgb: np.ndarray):
    """hands  face  full-frame fallback"""
    h, w, _ = rgb.shape
    res = mp_hol.process(rgb)
    pts = []
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            pts += [(int(lm.x*w), int(lm.y*h)) for lm in hand.landmark]
    if not pts and res.face_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in res.face_landmarks.landmark]
    if not pts:
        return 0,0,w,h
    xs, ys = zip(*pts); pad = 60
    x1,y1 = max(min(xs)-pad,0), max(min(ys)-pad,0)
    x2,y2 = min(max(xs)+pad,w), min(max(ys)+pad,h)
    side  = max(x2-x1, y2-y1); cx,cy=(x1+x2)//2,(y1+y2)//2; half=side//2
    return max(cx-half,0), max(cy-half,0), min(cx+half,w), min(cy+half,h)

def crop_and_tensor(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    l,t,r,b = detect_bbox(rgb)
    im = Image.fromarray(rgb[t:b, l:r])
    return TRANSFORM(im)

# ----------------------- load class names
CLS_NAMES = [ln.strip() for ln in open(CLASSES)]

@st.cache_resource(show_spinner=False)
def load_model(stage:str):
    if stage=="Stage C (32-frame Attention)":
        model = TemporalAttn(len(CLS_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(CKPT_C, map_location=DEVICE))
        n_frames = 32
    else:
        model = TSM_Attn(len(CLS_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(CKPT_D, map_location=DEVICE))
        n_frames = 64
    model.eval()
    return model, n_frames

# ------------------------- UI layout
st.title("ASL Sign Classification Demo")

stage = st.selectbox("Choose model",
                     ["Stage C (32-frame Attention)",
                      "Stage D (TSM-Attention 64-frame)"])

model, NUM_FRAMES = load_model(stage)

uploaded = st.file_uploader("Upload a video file (.mp4 / .avi / .mov)",
                            type=["mp4","avi","mov"])

if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read()); tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok: break
        frames.append(frm)
    cap.release()

    if len(frames) < 2:
        st.warning("Video too short.")
    else:
        idxs = np.linspace(0, len(frames)-1, NUM_FRAMES, dtype=int)
        clip = torch.stack([crop_and_tensor(frames[i]) for i in idxs],
                           dim=1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = F.softmax(model(clip), dim=1)[0]
        idx = prob.argmax().item()
        st.success(f"**Prediction:** {CLS_NAMES[idx].upper()} "
                   f"({prob[idx]*100:.1f} %)")
        st.video(uploaded)
