# Word-Based American Sign Language Recognition Using 2D CNN MobileNet-V2 with TSM and Attention

## 📖 Project Overview
This project, **Word-Based American Sign Language Recognition Using 2D CNN MobileNet-V2 with Temporal Shift Module and Attention**, focuses on developing an efficient deep learning pipeline for recognizing ASL signs from videos.  

The system is built on **MobileNetV2** as the backbone, and the trained weights are passed forward at each stage for progressive improvement:
- **Stage A**: Baseline MobileNetV2 CNN  
- **Stage B**: MobileNetV2 + Attention  
- **Stage C**: MobileNetV2 + Attention with stronger augmentation/regularization  
- **Stage D**: MobileNetV2 + Temporal Shift Module (TSM) + Attention  

On the **MS-ASL100 test set**, the final **Stage D** model achieved:  
- **Top-1 Accuracy = 51.97%**  
- **Top-5 Accuracy = 77.49%**  

(For reference, Stage C reached **Top-1 = 49.42%** and **Top-5 = 74.71%**.)  

The system supports **real-time webcam recognition** and a **Streamlit web app** for uploaded videos.
<img width="1767" height="471" alt="image" src="https://github.com/user-attachments/assets/6fa46579-cc3a-4e85-af37-dda5aec17b9c" />


---

## ⚙️ System Requirements & Environment Setup

This project was developed and tested on the following system (as reported in the thesis):
- **CPU**: 13th Gen Intel Core i7-13620H  
- **GPU**: NVIDIA GeForce RTX 4050 (6 GB VRAM)  
- **RAM**: 16 GB  
- **Operating System**: Windows  
- **Python**: 3.12  
- **IDE**: PyCharm  
- **Deep Learning Framework**: PyTorch (+ TorchVision)  

### 🔹 GPU Setup on Windows
To ensure all scripts run on the discrete NVIDIA GPU:
1. Open **NVIDIA Control Panel → Manage 3D settings → Program Settings**.  
2. Add/select the Python interpreter you are using (e.g., `python.exe` from Python 3.12).  
3. Set **Preferred graphics processor** to **High-performance NVIDIA processor**.  
4. Apply changes.  

> Each script also checks CUDA availability at runtime and falls back to CPU if unavailable.

---

## 🐍 Python Environment & Dependencies

This project was developed in **PyCharm** using a locally installed **Python 3.12 interpreter** (`python.exe`).  
No virtual environment was used during development. All libraries were installed directly from the PyCharm terminal with **pip**.  

### Installing Dependencies
All required libraries are listed in **requirements.txt**. To install them, run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer installing without a file, you can copy-paste this command:
```bash
pip install torch torchvision opencv-python mediapipe numpy tqdm scikit-learn tabulate Pillow streamlit ffmpeg-python yt-dlp
```

📌 **Notes**:  
- Since a system interpreter was used, packages are installed globally for Python 3.12.  
- If you want isolation, you can later set up a virtual environment and reinstall dependencies using the same file.  
- For GPU acceleration, ensure the correct **CUDA-enabled PyTorch build** is installed for your GPU.  

---

## 🛠️ External Tools Required

In addition to Python libraries, the following external tools are needed:
- **FFmpeg** → required for trimming and extracting video frames.  
  - Install FFmpeg for Windows and add `<ffmpeg>\bin` to your system PATH,  
  - or update the `ffmpeg_location` in `download_trim.py` to point to `ffmpeg.exe`.  
- **yt-dlp** → Python package already listed in requirements, used for downloading MS-ASL videos from YouTube.  
- **NVIDIA GPU Driver** → keep drivers updated to ensure CUDA compatibility with PyTorch.  
- **Webcam (optional)** → required for real-time recognition scripts (`live_test_C.py`, `live_test_D.py`).  
- **Browser (optional)** → for running the Streamlit demo (`streamlit run app.py`).  

---

## 🚀 How to Run the Project  

### 1️⃣ Dataset Preparation  
All artifacts are created under the top-level `data/` directory by running the provided scripts in order.  

#### a) Filter to ASL100 (creates `data/lists/`)  
Run the file `filter_asl100.py`.  
- Reads the original MS-ASL metadata.  
- Filters the dataset to the first **100 classes**.  
- Creates a `lists/` folder containing train/val/test JSON files.  

```
data/
 └── lists/
      ├── ASL100_train.json
      ├── ASL100_val.json
      └── ASL100_test.json
```

#### b) Download & Trim (creates `data/raw/` and `data/clips/`)  
Run the file `download_trim.py`.  
- Downloads raw YouTube videos for each JSON sample using **yt-dlp**.  
- Stores full videos in `data/raw/{split}/{class}/`.  
- Trims videos using **FFmpeg**, saving them in `data/clips/{split}/{class}/`.  

```
data/
 ├── lists/
 │    ├── ASL100_train.json
 │    ├── ASL100_val.json
 │    └── ASL100_test.json
 ├── raw/          # downloaded full videos
 │    ├── train/
 │    ├── val/
 │    └── test/
 └── clips/        # time-trimmed clips
      ├── train/
      ├── val/
      └── test/
```

#### c) Extract Frames at 16 FPS (creates `data/images/`)  
Run the file `extract_frames.py`.  
- Extracts frames from trimmed clips using **OpenCV (cv2)**.  
- Saves frames into `data/images/{split}/{class}/{video_id}/` at ~16 FPS.  

```
data/
 ├── lists/
 ├── raw/
 ├── clips/
 └── images/       # extracted frames (≈16 FPS)
      ├── train/
      ├── val/
      └── test/
         └── {class}/
             └── {video_id}/
                 ├── 000001.jpg
                 ├── 000002.jpg
                 └── ...
```

📌 **Notes**:  
- Training/evaluation reads from `data/images/`.  
- Scripts use **absolute paths**; update them for your machine if needed.  
- Ensure **FFmpeg** and **yt-dlp** are installed for this step.  

---

### 2️⃣ Model Training  
Training happens in four stages, each using weights from the previous stage.  

- **Stage A** (Baseline MobileNetV2) → Run `stage_A.py`  
  - Trains CNN on single frames.  
  - Saves `mnet_frame_pretrain_correct.pth`.  

- **Stage B** (MobileNetV2 + Attention, 32 frames) → Run `stage_B.py`  
  - Adds temporal attention.  
  - Saves `stage_b_cnn_attn_final.pth`.  

- **Stage C** (MobileNetV2 + Attention + stronger regularization) → Run `stage_C_new.py`  
  - Uses stronger augmentation and early stopping.  
  - Saves `stage_c_adv_cnn_attn.pth`.  

- **Stage D** (MobileNetV2 + TSM + Attention, 64 frames, final) → Run `stage_D_new.py`  
  - Adds Temporal Shift Module (TSM).  
  - Saves `stage_d_tsm_attn.pth`.  

---

### 3️⃣ Model Evaluation  
Run these scripts to check performance on the test set:  
- `Evaluation_B.py` → Evaluates Stage B.  
- `Evaluation_C.py` → Evaluates Stage C.  
- `evaluation_D.py` → Evaluates Stage D (prints Precision, Recall, F1, and confusion matrix).  

---

### 4️⃣ Testing & Deployment  
- **Streamlit Web App**: Run `app.py`.  
  - Upload a video to get predictions (Stage C or Stage D).  
- **Live Webcam Recognition**:  
  - Run `live_test_C.py` for Stage C.  
  - Run `live_test_D.py` for Stage D (final).  
  - Opens webcam and shows real-time predictions with confidence scores.  

---
