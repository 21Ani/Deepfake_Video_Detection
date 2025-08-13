# 🎬 Deepfake Detection using Audio & Video Analysis

## 📌 Overview
We developed a platform to **detect Deepfake content** in both **audio and video files** using **deep learning models**.  
The system allows users to upload a file and get a **real-time prediction** of whether the input is genuine or fake.  
A **speedometer-style visualization** displays the confidence of the prediction.

The tool is deployed as a **Flask web application**, and users can interact through a **web interface**.

---

## 📑 Table of Contents
1. [Context](#context)  
2. [Data Sources](#data-sources)  
3. [Methodology](#methodology)  
4. [How to Use](#how-to-use)  
5. [Frontend & Speedometer](#frontend--speedometer)  
6. [Contributors](#contributors)  
7. [Technologies](#technologies)  

---

## 🤝 Contributors

## 0. 🚀 Technologies
- **Python**  
- **Flask** (Web Framework)  
- **PyTorch** (Audio Model)  
- **TensorFlow / Keras** (Video Model)  
- **Librosa** (Audio Processing)  
- **OpenCV** (Video Processing)  
- **scikit-image** (Image/Frame Processing)  
- **Matplotlib** (Visualization)  

---

## I. 📖 Context
Deepfake detection has become increasingly important as manipulated media can **mislead public opinion, commit fraud, and threaten cybersecurity**.  

This project targets **both audio and video content**:  
- **Audio deepfakes** are synthetic voices that imitate a target speaker.  
- **Video deepfakes** manipulate facial expressions or lip-sync to produce realistic but fake videos.  

By combining both modalities, our system increases robustness and reliability.

---

## II. 📂 Data Sources

### 1️⃣ Video Dataset – Deepfake Detection (DFD)  
- **Source:** [Kaggle – Deepfake Detection DFD](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)  
- **Total Files:** ~5000+ videos  
- **Classes:** `Deepfake`, `Original`  
- **Format:** `.mp4`  
- **Description:** Contains original and manipulated videos for deepfake detection research.  

### 2️⃣ Audio Dataset – Deepfake Audio Dataset  
- **Source:** [Hugging Face – Deepfake Audio Dataset](https://huggingface.co/datasets/Hemg/Deepfake-Audio-Dataset)  
- **Total Files:** 1000+ audio clips  
- **Classes:** `Deepfake`, `Original`  
- **Format:** `.wav`  
- **Description:** Synthetic and genuine speech clips to train audio deepfake detection models.  

---

## III. 🛠 Methodology
The system uses a **dual-pipeline approach**: one for audio, one for video.

### 🔹 Audio Detection Pipeline
1. Load audio file and resample to 22050 Hz.  
2. Extract **log-mel spectrogram** using Librosa.  
3. Resize spectrogram to fixed dimensions.  
4. Feed into a **PyTorch CNN model** for classification.  
5. Output: `Original` or `Deepfake` with confidence score.  

### 🔹 Video Detection Pipeline
1. Extract frames from video using OpenCV.  
2. Resize frames to 64×64 pixels and normalize pixel values.  
3. Select fixed number of frames to match model input.  
4. Feed frames into **Conv3D model (TensorFlow/Keras)**.  
5. Output: `Original` or `Deepfake` with confidence score.  

### 🔹 Overfitting Prevention & Accuracy Improvement
- Data augmentation for audio and video  
- Early stopping and checkpointing  
- Standardization of input length and dimensions  

---

## IV. 💻 How to Use

```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
