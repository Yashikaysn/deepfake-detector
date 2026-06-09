# 🛡️ DeepShield — Real-Time Deepfake Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.7%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> AI-powered deepfake detection system using EfficientNet-B4, trained on 100,000 real and fake face images.

---

## 🎯 What It Does
Detects whether a face image is **real** or **AI-generated (deepfake)** with **99.7% validation accuracy**.

Upload any portrait photo → automatic face detection → instant result with confidence score.

---

## 🧠 Model Architecture
- **Backbone:** EfficientNet-B4 (pretrained on ImageNet)
- **Face Detection:** MTCNN
- **Classifier:** Custom head (Linear → ReLU → Dropout → Sigmoid)
- **Training:** 10 epochs, AdamW optimizer, BCELoss
- **Dataset:** 140K Real and Fake Faces (Kaggle)

---

## 📊 Results

| Metric | Score |
|---|---|
| Validation Accuracy | 99.7% |
| Training Images | 100,000 |
| Validation Images | 20,000 |
| Training Platform | Google Colab (Tesla T4 GPU) |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Yashikaysn/deepfake-detector.git
cd deepfake-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 📁 Project Structure
deepfake-detector/
├── app/
│   └── app.py              # Streamlit web app
├── model/
│   ├── model.py            # EfficientNet-B4 architecture
│   └── train.py            # Training pipeline
├── preprocessing/
│   ├── extract_frames.py   # Video to frames
│   └── face_crop.py        # Face detection & cropping
└── README.md

---

## 🛠️ Tech Stack
- Python 3.10
- PyTorch + torchvision
- EfficientNet-B4 (timm)
- MTCNN (facenet-pytorch)
- Streamlit
- Google Colab (Tesla T4 GPU)

---

## 👩‍💻 Author
**Yashika Saxena**
B.Tech AI & ML — Institute of Technology and Management, Gwalior
