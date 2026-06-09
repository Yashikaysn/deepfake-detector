import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield — Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Hero title */
    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .hero-subtitle {
        text-align: center;
        color: #aaaacc;
        font-size: 1.1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    /* Upload box */
    .upload-box {
        border: 2px dashed #0072ff;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(0, 114, 255, 0.05);
        margin-bottom: 1.5rem;
    }

    /* Result cards */
    .result-real {
        background: linear-gradient(135deg, #0f9b58, #00c851);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-fake {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .result-confidence {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.3rem;
    }

    /* Stats row */
    .stat-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert {display: none;}

    /* Image captions */
    .caption {
        text-align: center;
        color: #aaaacc;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ── Face finder ───────────────────────────────────────────
mtcnn = MTCNN(margin=20, keep_all=False, image_size=224)

# ── Model ─────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

@st.cache_resource
def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(
        "model/best_model.pth",
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Hero Section ──────────────────────────────────────────
st.markdown('<p class="hero-title">🛡️ DeepShield</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Deepfake Detection — Trained on 100,000 faces with 99.7% accuracy</p>', unsafe_allow_html=True)

# ── Stats Bar ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size:1.5rem;font-weight:800;color:#00c6ff;">99.7%</div>
        <div style="color:#aaaacc;font-size:0.85rem;">Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size:1.5rem;font-weight:800;color:#00c6ff;">100K+</div>
        <div style="color:#aaaacc;font-size:0.85rem;">Training Images</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size:1.5rem;font-weight:800;color:#00c6ff;">EfficientNet-B4</div>
        <div style="color:#aaaacc;font-size:0.85rem;">Model Architecture</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Upload Section ────────────────────────────────────────
st.markdown("### 📤 Upload a Face Image")
st.markdown("Upload any portrait photo — we'll automatically detect and analyse the face.")

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

model = load_model()

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)
        st.markdown('<p class="caption">📷 Original Image</p>', unsafe_allow_html=True)

    # Face detection
    with st.spinner("🔍 Detecting face..."):
        face_tensor = mtcnn(image)

    if face_tensor is not None:
        face_np  = face_tensor.permute(1, 2, 0).numpy()
        face_np  = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        face_pil = Image.fromarray(face_np)

        with col2:
            st.image(face_pil, use_column_width=True)
            st.markdown('<p class="caption">🎯 Detected Face</p>', unsafe_allow_html=True)

        # Analysis
        with st.spinner("🧠 Analysing with AI..."):
            tensor = transform(face_pil).unsqueeze(0)
            with torch.no_grad():
                score = model(tensor).item()

        st.markdown("<br>", unsafe_allow_html=True)

        # Result
        if score > 0.5:
            confidence = score * 100
            st.markdown(f"""
            <div class="result-real">
                <p class="result-label">✅ REAL FACE</p>
                <p class="result-confidence">This appears to be an authentic human face</p>
                <p style="font-size:2rem;font-weight:800;color:white;margin-top:0.5rem;">{confidence:.1f}% Confident</p>
            </div>""", unsafe_allow_html=True)
        else:
            confidence = (1 - score) * 100
            st.markdown(f"""
            <div class="result-fake">
                <p class="result-label">🚨 FAKE FACE</p>
                <p class="result-confidence">This appears to be an AI-generated face</p>
                <p style="font-size:2rem;font-weight:800;color:white;margin-top:0.5rem;">{confidence:.1f}% Confident</p>
            </div>""", unsafe_allow_html=True)

        # Progress bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Detection Score**")
        st.progress(score)
        st.caption(f"Score: {score:.4f}  |  0.0 = Fake  →  1.0 = Real")

    else:
        st.warning("⚠️ No face detected! Please upload a clear portrait photo with a visible face.")

st.divider()
st.markdown('<p style="text-align:center;color:#555577;font-size:0.8rem;">Built with EfficientNet-B4 + PyTorch | Final Year Project — ITM Gwalior</p>', unsafe_allow_html=True)