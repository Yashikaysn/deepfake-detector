import gradio as gr
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from huggingface_hub import hf_hub_download

# ── Model definition (must match training exactly) ───────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = hf_hub_download(repo_id="Yashikaysn29/deepshield", filename="best_model.pth")
model = DeepfakeDetector()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict single frame ─────────────────────────────────────────────────────
def predict_image(pil_img):
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        prob = model(tensor).item()
    real_conf = round(prob * 100, 2)
    fake_conf = round((1 - prob) * 100, 2)
    label = "🔴 FAKE" if prob < 0.5 else "🟢 REAL"
    return label, fake_conf, real_conf

# ── Image handler ─────────────────────────────────────────────────────────────
def analyze_image(img):
    if img is None:
        return "No image provided", "", ""
    pil_img = Image.fromarray(img).convert("RGB")
    label, fake_pct, real_pct = predict_image(pil_img)
    return label, f"{fake_pct}%", f"{real_pct}%"

# ── Video handler ─────────────────────────────────────────────────────────────
def analyze_video(video_path):
    if video_path is None:
        return "No video provided", "", ""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_at = np.linspace(0, total - 1, min(16, total), dtype=int)
    fake_scores = []
    for idx in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, fake_pct, _ = predict_image(pil_img)
        fake_scores.append(fake_pct)
    cap.release()
    if not fake_scores:
        return "Could not read video", "", ""
    avg_fake = round(sum(fake_scores) / len(fake_scores), 2)
    avg_real = round(100 - avg_fake, 2)
    label = "🔴 FAKE" if avg_fake > 50 else "🟢 REAL"
    return label, f"{avg_fake}%", f"{avg_real}%"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="DeepShield — Deepfake Detector") as demo:
    gr.Markdown("""
    # 🛡️ DeepShield — Real-Time Deepfake Detector
    **Powered by EfficientNet-B4 | ~99% Validation Accuracy**
    Upload an image or video to check if it's real or AI-generated/deepfaked.
    """)
    with gr.Tabs():
        with gr.Tab("🖼️ Image"):
            with gr.Row():
                img_input = gr.Image(label="Upload Image")
                with gr.Column():
                    img_label = gr.Textbox(label="Verdict")
                    img_fake  = gr.Textbox(label="Fake Confidence")
                    img_real  = gr.Textbox(label="Real Confidence")
            gr.Button("Analyze Image", variant="primary").click(
                analyze_image, inputs=img_input,
                outputs=[img_label, img_fake, img_real]
            )
        with gr.Tab("🎥 Video"):
            with gr.Row():
                vid_input = gr.Video(label="Upload Video")
                with gr.Column():
                    vid_label = gr.Textbox(label="Verdict")
                    vid_fake  = gr.Textbox(label="Fake Confidence (avg)")
                    vid_real  = gr.Textbox(label="Real Confidence (avg)")
            gr.Button("Analyze Video", variant="primary").click(
                analyze_video, inputs=vid_input,
                outputs=[vid_label, vid_fake, vid_real]
            )
    gr.Markdown("---\n*Built by Yashika Saxena | B.Tech AI & ML, ITM Gwalior*")

demo.launch()