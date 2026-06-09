import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import DeepfakeDetector

# ── 1. Image transformations ──────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 2. Load dataset ───────────────────────────────────────
# Change this path to where your dataset is!
DATASET_PATH = "data/deepfakedata/real_vs_fake/real-vs-fake"

train_data = datasets.ImageFolder(DATASET_PATH + "/train", transform=train_transform)
val_data   = datasets.ImageFolder(DATASET_PATH + "/valid", transform=val_transform)

# Use only 5000 images for faster training
from torch.utils.data import Subset
train_data = Subset(train_data, range(5000))
val_data   = Subset(val_data,   range(1000))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)
print(f"Training images:   {len(train_data)}")
print(f"Validation images: {len(val_data)}")
print(f"Classes: {train_data.dataset.classes}")

# ── 3. Setup model ────────────────────────────────────────
model     = DeepfakeDetector()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn   = nn.BCELoss()

# ── 4. Training loop ──────────────────────────────────────
best_val_acc = 0

for epoch in range(10):
    # ── Training ──
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}...", flush=True)
        labels = labels.float().unsqueeze(1)
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((preds > 0.5).float() == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100

    # ── Validation ──
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.float().unsqueeze(1)
            preds = model(images)
            val_correct += ((preds > 0.5).float() == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total * 100
    print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%", flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/best_model.pth")
        print(f"  ✓ Best model saved! Val Acc: {val_acc:.1f}%", flush=True)

print(f"\nTraining complete! Best accuracy: {best_val_acc:.1f}%")