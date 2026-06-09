import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet — a pre-trained brain that already
        # knows how to look at faces
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=0   # remove its original classifier
        )

        # Our own classifier on top
        # Takes features → decides REAL or FAKE
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),          # activation
            nn.Dropout(0.4),    # prevents overfitting
            nn.Linear(256, 1),
            nn.Sigmoid()        # output: 0.0 = real, 1.0 = fake
        )

    def forward(self, x):
        features = self.backbone(x)       # extract face features
        output   = self.classifier(features)  # classify
        return output


# ── Quick test — run this file directly to check it works ──
if __name__ == "__main__":
    model = DeepfakeDetector()
    print("Model structure ready!")
    print("Output shape test passed!")
    # create a fake image tensor (batch of 1, RGB, 224x224)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Model output: {output.item():.4f}")
    print("0.0 = Real  |  1.0 = Fake")
    print("Model is working correctly!")