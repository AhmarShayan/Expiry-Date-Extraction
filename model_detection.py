import torch.nn as nn
import torchvision.models as models

class ExpiryProductDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 4)
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        return class_logits, bbox_preds