import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ProductDataset
from model_detection import ExpiryProductDetector
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ProductDataset(image_dir="data/test_images", label_file="data/test_labels.csv", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ExpiryProductDetector(num_classes=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
bbox_error = []

total = len(test_loader)

with torch.no_grad():
    for images, labels, bboxes in test_loader:
        outputs, preds = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == torch.tensor([int(l) for l in labels])).sum().item()
        error = torch.abs(preds - bboxes.float()).mean().item()
        bbox_error.append(error)

print(f"Classification Accuracy: {correct / total * 100:.2f}%")
print(f"Bounding Box Error: {np.mean(bbox_error):.4f}")