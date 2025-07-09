import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ProductDataset
from model_detection import ExpiryProductDetector
from model_dmy import ResNet45

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ProductDataset(image_dir="data/images", label_file="data/labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

model = ExpiryProductDetector(num_classes=5)
dmy_model = ResNet45(num_classes=10)  # for digit recognition (0-9)

criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.train()
for epoch in range(20):
    for images, labels, bboxes in dataloader:
        optimizer.zero_grad()
        logits, preds = model(images)
        loss_cls = criterion_cls(logits, torch.tensor([int(l) for l in labels]))
        loss_bbox = criterion_bbox(preds, bboxes.float())
        loss = loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed with loss: {loss.item():.4f}")