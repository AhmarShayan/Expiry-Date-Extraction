import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ProductDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self._load_labels(label_file)

    def _load_labels(self, label_file):
        data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                img = parts[0]
                label = parts[1]
                bbox = list(map(int, parts[2:6]))
                data.append((img, label, bbox))
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label, bbox = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(bbox)