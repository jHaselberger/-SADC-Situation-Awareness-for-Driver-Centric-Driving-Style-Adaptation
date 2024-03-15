
import os
import torch

from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms, labels=None, root_dir="/root/sadc/data/01_Images"): 
        self.X = [os.path.join(root_dir, p) for p in image_paths]
        self.y = labels
        self.has_labels = False if labels is None else True
        self.transforms = transforms 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.X[idx], mode='r').convert(mode="RGB")
        if self.transforms:
            img = self.transforms(img)
        return {"X": img, "y": self.y[idx]} if self.has_labels else {"X": img}