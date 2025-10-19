import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class FloorplanDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, target_transform=None):
        """
        image_dir: folder with input floorplan images
        mask_dir: folder with ground truth masks (same filename as images)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.int64)  # ensure integer mask
        else:
            mask = np.zeros((image.height, image.width), dtype=np.int64)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask)

        return image, mask
