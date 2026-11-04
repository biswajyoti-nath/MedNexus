import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, patient_list, transform=None):
        self.data_dir = data_dir
        self.patient_list = patient_list
        self.transform = transform
        
        self.images = []
        self.masks = []
        
        for patient in patient_list:
            patient_dir = os.path.join(data_dir, patient)
            masks = sorted(glob.glob(os.path.join(patient_dir, '*_mask.tif')))
            for m in masks:
                img_path = m.replace('_mask.tif', '.tif')
                self.images.append(img_path)
                self.masks.append(m)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Normalize mask to 0 and 1
        mask = mask / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        else:
            mask = mask.unsqueeze(0).float()
            image = image.float() / 255.0
            
        label = torch.tensor(1.0 if mask.max() > 0 else 0.0, dtype=torch.float32)
        
        return image, mask, label, img_path

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        ToTensorV2()
    ])
