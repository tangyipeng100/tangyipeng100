"""
Small-sample dataset utilities with strong augmentation.
Supports few-shot sampling (N-shot per class).
"""
import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SmallSampleDataset(Dataset):
    """
    Image classification dataset with few-shot sampling support.
    Directory structure:
        root/
          class_a/img1.jpg, img2.jpg, ...
          class_b/img1.jpg, ...
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(
        self,
        root: str,
        transform=None,
        shots_per_class: Optional[int] = None,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.transform = transform
        self.shots_per_class = shots_per_class

        self.classes = sorted([
            d.name for d in self.root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = self._load_samples(seed)

    def _load_samples(self, seed: int):
        rng = random.Random(seed)
        samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            imgs = [
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in self.EXTENSIONS
            ]
            if self.shots_per_class is not None:
                imgs = rng.sample(imgs, min(self.shots_per_class, len(imgs)))
            samples.extend((str(p), self.class_to_idx[cls]) for p in imgs)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, augment: bool, is_train: bool):
    """Build torchvision transform pipeline."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Build DataLoader from config dict."""
    is_train = split == 'train'
    root = cfg['data'][f'{split}_dir']
    shots = cfg['data'].get('shots_per_class') if is_train else None

    transform = build_transforms(
        image_size=cfg['data']['image_size'],
        augment=cfg['data'].get('augment', True),
        is_train=is_train,
    )

    dataset = SmallSampleDataset(
        root=root,
        transform=transform,
        shots_per_class=shots,
    )

    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=is_train,
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=is_train and len(dataset) > cfg['training']['batch_size'],
    )
