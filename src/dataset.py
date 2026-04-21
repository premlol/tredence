"""
src/dataset.py
--------------
CIFAR-10 data loading with standard augmentation for training.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    """
    Download (if needed) and return CIFAR-10 train/test DataLoaders.

    Training augmentation:
        RandomHorizontalFlip + RandomCrop(32, padding=4) + ColorJitter → Normalize

    Returns
    -------
    train_loader, test_loader, class_names
    """

    # ── Normalisation stats (pre-computed on CIFAR-10) ───────────────────────
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    class_names = train_set.classes
    return train_loader, test_loader, class_names
