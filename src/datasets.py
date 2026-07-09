"""
MVTec AD dataset loader and utilities.

Handles loading images and ground-truth masks for the autoencoder pipeline.
For PatchCore, we use anomalib's built-in MVTecAD datamodule instead.

Dataset structure (per category):
  train/good/         - defect-free training images
  test/good/          - defect-free test images
  test/<defect>/      - defective test images
  ground_truth/<defect>/ - binary masks (white = defect)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Standard ImageNet normalization — same as what PatchCore uses,
# so results are comparable.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image resolution — MVTec AD standard benchmark size
IMG_SIZE = 256

# Categories used in this project (subset of 15 for variety)
CATEGORIES = ["bottle", "hazelnut", "carpet"]


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transforms for the autoencoder pipeline.

    We normalize to [0, 1] range (not ImageNet stats) because the autoencoder
    uses Sigmoid output. This differs from PatchCore which uses ImageNet stats.

    Args:
        train: If True, applies training augmentations (currently none —
               anomaly detection typically avoids augmentation to preserve
               normal appearance statistics).
    """
    transform_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ]
    return transforms.Compose(transform_list)


def get_mask_transform() -> transforms.Compose:
    """Transform for ground-truth masks: resize and binarize."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec AD (used by the autoencoder pipeline).

    In train mode: loads only defect-free images from train/good/.
    In test mode: loads all test images with labels and masks.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Path to MVTec AD root (e.g., data/mvtec_ad)
            category: Category name (e.g., 'bottle')
            split: 'train' or 'test'
            transform: Image transform (default: get_transforms)
            mask_transform: Mask transform (default: get_mask_transform)
        """
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform or get_transforms(train=(split == "train"))
        self.mask_transform = mask_transform or get_mask_transform()

        self.samples: list[dict] = []
        self._load_samples()

    def _load_samples(self):
        """Scan directory structure and build sample list."""
        category_dir = self.root / self.category

        if self.split == "train":
            # Training: only defect-free images
            good_dir = category_dir / "train" / "good"
            if not good_dir.exists():
                raise FileNotFoundError(
                    f"Training directory not found: {good_dir}\n"
                    f"Run scripts/download_data.py first."
                )
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path": None,
                    "label": 0,  # 0 = normal
                    "defect_type": "good",
                })

        elif self.split == "test":
            # Test: all subdirectories under test/
            test_dir = category_dir / "test"
            gt_dir = category_dir / "ground_truth"

            if not test_dir.exists():
                raise FileNotFoundError(
                    f"Test directory not found: {test_dir}\n"
                    f"Run scripts/download_data.py first."
                )

            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                is_normal = defect_type == "good"

                for img_path in sorted(defect_dir.glob("*.png")):
                    # Find corresponding ground-truth mask
                    mask_path = None
                    if not is_normal:
                        # Mask filename: same stem + _mask.png
                        mask_name = img_path.stem + "_mask.png"
                        candidate = gt_dir / defect_type / mask_name
                        if candidate.exists():
                            mask_path = str(candidate)

                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": mask_path,
                        "label": 0 if is_normal else 1,
                        "defect_type": defect_type,
                    })
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train' or 'test'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          - image: transformed image tensor [3, 256, 256]
          - mask: ground-truth mask tensor [1, 256, 256] (zeros if normal)
          - label: 0 (normal) or 1 (anomalous)
          - defect_type: string name of defect type
          - image_path: original file path
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        # Load mask (or create zero mask for normal samples)
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_transform(mask)
            # Binarize: threshold at 0.5
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)

        return {
            "image": image,
            "mask": mask,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
        }


def get_dataloaders(
    category: str,
    data_root: str = "data/mvtec_ad",
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for a category.

    Validation set is split from training data (defect-free images only).

    Args:
        category: MVTec AD category name
        data_root: Path to dataset root
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of training data used for validation

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Training + validation (from train/good/)
    full_train = MVTecDataset(data_root, category, split="train")
    val_size = max(1, int(len(full_train) * val_split))
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Test set
    test_dataset = MVTecDataset(data_root, category, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_category_info(data_root: str = "data/mvtec_ad") -> dict:
    """
    Get information about available categories and their defect types.

    Returns dict: {category: {"train_count": int, "test_count": int,
                              "defect_types": list[str]}}
    """
    root = Path(data_root)
    info = {}

    for category in CATEGORIES:
        cat_dir = root / category
        if not cat_dir.exists():
            continue

        # Count training images
        train_good = cat_dir / "train" / "good"
        train_count = len(list(train_good.glob("*.png"))) if train_good.exists() else 0

        # Count test images and defect types
        test_dir = cat_dir / "test"
        test_count = 0
        defect_types = []
        if test_dir.exists():
            for subdir in sorted(test_dir.iterdir()):
                if subdir.is_dir():
                    count = len(list(subdir.glob("*.png")))
                    test_count += count
                    defect_types.append(f"{subdir.name} ({count})")

        info[category] = {
            "train_count": train_count,
            "test_count": test_count,
            "defect_types": defect_types,
        }

    return info


if __name__ == "__main__":
    # Quick test
    info = get_category_info()
    for cat, details in info.items():
        print(f"\n{cat}:")
        print(f"  Train (good): {details['train_count']}")
        print(f"  Test (total): {details['test_count']}")
        print(f"  Defect types: {', '.join(details['defect_types'])}")
