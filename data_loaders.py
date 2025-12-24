# filepath: c:\CODE\3d_BVnet\data_loaders.py
"""
Data loading utilities for 3D medical image segmentation.

This module provides unified data loading interfaces for different datasets:
- Task06_Lung (Medical Decathlon)
- OrganMNIST3D (MedMNIST)
- Toy data for testing

Usage:
    from data_loaders import create_dataloaders
    
    train_loader, valid_loader = create_dataloaders(
        data_name='toy_data_3d',
        batch_size=4,
        quan_type=torch.float32
    )
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from skimage.filters import threshold_otsu

# MONAI imports
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlipd,
    EnsureTyped,
)

# MedMNIST imports
try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False


# ==================== Utility Functions ====================

def otsu_3d(x, quan_type=np.float32):
    """
    Apply Otsu thresholding to 3D volume.
    
    Args:
        x: torch.Tensor or np.ndarray [1, D, H, W]
        quan_type: Output data type
        
    Returns:
        torch.Tensor [1, D, H, W] binary mask
    """
    if hasattr(x, "is_cuda"):
        x_np = x.cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(x, dtype=np.float32)
    
    t = threshold_otsu(x_np)
    y = (x_np > t).astype(np.float32)
    return torch.from_numpy(y)


def resize_3d(x, size, mode="trilinear"):
    """
    Resize 3D volume using interpolation.
    
    Args:
        x: torch.Tensor or np.ndarray [1, D, H, W]
        size: tuple (D_new, H_new, W_new)
        mode: interpolation mode ('trilinear', 'nearest', etc.)
        
    Returns:
        torch.Tensor [1, D_new, H_new, W_new]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    
    x = x.unsqueeze(0)  # Add batch dimension
    x = F.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners=False if mode != "nearest" else None
    )
    return x.squeeze(0)


# ==================== Dataset Classes ====================

class OtsuResizeDataset(Dataset):
    """
    Dataset that resizes images and generates binary labels using Otsu thresholding.
    
    Useful for creating segmentation targets from classification datasets.
    """
    
    def __init__(self, base_dataset, target_size, quan_type=torch.float32):
        """
        Args:
            base_dataset: Base dataset to wrap
            target_size: Target size tuple (D, H, W)
            quan_type: Output tensor dtype
        """
        self.base = base_dataset
        self.target_size = target_size
        self.quan_type = quan_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]  # Ignore original label
        
        # Resize image
        x = resize_3d(x, self.target_size, mode="trilinear")
        
        # Generate label using Otsu thresholding
        y = otsu_3d(x)
        
        x = x.to(dtype=self.quan_type)
        y = y.to(dtype=self.quan_type)
        
        return x, y


class BFloat16Loader:
    """Wrapper loader that converts batches to bfloat16."""
    
    def __init__(self, loader):
        self.loader = loader
    
    def __iter__(self):
        for batch in self.loader:
            yield [x.to(dtype=torch.bfloat16) for x in batch]
    
    def __len__(self):
        return len(self.loader)


# ==================== Transform Factories ====================

def get_task06_lung_transforms(resize_size=(120, 120, 120)):
    """
    Get MONAI transforms for Task06_Lung dataset.
    
    Args:
        resize_size: Target spatial size (D, H, W)
        
    Returns:
        Compose: MONAI transform pipeline
    """
    resize_x, resize_y, resize_z = resize_size
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024, a_max=3000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(
            keys=["image", "label"],
            spatial_size=[resize_x, resize_y, resize_z],
            mode=("trilinear", "nearest")
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image", "label"]),
    ])


# ==================== Dataset Loading Functions ====================

def load_task06_lung(
    data_dir,
    batch_size,
    resize_size=(120, 120, 120),
    train_split=48,
    num_workers=4,
    cache_rate=1.0
):
    """
    Load Task06_Lung dataset from Medical Decathlon.
    
    Args:
        data_dir: Path to Task06_Lung directory
        batch_size: Batch size for DataLoader
        resize_size: Target size for resizing (D, H, W)
        train_split: Number of samples for training
        num_workers: Number of workers for DataLoader
        cache_rate: Cache rate for CacheDataset (0-1)
    
    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Collect file paths
    images_tr_dir = os.path.join(data_dir, "imagesTr")
    images_ts_dir = os.path.join(data_dir, "imagesTs")
    labels_dir = os.path.join(data_dir, "labelsTr")
    
    train_images = sorted([
        os.path.join(images_tr_dir, f) 
        for f in os.listdir(images_tr_dir) 
        if f.endswith(".nii.gz")
    ])
    test_images = sorted([
        os.path.join(images_ts_dir, f) 
        for f in os.listdir(images_ts_dir) 
        if f.endswith(".nii.gz")
    ])
    labels = sorted([
        os.path.join(labels_dir, f) 
        for f in os.listdir(labels_dir) 
        if f.endswith(".nii.gz")
    ])

    print(f'[Task06_Lung] Train images: {len(train_images)}, Labels: {len(labels)}, Test images: {len(test_images)}')

    # Create data dictionaries
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, labels)]
    transforms = get_task06_lung_transforms(resize_size)

    # Create datasets
    train_ds = CacheDataset(
        data=data_dicts[:train_split], 
        transform=transforms, 
        cache_rate=cache_rate
    )
    valid_ds = CacheDataset(
        data=data_dicts[train_split:], 
        transform=transforms, 
        cache_rate=cache_rate
    )
    
    # Create data loaders
    train_loader = MonaiDataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    valid_loader = MonaiDataLoader(
        valid_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, valid_loader


def load_organmnist3d(
    batch_size,
    target_size=(64, 64, 64),
    quan_type=torch.float32,
    train_ratio=0.8,
    seed=42,
    num_workers=4
):
    """
    Load OrganMNIST3D dataset with Otsu thresholding for segmentation.
    
    Args:
        batch_size: Batch size for DataLoader
        target_size: Target size for resizing (D, H, W)
        quan_type: Data type for tensors
        train_ratio: Ratio of training data
        seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
    
    Returns:
        tuple: (train_loader, valid_loader)
    """
    if not MEDMNIST_AVAILABLE:
        raise ImportError("medmnist package is required. Install with: pip install medmnist")
    
    data_flag = "organmnist3d"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    
    # Load full dataset
    full_dataset = DataClass(split='train', download=True, size=64)

    # Split into train/valid
    total_len = len(full_dataset)
    train_len = int(train_ratio * total_len)
    valid_len = total_len - train_len
    
    train_base, valid_base = random_split(
        full_dataset,
        [train_len, valid_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # Wrap with Otsu thresholding dataset
    train_dataset = OtsuResizeDataset(train_base, target_size, quan_type=quan_type)
    valid_dataset = OtsuResizeDataset(valid_base, target_size, quan_type=quan_type)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f'[OrganMNIST3D] Train: {train_len}, Valid: {valid_len}, Target size: {target_size}')

    return train_loader, valid_loader


def load_toy_data_3d(
    batch_size,
    quan_type=torch.float32,
    data_dir="toy_data_3d"
):
    """
    Load toy 3D data for testing.
    
    Args:
        batch_size: Batch size for DataLoader
        quan_type: Data type for tensors
        data_dir: Directory containing toy data
    
    Returns:
        tuple: (train_loader, valid_loader, input_shape)
    """
    from dataset import get_train_data
    
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    
    # Convert to tensors with channel-first format [B, C, D, H, W]
    train_tensor = torch.tensor(train, dtype=quan_type).permute(0, 4, 1, 2, 3)
    train_targets_tensor = torch.tensor(train_targets, dtype=quan_type).permute(0, 4, 1, 2, 3)
    valid_tensor = torch.tensor(valid, dtype=quan_type).permute(0, 4, 1, 2, 3)
    valid_targets_tensor = torch.tensor(valid_targets, dtype=quan_type).permute(0, 4, 1, 2, 3)
    
    train_dataset = TensorDataset(train_tensor, train_targets_tensor)
    valid_dataset = TensorDataset(valid_tensor, valid_targets_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    print(f'[Toy3D] Train: {len(train)}, Valid: {len(valid)}, Shape: {train.shape}')
    
    return train_loader, valid_loader, input_shape


# ==================== Main Factory Function ====================

def create_dataloaders(
    data_name,
    batch_size,
    quan_type=torch.float32,
    **kwargs
):
    """
    Factory function to create data loaders based on dataset name.
    
    Args:
        data_name: Name of the dataset:
            - 'Task06_Lung': Medical Decathlon lung CT
            - 'organmnist3d': MedMNIST 3D organ classification
            - 'toy_data_3d': Synthetic test data
        batch_size: Batch size for DataLoader
        quan_type: Data type for tensors
        **kwargs: Additional arguments passed to specific loaders
    
    Returns:
        tuple: (train_loader, valid_loader) or (train_loader, valid_loader, input_shape)
    
    Example:
        >>> train_loader, valid_loader = create_dataloaders(
        ...     data_name='toy_data_3d',
        ...     batch_size=4
        ... )
    """
    print(f'[DataLoader] Loading dataset: {data_name}')
    
    if data_name == 'Task06_Lung':
        data_dir = kwargs.get('data_dir', "../Task06_Lung/")
        resize_size = kwargs.get('resize_size', (120, 120, 120))
        train_split = kwargs.get('train_split', 48)
        num_workers = kwargs.get('num_workers', 4)
        
        return load_task06_lung(
            data_dir=data_dir,
            batch_size=batch_size,
            resize_size=resize_size,
            train_split=train_split,
            num_workers=num_workers
        )
    
    elif data_name == 'organmnist3d':
        target_size = kwargs.get('target_size', (64, 64, 64))
        train_ratio = kwargs.get('train_ratio', 0.8)
        seed = kwargs.get('seed', 42)
        num_workers = kwargs.get('num_workers', 4)
        
        return load_organmnist3d(
            batch_size=batch_size,
            target_size=target_size,
            quan_type=quan_type,
            train_ratio=train_ratio,
            seed=seed,
            num_workers=num_workers
        )
    
    elif data_name == 'toy_data_3d':
        data_dir = kwargs.get('data_dir', 'toy_data_3d')
        return load_toy_data_3d(
            batch_size=batch_size,
            quan_type=quan_type,
            data_dir=data_dir
        )
    
    else:
        raise ValueError(
            f"Unknown dataset: {data_name}. "
            f"Available: 'Task06_Lung', 'organmnist3d', 'toy_data_3d'"
        )


# ==================== Testing ====================

if __name__ == "__main__":
    # Test toy data loading
    print("Testing toy data loading...")
    train_loader, valid_loader, input_shape = create_dataloaders(
        data_name='toy_data_3d',
        batch_size=2
    )
    
    for batch_x, batch_y in train_loader:
        print(f"Train batch - X: {batch_x.shape}, Y: {batch_y.shape}")
        break
    
    for batch_x, batch_y in valid_loader:
        print(f"Valid batch - X: {batch_x.shape}, Y: {batch_y.shape}")
        break
    
    print("Data loading test complete!")
