# preprocess_3d_medical_segmentation.py
import os

from monai.data import Dataset, DataLoader, CacheDataset
from monai.config import print_config
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import nibabel as nib
from collections import Counter

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImage,
    LoadImaged,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd, 
    EnsureTyped,
    Resized
)
from monai.utils import set_determinism

print_config()


# print_config()

data_dir = "../Task06_Lung/"

import os
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.startswith("._"):
            os.remove(os.path.join(root, f))
            print("Deleted:", f)
            
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])

num_total = len(image_class)
print(f"Total image count: {num_total}")
# print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")
    

images = sorted([os.path.join(data_dir, "imagesTr", f) for f in os.listdir(os.path.join(data_dir, "imagesTr")) if f.endswith(".nii.gz")])
labels = sorted([os.path.join(data_dir, "labelsTr", f) for f in os.listdir(os.path.join(data_dir, "labelsTr")) if f.endswith(".nii.gz")])

x = os.listdir(os.path.join(data_dir, "imagesTr"))
# print(x)
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]


# ---------------------------
# 2. 定义训练集预处理 pipeline
# ---------------------------
train_transforms = Compose([
    # --- 加载和添加通道 ---
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),

    # --- 统一方向 & 体素间距 ---
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest")  # 图像双线性，标签最近邻
    ),

    # --- 强度归一化 (CT/MRI 不同策略) ---
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,  # 例如 CT HU 值
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    Resized(
            keys=["image", "label"],
            spatial_size=[120,120,120],
            mode=("trilinear", "nearest")  # 图像连续，标签最近邻
    ),

    # --- 前景裁剪 (去掉大面积黑背景) ---
    # CropForegroundd(keys=["image", "label"], source_key="image"),

    # --- 随机裁 patch ---
    # RandSpatialCropd(
    #     keys=["image", "label"],
    #     roi_size=(128, 128, 128),
    #     random_center=True,
    #     random_size=False
    # ),

    # --- 数据增强 ---
    # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    # RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

    # --- 转换为 tensor ---
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

batch = next(iter(train_loader))
print("Image shape:", batch["image"].shape)
print("Label shape:", batch["label"].shape)