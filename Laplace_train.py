import torch
# from laplace import Laplace
from laplace import Laplace
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset, random_split
import time
import sys
import pandas as pd
from sklearn.utils import Bunch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import os
from torch.autograd import Variable
import time
import pickle
from tqdm import tqdm
import copy
from skimage.filters import threshold_otsu
import math, time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_train_data
from model import get_model
from utils import AnnealingCallback, ex

from monai.metrics import DiceMetric
import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.config import print_config
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import nibabel as nib
from collections import Counter

import medmnist
from medmnist import INFO

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
    Resized,
    RandFlipd
)
from monai.utils import set_determinism

import numpy as np
import nibabel as nib
import glob

@torch.no_grad()
def predict_cpu(dataloader, model, laplace=False):
    py = []

    idx = 0
    for x, _ in dataloader:
        time_start = time.time()
        if laplace:
#             print('idx', idx)
            py.append(model(x))
            
            time_end = time.time()
            print('idx', idx, 'time', time_end - time_start)
            idx += 1
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py).cpu().numpy()

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu().numpy()

@ex.capture
def schedule(epoch, initial_learning_rate, lr_decay_start_epoch):
    """Defines exponentially decaying learning rate."""
    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))

def otsu_3d(x, quan_type=np.float32):
    """
    x: torch.Tensor [1, D, H, W]
    return: torch.Tensor [1, D, H, W] binary mask
    """
    if hasattr(x, "is_cuda"):
        x_np = x.cpu().numpy().astype(quan_type)
    else:
        x_np = x.astype(quan_type)
    t = threshold_otsu(x_np)
    y = (x_np > t).astype(quan_type)
    return torch.from_numpy(y)
def resize_3d(x, size, mode):
    """
    x: torch.Tensor [1, D, H, W]
    size: (D_new, H_new, W_new)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.unsqueeze(0)
    x = F.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners = False if mode != "nearest" else None
    )
    return x.squeeze(0)
    
class OtsuResizeDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_size):
        self.base = base_dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]     # [1,28,28,28]

        # resize image
        x = resize_3d(
            x, self.target_size, mode="trilinear"
        )
        # x = x.permute(0, 2, 3, 4, 1)
        
        # generate label AFTER resize
        y = otsu_3d(x)

        return x, y

# class MyDataset(Dataset):
#     def __init__(self, data, dim, data_name=None):
#         self.data = data
    
#     def __getitem__(self, idx):
#         if 'Lung' in data_name:
#             x = self.data["image"]
#             y = self.data["label"]
#             if dim == 3:
#                 # batch_x = batch_x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
#                 x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                
#                 y = y.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
#                 y = y.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

#                 # print('dim', dim)
#             else:
#                 # batch_x = batch_x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
#                 y = y.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
#                 # print('dim', dim)
                
#         x = self.data[idx]
#         # 假设 x 的形状是 (C, D, H, W)
#         # 在这里直接 permute 成 (D, H, W, C)
#         x = x.permute(1, 2, 3, 0)
#         return dta
    
#     def __len__(self):
#         return len(self.data)

@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch, train_laplace_direct,
          kl_start_epoch, kl_alpha_increase_per_epoch, dim=2,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10):
    print('Starting training...')
    quan_type = torch.float32
    # quan_type = torch.float16
    # quan_type = torch.bfloat16
    """Trains a model (PyTorch version)."""
    print('loading data...')
    # Loads or creates training data.
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    print('getting model...')
    print('dim in train.py', dim )
    # Loads or creates model.
    model, checkpoint_path, kl_alpha = get_model(input_shape,
                                        scale_factor=len(train)/batch_size,
                                        weights_path=weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device, dtype=quan_type)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    criterion = nn.BCELoss()  # or another suitable loss
    # criterion = nn.BCEWithLogitsLoss()

    # train_loader = DataLoader(list(zip(train, train_targets)), batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(list(zip(valid, valid_targets)), batch_size=batch_size)
    
    #load data
    data_dir = "../Task06_Lung/"
    train_images = sorted([os.path.join(data_dir, "imagesTr", f) for f in os.listdir(os.path.join(data_dir, "imagesTr")) if f.endswith(".nii.gz")])
    test_images  = sorted([os.path.join(data_dir, "imagesTs", f) for f in os.listdir(os.path.join(data_dir, "imagesTs")) if f.endswith(".nii.gz")])
    
    labels = sorted([os.path.join(data_dir, "labelsTr", f) for f in os.listdir(os.path.join(data_dir, "labelsTr")) if f.endswith(".nii.gz")])

    print('train shape, label shape')
    print(np.shape(train_images), np.shape(labels), np.shape(test_images))
    
    x = os.listdir(os.path.join(data_dir, "imagesTr"))
    resize_x, resize_y, resize_z = 120, 120, 120
    # resize_x, resize_y, resize_z = 240, 240, 240
    train_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, labels)]
    # test_data_dicts  = [{"image": img, "label": lbl} for img, lbl in zip(test_images,  labels)]

    

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
            a_min=-1024, a_max=3000,  # 例如 CT HU 值
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        # --- 前景裁剪 (去掉大面积黑背景) ---
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        Resized(
                keys=["image", "label"],
                spatial_size=[resize_x, resize_y, resize_z],
                mode=("trilinear", "nearest")  # 图像连续，标签最近邻
        ),
    
        
    
        # --- 随机裁 patch ---
        # RandSpatialCropd(
        #     keys=["image", "label"],
        #     roi_size=(128, 128, 128),
        #     random_center=True,
        #     random_size=False
        # ),
    
        # --- 数据增强 ---
        RandFlipd(keys=["image", "label"], prob=1, spatial_axis=0),
        # RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    
        # --- 转换为 tensor ---
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # train_ds = CacheDataset(data=train_data_dicts[0:48], transform=train_transforms, cache_rate=1.0)
    # test_ds  = CacheDataset(data=train_data_dicts[48:60], transform=train_transforms, cache_rate=1.0)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    data_flag = "organmnist3d"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    
    # full_dataset = DataClass(
    #     split="train",
    #     download=True
    # )
    full_dataset = DataClass(split='train', download=True, size=64)

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    test_len  = total_len - train_len
    
    train_base, test_base = random_split(
        full_dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    target_size = (64, 64, 64)
    # target_size = (50,50,50)

    train_dataset = OtsuResizeDataset(train_base, target_size)
    test_dataset  = OtsuResizeDataset(test_base, target_size)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    valid_loader = test_loader
    
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
    print(x.min(), x.max())
    print(y.unique())

    # train_tensor = torch.tensor(train, dtype=quan_type).cuda()
    # train_targets_tensor = torch.tensor(train_targets, dtype=quan_type).cuda()
    # valid_tensor = torch.tensor(valid, dtype=quan_type).cuda()
    # valid_targets_tensor = torch.tensor(valid_targets, dtype=quan_type).cuda()

    # train_dataset = TensorDataset(train_tensor, train_targets_tensor)
    # valid_dataset = TensorDataset(valid_tensor, valid_targets_tensor)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    dice = DiceMetric(include_background=False, reduction="mean")
    train_dice_score = []
    val_dice_score = []
    time_vnet_s = time.time()
    if not train_laplace_direct:
        print('start')
        for epoch in range(initial_epoch, epochs):
            model.train()
            epoch_loss = 0
            train_dice = []
            for batch in train_loader:
                if isinstance(batch, dict):
                    batch_x = batch["image"]
                    batch_y = batch["label"]
                else:
                    batch_x = batch[0]
                    batch_y = batch[1]
                batch_x, batch_y = batch_x.to(device, dtype=quan_type), batch_y.to(device, dtype=quan_type)
                optimizer.zero_grad()
                # batch_x = batch_x.permute(0, 3, 1, 2)
                # if dim == 3:
                #     # batch_x = batch_x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                #     batch_x = batch_x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                    
                #     batch_y = batch_y.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                #     batch_y = batch_y.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
                #     if isinstance(batch, dict):
                #         batch_y = batch_y.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
                #     # print('dim', dim)
                # else:
                #     # batch_x = batch_x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                #     batch_y = batch_y.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                #     # print('dim', dim)
                # batch_y = batch_y.permute(0, 3, 1, 2)
                # print('batch_x.shape()################', batch_x.shape )
                # print('batch_y.shape()################', batch_y.shape )
                
                outputs = model(batch_x)
                # print('outputs size', outputs.size())
                # print('outputs', outputs.min() , outputs.max())
                # print('batch_y', batch_y.min() , batch_y.max())
                m = outputs.cpu().detach().numpy().mean()
                # print(m)

                if outputs.min() < 0 or outputs.max() > 1:
                    print("BAD OUTPUT RANGE:", outputs.min().item(), outputs.max().item())
                    exit()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                score = dice(outputs>m, batch_y).cpu().numpy()

                score = np.mean(score)
                train_dice.append(score)
            train_dice_score.append(np.mean(train_dice))
            # print(score)
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, train_dice: {np.mean(train_dice)}")

            # Validation
            model.eval()
            val_loss = 0
            val_dice = []
            with torch.no_grad():
                # for val_x, val_y in valid_loader:
                for val in valid_loader:
                    if isinstance(val, dict):
                        val_x = val["image"]
                        val_y = val["label"]
                    else:
                        val_x = batch[0]
                        val_y = batch[1]

                    val_x, val_y = val_x.to(device, dtype=quan_type), val_y.to(device, dtype=quan_type)
                    # print('val_x.type################', val_x.type() )
                    # print('val_y.type################', val_y.type() )
                    # val_x = val_x.permute(0, 3, 1, 2)
                    # if dim == 3:
                    #     # val_x = val_x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
                    #     val_x = val_x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                        
                    #     val_y = val_y.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, C, D, H, W)
                    #     val_y = val_y.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
                    #     if isinstance(val, dict):
                    #         val_y = val_y.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
                    #     # print('dim', dim)
                    # else:
                    #     # val_x = val_x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    #     val_y = val_y.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    #     # print('dim', dim)

                    val_outputs = model(val_x)
                    # print('val_outputs.type################', val_outputs.type() )
                    v_loss = criterion(val_outputs, val_y)
                    val_loss += v_loss.item()
                    
                    score = dice(val_outputs>m, val_y).cpu().numpy()
                    score = np.mean(score)
                    val_dice.append(score)
                    
            val_dice_score.append(np.mean(val_dice))
            val_loss /= len(valid_loader)
            print(f"Validation Loss: {val_loss:.4f}, Dice: {np.mean(val_dice)}")

            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print("Best model saved.")

            # Learning rate scheduling
            for param_group in optimizer.param_groups:
                param_group['lr'] = schedule(epoch, initial_learning_rate, lr_decay_start_epoch)

            # Annealing callback (if applicable)
            if kl_alpha is not None:
                AnnealingCallback(kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch)

        print('time', time.time() - time_vnet_s)
        # After training, fit Laplace approximation
        print('Fitting Laplace approximation...')
        la = Laplace(model, 'classification', 
                subset_of_weights='all', 
                hessian_structure='diag')
        la.fit(train_loader)
        la.optimize_prior_precision(method='marglik')

        probs_laplace = predict(valid_loader, la, laplace=True)

        print('probs_laplace.npy', probs_laplace)
    else:
        print('  ')
        # After training, fit Laplace approximation
        # print('Fitting Laplace approximation...')
        # la = Laplace(model, 'classification', 
        #         subset_of_weights='all', 
        #         hessian_structure='diag')
        # la.fit(train_loader)
        # la.optimize_prior_precision(method='marglik')

        # probs_laplace = predict_cpu(valid_loader, la, laplace=True)

        # print('probs_laplace.npy', probs_laplace)

if __name__ == "__main__":
    # Example usage (replace with actual parameters or argument parsing as needed)
    train(weights_path="model.pth", epochs=50, batch_size=16, initial_epoch=0,
          kl_start_epoch=10, kl_alpha_increase_per_epoch=0.1, dim=2,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10)
