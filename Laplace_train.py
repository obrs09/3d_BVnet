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
from laplace.utils import FeatureExtractor
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
from bayesian_vnet import FeatureToLogits, VoxelWiseLinearHead
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
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask

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
            py.append(model(x.cuda())[0])
        else:
            py.append(torch.softmax(model(x.cuda())[0], dim=-1))

    # print(py)

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
        x_np = x.cpu().numpy().astype(np.float32)
    else:
        x_np = x.astype(np.float32)
    t = threshold_otsu(x_np)
    y = (x_np > t).astype(np.float32)
    
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
    def __init__(self, base_dataset, target_size, quan_type):
        self.base = base_dataset
        self.target_size = target_size
        self.quan_type = quan_type

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
        
        x = x.to(dtype=self.quan_type)
        y = y.to(dtype=self.quan_type)
        # print(x.dtype)
        # print(y.dtype)

        return x, y

class BFloat16Loader:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            # 假设 batch 是 (x, y) 结构
            yield [x.to(dtype=torch.bfloat16) for x in batch]
    def __len__(self):
        return len(self.loader)

@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch, train_laplace_direct, data_name,
          kl_start_epoch, kl_alpha_increase_per_epoch, dim=2,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10):
    print('Starting training...')
    quan_type = torch.float32
    # quan_type = torch.float16
    # quan_type = torch.bfloat16

    # print('loading data...')
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    # print('getting model...')
    print('dim in train.py', dim )
    # Loads or creates model.
    model, checkpoint_path, kl_alpha = get_model(input_shape,
                                        scale_factor=len(train)/batch_size,
                                        weights_path=weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device, dtype=quan_type)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()  # or another suitable loss
    # criterion = nn.BCEWithLogitsLoss()

    # train_loader = DataLoader(list(zip(train, train_targets)), batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(list(zip(valid, valid_targets)), batch_size=batch_size)
    
    #load data
    if data_name == 'Task06_Lung':
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
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")  # 图像双线性，标签最近邻
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024, a_max=3000,  # 例如 CT HU 值
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(
                    keys=["image", "label"],
                    spatial_size=[resize_x, resize_y, resize_z],
                    mode=("trilinear", "nearest")  # 图像连续，标签最近邻
            ),
            RandFlipd(keys=["image", "label"], prob=1, spatial_axis=0),
            EnsureTyped(keys=["image", "label"]),
        ])
    
        # train_ds = CacheDataset(data=train_data_dicts[0:48], transform=train_transforms, cache_rate=1.0)
        # test_ds  = CacheDataset(data=train_data_dicts[48:60], transform=train_transforms, cache_rate=1.0)
        # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        # valid_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    elif data_name == 'organmnist3d':

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
    
        train_dataset = OtsuResizeDataset(train_base, target_size, quan_type=quan_type)
        test_dataset  = OtsuResizeDataset(test_base, target_size, quan_type=quan_type)
    
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
        
        # x, y = next(iter(train_loader))
        # print(x.shape, y.shape)
        # print(x.min(), x.max())
        # print(y.unique())

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
                if (quan_type == torch.float32) or (quan_type == torch.float16):
                    m = outputs.cpu().detach().numpy().mean()
                elif quan_type == torch.bfloat16:
                    m = outputs.cpu().detach().float().numpy().mean()
                # print(m)

                # if outputs.min() < 0 or outputs.max() > 1:
                #     print("BAD OUTPUT RANGE:", outputs.min().item(), outputs.max().item())
                #     exit()
                
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
        # subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=128)
        # subnetwork_indices = subnetwork_mask.select()
        # model = FeatureExtractor(model, last_layer_name='head')

        # subnetwork_mask = ModuleNameSubnetMask(model, module_names=["head"])
        # subnetwork_mask.select()
        # subnetwork_indices = subnetwork_mask.indices

        
        # la = Laplace(model, "classification",
        #      subset_of_weights="subnetwork",
        #      hessian_structure="diag",
        #      subnetwork_indices=subnetwork_indices)
        # la = Laplace(model.head, 'classification', 
        #         subset_of_weights='all', 
        #         hessian_structure='diag')
        # la.fit(BFloat16Loader(train_loader))
        # la.fit(train_loader)

        # feature_loader = []
        # with torch.no_grad():
        #     for x, y in train_loader:
        #         f = model.backbone(x.to(device))
        #         feature_loader.append((f, y.to(device)))
        # feature_loader = torch.tensor(feature_loader).to(device)

        # la = Laplace(model.head, 'classification', 
        #         subset_of_weights='all', 
        #         hessian_structure='diag')
        # la.fit(feature_loader)

        # for p in model.backbone.parameters():
        #     p.requires_grad = False

        # la = Laplace(
        #     model,
        #     likelihood="classification",
        #     subset_of_weights="all",
        #     hessian_structure="diag"
        # )
        model.head = VoxelWiseLinearHead(32, 1)
        laplace_model = FeatureToLogits(model.backbone, model.head).to(device)

        la = Laplace(
            laplace_model,
            likelihood="regression",
            subset_of_weights="last_layer",   
            hessian_structure="kron"
        )

        print(next(model.head.parameters()).device)
        print(next(model.backbone.parameters()).device)
        
        la.fit(train_loader)
        
        la.optimize_prior_precision(method='marglik')

        print('fit success')

        probs_laplace = predict(valid_loader, la, laplace=True)
        # print('probs_laplace', probs_laplace)
        probs_laplace = torch.sigmoid(torch.from_numpy(probs_laplace))
        # print('sm_probs_laplace', probs_laplace)

        np.save("probs_laplace.npy", probs_laplace)
        print('probs_laplace.npy', np.shape(probs_laplace))

        val_y_list = []
        for batch in valid_loader:
            _, labels = batch  # 假设 DataLoader 返回 (data, labels)
            val_y_list.append(labels)

        val_y = torch.cat(val_y_list)
        # val_y = np.concatenate(val_y_list, axis=0)

        # 3. 根据你的数据集类型,可能需要调整维度
        # OrganMNIST3D 的标签形状通常是 (N, 1, D, H, W) 或 (N, D, H, W)
        # if val_y.ndim == 5 and val_y.shape[1] == 1:
        #     val_y = val_y.squeeze(1)  # 去掉 channel 维度
        
        # 4. 转换为 numpy 数组(如果还不是)
        probs_laplace_np = probs_laplace.numpy() if isinstance(probs_laplace, torch.Tensor) else probs_laplace

        
        # 2. 确保 probs_laplace 和 val_y 的形状匹配
        print('val_y shape:', val_y.shape)
        print('probs_laplace shape:', probs_laplace.shape)

        target = valid_loader
        score = dice(val_outputs>m, val_y.to(device))
        print(score)
        score = np.mean(score)
        print("dice score", score)
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
