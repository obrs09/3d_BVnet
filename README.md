# 3D Bayesian VNet with Laplace Approximation

A PyTorch implementation of 3D medical image segmentation with uncertainty quantification using Bayesian VNet and Laplace approximation.

## Overview

This project provides:
- **3D VNet** for volumetric medical image segmentation
- **Laplace Approximation** for uncertainty quantification
- Support for multiple datasets (Task06_Lung, OrganMNIST3D, toy data)
- Configurable training precision (float32, float16, bfloat16)

## Project Structure

```
3d_BVnet/
├── configs/                    # Configuration files
│   ├── toy_config.json         # 2D toy data config
│   ├── toy_config_3d.json      # 3D toy data config
│   └── lung_CT_3d_config.json  # Lung CT config
├── Laplace_train.py            # Main training script with Laplace
├── train.py                    # Basic training script
├── test.py                     # Testing script
├── model.py                    # Model loading utilities
├── bayesian_vnet.py            # 3D Bayesian VNet architecture
├── bayesian_unet.py            # 2D Bayesian UNet architecture
├── data_loaders.py             # Dataset loading utilities
├── dataset.py                  # Dataset processing
├── utils.py                    # Utility functions
├── groupnorm.py                # Group normalization layer
├── generate_toy_data.py        # 2D toy data generator
├── generate_toy_data_3d.py     # 3D toy data generator
├── toy_train_test_3d_laplace.sh # Training shell script
└── README.md                   # This file
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install monai
pip install laplace-torch
pip install medmnist
pip install sacred
pip install scikit-image
pip install nibabel
pip install numpy matplotlib
```

## Datasets

### Supported Datasets

| Dataset | Description | Config |
|---------|-------------|--------|
| `toy_data_3d` | Synthetic 3D data for testing | `configs/toy_config_3d.json` |
| `organmnist3d` | MedMNIST 3D organ dataset | `configs/toy_config_3d.json` |
| `Task06_Lung` | Medical Decathlon Lung CT | `configs/lung_CT_3d_config.json` |

### Data Preparation

#### Toy Data (Auto-generated)
```bash
python generate_toy_data_3d.py
```

#### OrganMNIST3D (Auto-download)
The dataset will be automatically downloaded when first used.

#### Task06_Lung
Download from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) and place in `../Task06_Lung/`.

## Usage

### Quick Start

```bash
# Train with toy data (default)
./toy_train_test_3d_laplace.sh

# Train with OrganMNIST3D
./toy_train_test_3d_laplace.sh organmnist3d configs/toy_config_3d.json

# Train with Lung CT
./toy_train_test_3d_laplace.sh Task06_Lung configs/lung_CT_3d_config.json
```

### Shell Script Parameters

```bash
./toy_train_test_3d_laplace.sh [data_name] [config_file] [train_quan_type] [test_quan_type] [load_model_path]
```

| Parameter | Default | Options |
|-----------|---------|---------|
| `data_name` | `toy_data_3d` | `toy_data_3d`, `organmnist3d`, `Task06_Lung` |
| `config_file` | `configs/toy_config_3d.json` | Path to config file |
| `train_quan_type` | `float32` | `float32`, `float16`, `bfloat16` |
| `test_quan_type` | `float32` | `float32`, `float16`, `bfloat16` |
| `load_model_path` | (none) | Path to pre-trained model |

### Examples

```bash
# Train with float16 precision, test with float32
./toy_train_test_3d_laplace.sh toy_data_3d configs/toy_config_3d.json float16 float32

# Resume training from pre-trained model
./toy_train_test_3d_laplace.sh toy_data_3d configs/toy_config_3d.json float32 float32 weights/pretrained.pth

# Train Lung CT with bfloat16
./toy_train_test_3d_laplace.sh Task06_Lung configs/lung_CT_3d_config.json bfloat16 float32
```

### Python Direct Call

```bash
# Using Sacred experiment
python Laplace_train.py with configs/toy_config_3d.json

# Override parameters
python Laplace_train.py with configs/toy_config_3d.json \
    "data_name=organmnist3d" \
    "epochs=100" \
    "batch_size=8" \
    "train_quan_type=float16"
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_name` | Dataset to use | `toy_data_3d` |
| `epochs` | Number of training epochs | `50` |
| `batch_size` | Batch size | `4` |
| `initial_learning_rate` | Initial learning rate | `0.001` |
| `lr_decay_start_epoch` | Epoch to start LR decay | `10` |
| `train_quan_type` | Training precision | `float32` |
| `test_quan_type` | Testing/inference precision | `float32` |
| `train_laplace_direct` | Skip VNet training, fit Laplace directly | `false` |
| `load_model_path` | Path to pre-trained model | `null` |
| `dim` | Dimension (2 or 3) | `3` |

### Example Config (toy_config_3d.json)

```json
{
    "batch_size": 4,
    "data_name": "toy_data_3d",
    "epochs": 50,
    "initial_learning_rate": 0.001,
    "train_quan_type": "float32",
    "test_quan_type": "float32",
    "train_laplace_direct": false,
    "load_model_path": null,
    "dim": 3
}
```

## Model Architecture

### VNet Backbone

```
Input → Down1 → Pool → Down2 → Pool → Down3 → Pool → Down4
                                                      ↓
Output ← Up3 ← Up2 ← Up1 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

- **Encoder**: 4 down-sampling stages with Conv3D + GroupNorm + ReLU
- **Decoder**: 3 up-sampling stages with skip connections
- **Output**: Voxel-wise segmentation logits

### Laplace Approximation

The Laplace approximation is applied to the last layer for uncertainty quantification:

1. Train VNet backbone (deterministic)
2. Replace head with `VoxelWiseLinearHead`
3. Fit Laplace approximation on the last layer
4. Optimize prior precision using marginal likelihood

## Output Files

### Saved Models

Models are saved with descriptive filenames:

```
{weights_dir}/{data_name}_best_ep{epochs}_bs{batch_size}_{train_quan_type}_lr{lr}.pth
{weights_dir}/{data_name}_best_ep{epochs}_bs{batch_size}_{train_quan_type}_lr{lr}_laplace.pth
```

Example:
- `toy_weights_3d/toy_data_3d_best_ep50_bs4_float32_lr1e03.pth` - VNet model
- `toy_weights_3d/toy_data_3d_best_ep50_bs4_float32_lr1e03_laplace.pth` - Laplace model

### Laplace Model Contents

```python
{
    'laplace_model_state_dict': ...,  # FeatureToLogits model state
    'laplace_state_dict': ...,         # Laplace approximation state
    'head_in_channels': 32,            # Head input channels
    'head_out_channels': 1,            # Head output channels
}
```

### Predictions

- `probs_laplace.npy` - Laplace predictions on validation set

## Loading Saved Models

### Load VNet Model

```python
import torch
from model import get_model

# Create model
model, _, _ = get_model(input_shape=1)

# Load weights
model.load_state_dict(torch.load('weights/model.pth'))
model.eval()
```

### Load Laplace Model

```python
from Laplace_train import load_laplace_model
from bayesian_vnet import BayesianVNet

# Create backbone
model = BayesianVNet(in_channels=1)
model.load_state_dict(torch.load('weights/model.pth'))

# Load Laplace model
la, laplace_model = load_laplace_model(
    'weights/model_laplace.pth',
    model.backbone,
    device='cuda'
)

# Inference with uncertainty
mean, var = la(input_tensor)
```

## Training Modes

### Mode 1: Full Training (Default)

Train VNet first, then fit Laplace approximation:

```bash
python Laplace_train.py with configs/toy_config_3d.json "train_laplace_direct=false"
```

### Mode 2: Direct Laplace Training

Skip VNet training, fit Laplace directly on fresh model:

```bash
python Laplace_train.py with configs/toy_config_3d.json "train_laplace_direct=true"
```

### Mode 3: Resume Training

Load pre-trained model and continue training:

```bash
python Laplace_train.py with configs/toy_config_3d.json \
    "load_model_path=weights/pretrained.pth" \
    "initial_epoch=25"
```

## Metrics

- **Dice Score**: Primary segmentation metric
- **BCE Loss**: Binary cross-entropy with logits loss

## Precision Options

| Type | Description | Use Case |
|------|-------------|----------|
| `float32` | Full precision | Best accuracy, standard training |
| `float16` | Half precision | Faster training, less memory |
| `bfloat16` | Brain float | Good for large models, TPU-friendly |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python Laplace_train.py with configs/toy_config_3d.json "batch_size=2"

# Use lower precision
python Laplace_train.py with configs/toy_config_3d.json "train_quan_type=float16"
```

### Dataset Not Found

```bash
# For toy data, generate first
python generate_toy_data_3d.py

# For Task06_Lung, check path
ls ../Task06_Lung/
```

## References

- [Laplace Redux](https://github.com/AlexImmer/Laplace) - Laplace approximation library
- [MONAI](https://monai.io/) - Medical image analysis framework
- [MedMNIST](https://medmnist.com/) - Medical image classification benchmarks
- [VNet](https://arxiv.org/abs/1606.04797) - V-Net architecture paper

## License

MIT License
