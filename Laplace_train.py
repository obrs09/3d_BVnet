"""
Laplace Approximation Training for 3D Medical Image Segmentation.

This module provides training functionality with Laplace approximation for
uncertainty quantification in 3D segmentation tasks.

Usage:
    # Via Sacred experiment
    python Laplace_train.py with configs/toy_config_3d.json
    
    # Direct call (for debugging)
    python Laplace_train.py
"""

import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from laplace import Laplace
from monai.metrics import DiceMetric

# Local imports
from data_loaders import create_dataloaders
from dataset import get_train_data
from model import get_model
from bayesian_vnet import FeatureToLogits, VoxelWiseLinearHead
from utils import AnnealingCallback, ex, predict_cpu, predict_gpu, schedule


# ==================== Configuration ====================

# Supported datasets
SUPPORTED_DATASETS = ['Task06_Lung', 'organmnist3d', 'toy_data_3d']

# Supported quantization types
QUAN_TYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def get_quan_type(quan_type_str):
    """Convert string to torch dtype."""
    if isinstance(quan_type_str, torch.dtype):
        return quan_type_str
    if quan_type_str not in QUAN_TYPE_MAP:
        raise ValueError(f"Unknown quan_type: {quan_type_str}. Supported: {list(QUAN_TYPE_MAP.keys())}")
    return QUAN_TYPE_MAP[quan_type_str]


def get_model_save_path(base_dir, data_name, epochs, batch_size, train_quan_type, 
                        initial_learning_rate, epoch=None, is_best=False):
    """
    Generate model save path with important parameters.
    
    Args:
        base_dir: Base directory for saving models
        data_name: Dataset name
        epochs: Total epochs
        batch_size: Batch size
        train_quan_type: Training precision
        initial_learning_rate: Learning rate
        epoch: Current epoch (optional)
        is_best: Whether this is the best model
    
    Returns:
        str: Model save path
    """
    os.makedirs(base_dir, exist_ok=True)
    
    lr_str = f"{initial_learning_rate:.0e}".replace("-", "")
    
    if is_best:
        filename = f"{data_name}_best_ep{epochs}_bs{batch_size}_{train_quan_type}_lr{lr_str}.pth"
    elif epoch is not None:
        filename = f"{data_name}_ep{epoch:03d}_bs{batch_size}_{train_quan_type}_lr{lr_str}.pth"
    else:
        filename = f"{data_name}_ep{epochs}_bs{batch_size}_{train_quan_type}_lr{lr_str}.pth"
    
    return os.path.join(base_dir, filename)


# ==================== Training Functions ====================

def train_one_epoch(model, train_loader, optimizer, criterion, dice_metric, 
                    device, quan_type, epoch, epochs):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0
    train_dice = []
    
    for batch in train_loader:
        # Handle different batch formats
        if isinstance(batch, dict):
            batch_x = batch["image"]
            batch_y = batch["label"]
        else:
            batch_x = batch[0]
            batch_y = batch[1]
        
        batch_x = batch_x.to(device, dtype=quan_type)
        batch_y = batch_y.to(device, dtype=quan_type)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Calculate threshold for dice metric
        if quan_type in [torch.float32, torch.float16]:
            m = outputs.cpu().detach().numpy().mean()
        elif quan_type == torch.bfloat16:
            m = outputs.cpu().detach().float().numpy().mean()
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Calculate dice score
        score = dice_metric(outputs > m, batch_y).cpu().numpy()
        train_dice.append(np.mean(score))
    
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = np.mean(train_dice)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice, m


def validate(model, valid_loader, criterion, dice_metric, device, quan_type, threshold):
    """Validate model."""
    model.eval()
    val_loss = 0
    val_dice = []
    
    with torch.no_grad():
        for val in valid_loader:
            if isinstance(val, dict):
                val_x = val["image"]
                val_y = val["label"]
            else:
                val_x = val[0]
                val_y = val[1]

            val_x = val_x.to(device, dtype=quan_type)
            val_y = val_y.to(device, dtype=quan_type)
            
            val_outputs = model(val_x)
            v_loss = criterion(val_outputs, val_y)
            val_loss += v_loss.item()
            
            score = dice_metric(val_outputs > threshold, val_y).cpu().numpy()
            val_dice.append(np.mean(score))
    
    avg_loss = val_loss / len(valid_loader)
    avg_dice = np.mean(val_dice)
    print(f"Validation Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice, val_outputs


def fit_laplace_approximation(model, train_loader, valid_loader, device, dice_metric,
                               save_path=None):
    """Fit Laplace approximation to the trained model."""
    print('Fitting Laplace approximation...')
    
    # Replace head with voxel-wise linear head
    model.head = VoxelWiseLinearHead(32, 1)
    laplace_model = FeatureToLogits(model.backbone, model.head).to(device)

    la = Laplace(
        laplace_model,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="kron"
    )

    print(f'Head device: {next(model.head.parameters()).device}')
    print(f'Backbone device: {next(model.backbone.parameters()).device}')
    
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')
    
    print('Laplace fit successful!')
    
    # Save Laplace model if path is provided
    if save_path is not None:
        laplace_save_path = save_path.replace('.pth', '_laplace.pth')
        torch.save({
            'laplace_model_state_dict': laplace_model.state_dict(),
            'laplace_state_dict': la.state_dict(),
            'head_in_channels': 32,
            'head_out_channels': 1,
        }, laplace_save_path)
        print(f'Laplace model saved to: {laplace_save_path}')
    
    return la, laplace_model


def load_laplace_model(load_path, backbone, device):
    """
    Load a saved Laplace model.
    
    Args:
        load_path: Path to the saved Laplace model
        backbone: Backbone model (VNetBackbone)
        device: Device to load the model on
    
    Returns:
        la: Laplace approximation object
        laplace_model: FeatureToLogits model
    """
    print(f'Loading Laplace model from: {load_path}')
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Recreate the head and model
    head = VoxelWiseLinearHead(
        checkpoint.get('head_in_channels', 32),
        checkpoint.get('head_out_channels', 1)
    )
    laplace_model = FeatureToLogits(backbone, head).to(device)
    laplace_model.load_state_dict(checkpoint['laplace_model_state_dict'])
    
    # Recreate Laplace approximation
    la = Laplace(
        laplace_model,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="kron"
    )
    la.load_state_dict(checkpoint['laplace_state_dict'])
    
    print('Laplace model loaded successfully!')
    
    return la, laplace_model


def evaluate_laplace(la, valid_loader, dice_metric, device):
    """Evaluate Laplace model on validation set."""
    probs_laplace = predict_gpu(valid_loader, la, laplace=True)
    probs_laplace = torch.sigmoid(torch.from_numpy(probs_laplace))

    np.save("probs_laplace.npy", probs_laplace)
    print(f'Saved probs_laplace.npy with shape: {probs_laplace.shape}')

    # Collect validation labels
    val_y_list = []
    for batch in valid_loader:
        _, labels = batch
        val_y_list.append(labels)
    val_y = torch.cat(val_y_list)

    print(f'val_y shape: {val_y.shape}')
    print(f'probs_laplace shape: {probs_laplace.shape}')
    
    return probs_laplace, val_y


# ==================== Main Training Function ====================

@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch, train_laplace_direct, data_name,
          kl_start_epoch, kl_alpha_increase_per_epoch, dim=2,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10,
          train_quan_type='float32', test_quan_type='float32',
          load_model_path=None):
    """
    Main training function.
    
    Args:
        weights_path: Path to save/load model weights
        epochs: Number of training epochs
        batch_size: Batch size for training
        initial_epoch: Starting epoch (for resume training)
        train_laplace_direct: If True, skip VNet training and fit Laplace directly
        data_name: Dataset name ('Task06_Lung', 'organmnist3d', 'toy_data_3d')
        kl_start_epoch: Epoch to start KL annealing
        kl_alpha_increase_per_epoch: KL weight increase per epoch
        dim: Dimension (2 or 3)
        initial_learning_rate: Initial learning rate
        lr_decay_start_epoch: Epoch to start LR decay
        train_quan_type: Precision for training ('float32', 'float16', 'bfloat16')
        test_quan_type: Precision for testing/inference ('float32', 'float16', 'bfloat16')
        load_model_path: Path to pre-trained model to load (optional)
    """
    print('Starting training...')
    
    # Configuration
    train_dtype = get_quan_type(train_quan_type)
    test_dtype = get_quan_type(test_quan_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Dimension: {dim}')
    print(f'Train precision: {train_quan_type} -> {train_dtype}')
    print(f'Test precision: {test_quan_type} -> {test_dtype}')
    
    # Load data (use train dtype for data loading)
    print('Loading data...')
    train_loader, valid_loader = create_dataloaders(data_name, batch_size, train_dtype)
    
    # Load model
    print('Loading model...')
    input_shape, train_data, valid_data, train_targets, valid_targets = get_train_data()
    model, checkpoint_path, kl_alpha = get_model(
        input_shape,
        scale_factor=len(train_data) / batch_size,
        weights_path=weights_path
    )
    model = model.to(device, dtype=train_dtype)
    
    # Load pre-trained model if specified
    if load_model_path is not None and os.path.exists(load_model_path):
        print(f'Loading pre-trained model from: {load_model_path}')
        state_dict = torch.load(load_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print('Pre-trained model loaded successfully!')
    elif load_model_path is not None:
        print(f'Warning: Model path not found: {load_model_path}, starting with fresh model')
    
    # Generate model save paths with important parameters
    weights_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "weights"
    best_model_path = get_model_save_path(
        weights_dir, data_name, epochs, batch_size, 
        train_quan_type, initial_learning_rate, is_best=True
    )
    print(f'Best model will be saved to: {best_model_path}')

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    dice = DiceMetric(include_background=False, reduction="mean")
    
    # Training tracking
    best_val_loss = float('inf')
    train_dice_scores = []
    val_dice_scores = []
    time_start = time.time()
    
    if not train_laplace_direct:
        # Training loop
        print('Starting training loop...')
        for epoch in range(initial_epoch, epochs):
            # Train (use train_dtype)
            train_loss, train_dice_score, threshold = train_one_epoch(
                model, train_loader, optimizer, criterion, dice,
                device, train_dtype, epoch, epochs
            )
            train_dice_scores.append(train_dice_score)
            
            # Validate (use train_dtype during training phase)
            val_loss, val_dice_score, val_outputs = validate(
                model, valid_loader, criterion, dice, 
                device, train_dtype, threshold
            )
            val_dice_scores.append(val_dice_score)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to: {best_model_path}")

            # Learning rate scheduling
            for param_group in optimizer.param_groups:
                param_group['lr'] = schedule(epoch, initial_learning_rate, lr_decay_start_epoch)

            # KL annealing (if applicable)
            if kl_alpha is not None:
                AnnealingCallback(kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch)

        print(f'Training time: {time.time() - time_start:.2f}s')
        
        # Convert model to test precision for Laplace fitting
        if train_dtype != test_dtype:
            print(f'Converting model from {train_dtype} to {test_dtype} for Laplace fitting...')
            model = model.to(dtype=test_dtype)
        
        # Fit Laplace approximation
        la, laplace_model = fit_laplace_approximation(
            model, train_loader, valid_loader, device, dice,
            save_path=best_model_path
        )
        
        # Evaluate Laplace model
        probs_laplace, val_y = evaluate_laplace(la, valid_loader, dice, device)
        
        # Final dice score
        score = dice(val_outputs > threshold, val_y.to(device))
        print(f"Final Dice score: {np.mean(score.cpu().numpy()):.4f}")
    
    else:
        # Direct Laplace training mode - skip VNet training, use fresh model
        print('Direct Laplace training mode - using fresh model...')
        
        # Convert model to test precision for Laplace fitting
        if train_dtype != test_dtype:
            print(f'Converting model from {train_dtype} to {test_dtype} for Laplace fitting...')
            model = model.to(dtype=test_dtype)
        
        # Fit Laplace approximation directly on untrained model
        la, laplace_model = fit_laplace_approximation(
            model, train_loader, valid_loader, device, dice,
            save_path=best_model_path
        )
        
        # Evaluate Laplace model
        probs_laplace, val_y = evaluate_laplace(la, valid_loader, dice, device)
        
        # Calculate dice score using Laplace predictions
        threshold = 0.5  # Default threshold for untrained model
        score = dice(probs_laplace > threshold, val_y)
        print(f"Laplace Dice score: {np.mean(score.cpu().numpy()):.4f}")


if __name__ == "__main__":
    train(
        weights_path="model.pth", 
        epochs=50, 
        batch_size=4, 
        initial_epoch=0,
        train_laplace_direct=False,
        data_name="toy_data_3d",
        kl_start_epoch=10, 
        kl_alpha_increase_per_epoch=0.1, 
        dim=3,
        initial_learning_rate=1e-3, 
        lr_decay_start_epoch=10
    )
