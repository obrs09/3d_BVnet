
import os
import torch
import torch.nn as nn
import torch.optim as optim
from bayesian_unet import bayesian_unet
from bayesian_vnet import bayesian_vnet
from dropout_unet import dropout_unet
from dropout_vnet import dropout_vnet
from utils import ex, get_latest_file, variational_free_energy_loss

@ex.capture
def load_model(input_shape, weights_path, net, prior_std,
               kernel_size, activation, padding, num_gpus):
    """Loads model from .pth file (PyTorch)."""
    model = net(input_shape,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                prior_std=prior_std)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model

@ex.capture
def get_model(input_shape, weights_dir, resume, bayesian,
              vnet, prior_std, kernel_size, activation, padding,
              kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch,
              ensemble, num_gpus, initial_epoch,
              scale_factor=1, weights_path=None):
    """Loads or creates model (PyTorch)."""
    os.makedirs(weights_dir + "/bayesian", exist_ok=True)
    os.makedirs(weights_dir + "/dropout", exist_ok=True)

    # Sets variables for bayesian or dropout model.
    if bayesian:
        checkpoint_path = (weights_dir + "/bayesian/bayesian-{epoch:02d}.pth")
        net = bayesian_vnet if vnet else bayesian_unet
        if weights_path:
            latest_weights_path = weights_path
        else:
            latest_weights_path = get_latest_file(weights_dir + "/bayesian")
    else:
        checkpoint_path = (weights_dir + "/dropout/dropout-{epoch:02d}.pth")
        net = dropout_vnet if vnet else dropout_unet
        if weights_path:
            latest_weights_path = weights_path
        else:
            latest_weights_path = get_latest_file(weights_dir + "/dropout")

    # Loads or creates model.
    if latest_weights_path and resume:
        model = load_model(input_shape, latest_weights_path, net)
    else:
        model = net(input_shape,
                    kernel_size=kernel_size,
                    activation=activation,
                    padding=padding,
                    prior_std=prior_std)

    # Prints model summary (PyTorch style)
    print(model)

    # Sets loss function.
    if bayesian:
        if initial_epoch >= kl_start_epoch:
            kl_alpha = min(1., kl_alpha + (initial_epoch - kl_start_epoch) * kl_alpha_increase_per_epoch)
        # In PyTorch, kl_alpha is just a float
        loss = variational_free_energy_loss(model, scale_factor, kl_alpha)
    else:
        kl_alpha = None
        loss = nn.BCELoss()

    # In PyTorch, optimizer is created outside this function
    return model, checkpoint_path, kl_alpha
