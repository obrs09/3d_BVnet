import os
import numpy as np
from sacred import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal, Independent

ex = Experiment()
ex.add_config("configs/toy_config.json")


# ==================== Prediction Functions ====================

@torch.no_grad()
def predict_cpu(dataloader, model, laplace=False):
    """Predict on CPU."""
    import time
    py = []
    idx = 0
    
    for x, _ in dataloader:
        time_start = time.time()
        if laplace:
            py.append(model(x))
            time_end = time.time()
            print(f'idx {idx}, time {time_end - time_start:.4f}s')
            idx += 1
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py).cpu().numpy()


@torch.no_grad()
def predict_gpu(dataloader, model, laplace=False):
    """Predict on GPU."""
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda())[0])
        else:
            py.append(torch.softmax(model(x.cuda())[0], dim=-1))

    return torch.cat(py).cpu().numpy()


@ex.capture
def schedule(epoch, initial_learning_rate, lr_decay_start_epoch):
    """Defines exponentially decaying learning rate."""
    import math
    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))


# ==================== Utility Functions ====================

def round_down(num, factor):
    """Rounds num to next lowest multiple of factor."""
    return (num // factor) * factor

def acc(a, b):
    """Calculates number of matches in two np arrays."""
    return np.count_nonzero(a == b) / a.size

def absolute_file_paths(directory, match=""):
    """Gets absolute file paths from a directory.
    Does not include subdirectories.
    Args:
        match: Returns only paths of files containing the given string.
    """
    paths = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if match in f:
                paths.append(os.path.abspath(os.path.join(root, f)))
        break
    return paths

def get_latest_file(directory, match=""):
    """Gets the absolute file path of the last modified file in a directory.
    Args:
        match: Returns only paths of files containing the given string.
    """
    paths = absolute_file_paths(directory, match=match)
    if paths:
        return max(paths, key=os.path.getctime)
    else:
        return None

def standardize(raw):
    """Transforms data to have mean 0 and std 1."""
    return (raw - np.mean(raw)) / np.std(raw)

def variational_free_energy_loss(model, scale_factor, kl_alpha):
    """Defines variational free energy loss for PyTorch.
    Sum of KL divergence (supplied by model.losses) and binary cross-entropy.
    """
    # KL Divergence should be applied once per epoch only, so
    # scale_factor should be num_samples / batch_size.
    kl = sum(model.losses) / scale_factor if hasattr(model, 'losses') else 0
    def loss(y_true, y_pred):
        bce = F.binary_cross_entropy(y_pred, y_true)
        return bce + kl_alpha * kl
    return loss

@ex.capture
def normal_prior(prior_std):
    """Defines normal distribution prior for Bayesian neural network (PyTorch)."""
    def prior_fn(dtype, shape, name=None, trainable=True, add_variable_fn=None):
        dist = Normal(loc=torch.zeros(shape, dtype=dtype), scale=torch.full(shape, prior_std, dtype=dtype))
        return Independent(dist, 1)
    return prior_fn

class AnnealingCallback:
    def __init__(self, kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch):
        self.kl_alpha = kl_alpha
        self.kl_start_epoch = kl_start_epoch
        self.kl_alpha_increase_per_epoch = kl_alpha_increase_per_epoch
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= self.kl_start_epoch - 2:
            new_kl_alpha = min(self.kl_alpha + self.kl_alpha_increase_per_epoch, 1.)
            self.kl_alpha = new_kl_alpha
        print("Current KL Weight is " + str(self.kl_alpha))
