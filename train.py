import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_train_data
from model import get_model
from utils import AnnealingCallback, ex

# Ignores TensorFlow CPU messages (not needed in PyTorch, but kept for compatibility)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@ex.capture
def schedule(epoch, initial_learning_rate, lr_decay_start_epoch):
    """Defines exponentially decaying learning rate."""
    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))

@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch,
          kl_start_epoch, kl_alpha_increase_per_epoch,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10):
    """Trains a model (PyTorch version)."""
    print('loading data...')
    # Loads or creates training data.
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    print('getting model...')
    # Loads or creates model.
    model, checkpoint_path, kl_alpha = get_model(input_shape,
                                        scale_factor=len(train)/batch_size,
                                        weights_path=weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    criterion = nn.BCELoss()  # or another suitable loss

    train_loader = DataLoader(list(zip(train, train_targets)), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(list(zip(valid, valid_targets)), batch_size=batch_size)

    best_val_loss = float('inf')
    for epoch in range(initial_epoch, epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float).permute(0, 3, 1, 2), batch_y.to(device, dtype=torch.float)
            optimizer.zero_grad()
            print('batch_x.shape################', batch_x.shape )
            batch_x = batch_x.permute(1,0,2,3)  
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in valid_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                val_loss += v_loss.item()
        val_loss /= len(valid_loader)
        print(f"Validation Loss: {val_loss:.4f}")

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

if __name__ == "__main__":
    # Example usage (replace with actual parameters or argument parsing as needed)
    train(weights_path="model.pth", epochs=50, batch_size=16, initial_epoch=0,
          kl_start_epoch=10, kl_alpha_increase_per_epoch=0.1,
          initial_learning_rate=1e-3, lr_decay_start_epoch=10)
