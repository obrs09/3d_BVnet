
import torch
import torch.nn as nn
import torch.nn.functional as F
from groupnorm import GroupNormalization

def down_stage(in_channels, out_channels, kernel_size=3, activation=F.relu, padding=1):
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        GroupNormalization(num_channels=out_channels),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
        GroupNormalization(num_channels=out_channels),
        nn.ReLU(),
    ]
    return nn.Sequential(*layers)

def up_stage(in_channels, skip_channels, out_channels, kernel_size=3, activation=F.relu, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(in_channels, out_channels, 2, padding=padding),
        GroupNormalization(num_channels=out_channels),
        nn.ReLU(),
        # Concatenation with skip connection will be handled in forward
        nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size, padding=padding),
        GroupNormalization(num_channels=out_channels),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
        GroupNormalization(num_channels=out_channels),
        nn.ReLU(),
        nn.Dropout3d(0.5)
    )

def end_stage(in_channels, kernel_size=3, activation=F.relu, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, 1, kernel_size, padding=padding),
        nn.Conv3d(1, 1, 1),
        nn.Sigmoid()
    )

class DropoutVNet(nn.Module):
    def __init__(self, input_shape=(1, 280, 280, 280), kernel_size=3, activation=F.relu, padding=1):
        super().__init__()
        self.down1 = down_stage(input_shape[0], 16, kernel_size, activation, padding)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = down_stage(16, 32, kernel_size, activation, padding)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = down_stage(32, 64, kernel_size, activation, padding)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = down_stage(64, 128, kernel_size, activation, padding)
        self.dropout = nn.Dropout3d(0.5)
        self.up1 = up_stage(128, 64, 64, kernel_size, activation, padding)
        self.up2 = up_stage(64, 32, 32, kernel_size, activation, padding)
        self.up3 = up_stage(32, 16, 16, kernel_size, activation, padding)
        self.end = end_stage(16, kernel_size, activation, padding)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        c4 = self.dropout(c4)
        u1 = self.up1(c4)
        u1_cat = torch.cat([c3, u1], dim=1)
        u2 = self.up2(u1_cat)
        u2_cat = torch.cat([c2, u2], dim=1)
        u3 = self.up3(u2_cat)
        u3_cat = torch.cat([c1, u3], dim=1)
        out = self.end(u3_cat)
        return out

def dropout_vnet(input_shape=(1, 280, 280, 280), kernel_size=3, activation=F.relu, padding=1, **kwargs):
    return DropoutVNet(input_shape, kernel_size, activation, padding)
