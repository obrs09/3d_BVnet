import torch
import torch.nn as nn
import torch.nn.functional as F
from groupnorm import GroupNormalization

# -------------------------
# Backbone: deterministic VNet feature extractor
# -------------------------
class VNetBackbone(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.down1 = self._down(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.down2 = self._down(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.down3 = self._down(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.down4 = self._down(base_channels * 4, base_channels * 8)

        # self.up1 = self._up(base_channels * 8, base_channels * 4)
        # self.up2 = self._up(base_channels * 4, base_channels * 2)
        # self.up3 = self._up(base_channels * 2, base_channels)

        self.up1 = self._up(base_channels * 8, base_channels * 4)
        # 输出: 64
        
        self.up2 = self._up(base_channels * 4 + base_channels * 4, base_channels * 2)
        # 输入: 64 (u1) + 64 (c3) = 128 → 输出 32
        
        self.up3 = self._up(base_channels * 2 + base_channels * 2, base_channels)
        # 输入: 32 (u2) + 32 (c2) = 64 → 输出 16

    def _down(self, cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            GroupNormalization(num_channels=cout),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            GroupNormalization(num_channels=cout),
        )

    def _up(self, cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            GroupNormalization(num_channels=cout),
        )

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)

        u1 = self.up1(c4)
        u1 = torch.cat([u1, c3], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, c2], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, c1], dim=1)

        return u3  # high-res feature map


# -------------------------
# Bayesian head (Laplace target)
# -------------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # 1x1x1 Conv = voxel-wise linear classifier
        self.classifier = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)  # logits


# -------------------------
# Full model wrapper
# -------------------------
class BayesianVNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=1):
        super().__init__()
        self.backbone = VNetBackbone(in_channels, base_channels)
        self.head = SegmentationHead(base_channels * 2, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from groupnorm import GroupNormalization
from utils import normal_prior

# class BayesianConv3dFlipout(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, activation=F.relu, padding=1, prior_fn=None):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
#         self.activation = F.sigmoid
#     def forward(self, x):
#         return self.activation(self.conv(x))

# def down_stage(in_channels, out_channels, kernel_size=3, activation=F.relu, padding=1):
#     layers = [
#         nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
#         nn.ReLU(),
#         GroupNormalization(num_channels=out_channels),
#         nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
#         nn.ReLU(),
#         GroupNormalization(num_channels=out_channels),
        
#     ]
#     return nn.Sequential(*layers)

# def up_stage_CG(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
#     return nn.Sequential(
#         BayesianConv3dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
#         GroupNormalization(num_channels=out_channels),
#     )

# def up_stage_GCG(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
#     return nn.Sequential(
#         GroupNormalization(num_channels=in_channels),
#         BayesianConv3dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
#         GroupNormalization(num_channels=out_channels),
#     )


# def up_sample():
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#     )

# def up_stage(in_channels, skip_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         BayesianConv3dFlipout(in_channels, out_channels, 2, activation=activation, padding=padding, prior_fn=prior_fn),
#         GroupNormalization(num_channels=out_channels),
#         BayesianConv3dFlipout(out_channels + skip_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
#         GroupNormalization(num_channels=out_channels),
#         BayesianConv3dFlipout(out_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
#         GroupNormalization(num_channels=out_channels),
#     )


# def end_stage(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
#     return nn.Sequential(
#         BayesianConv3dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
#         # BayesianConv3dFlipout(1, 1, 1, activation=torch.sigmoid, padding='valid', prior_fn=prior_fn)
#         nn.Conv3d(1, 1, 1, padding='valid')
#     )


# class BayesianVNet(nn.Module):
#     def __init__(self, input_shape=(1, 280, 280, 280), kernel_size=2, activation=F.relu, padding=1, prior_std=1):
#         super().__init__()
#         prior_fn = normal_prior(prior_std)
#         self.down1 = down_stage(input_shape, 16, kernel_size, activation, padding)
#         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.down2 = down_stage(16, 32, kernel_size, activation, padding)
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.down3 = down_stage(32, 64, kernel_size, activation, padding)
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.down4 = down_stage(64, 128, kernel_size, activation, padding)
#         self.ups_1 = up_sample()
#         self.up_1_cg1 = up_stage_CG(128, 64, prior_fn, 2, activation, padding)
#         self.up_1_gcg = up_stage_GCG(128, 64, prior_fn, 3, activation, padding)
#         self.up_1_cg2 = up_stage_CG(64, 64, prior_fn, 3, activation, padding)
#         self.ups_2 = up_sample()
#         self.up_2_cg1 = up_stage_CG(64, 32, prior_fn, 2, activation, padding)
#         self.up_2_gcg = up_stage_GCG(64, 32, prior_fn, 3, activation, padding)
#         self.up_2_cg2 = up_stage_CG(32, 32, prior_fn, 3, activation, padding)
#         self.ups_3 = up_sample()
#         self.up_3_cg1 = up_stage_CG(32, 16, prior_fn, 2, activation, padding)
#         self.up_3_gcg = up_stage_GCG(32, 16, prior_fn, 3, activation, padding)
#         self.up_3_cg2 = up_stage_CG(16, 16, prior_fn, 3, activation, padding)
#         self.end = end_stage(16, 1, prior_fn, kernel_size, activation, 'same')


#     def forward(self, x):
#         if isinstance(x, dict):
#             x = x['image']
#             x = x.permute(0, 2, 3, 4, 1)
#         c1 = self.down1(x)
#         p1 = self.pool1(c1)
#         c2 = self.down2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.down3(p2)
#         p3 = self.pool3(c3)
#         c4 = self.down4(p3)
#         c4 = self.ups_1(c4)
#         c4 = self.up_1_cg1(c4)
#         merge_1 = torch.cat([c3, c4], dim=1)
#         c5 = self.up_1_gcg(merge_1)
#         c5 = self.up_1_cg2(c5)
#         c5 = self.ups_2(c5)
#         c5 = self.up_2_cg1(c5)
#         merge_2 = torch.cat([c2, c5], dim=1)
#         c5 = self.up_2_gcg(merge_2)
#         c5 = self.up_2_cg2(c5)
#         c6 = self.ups_3(c5)
#         c6 = self.up_3_cg1(c6)
#         merge_3 = torch.cat([c1, c6], dim=1)
#         c7 = self.up_3_gcg(merge_3)
#         c7 = self.up_3_cg2(c7)
#         out = self.end(c7)
#         return out

def bayesian_vnet(input_shape=(1, 280, 280, 280), kernel_size=3, activation=F.relu, padding=1, **kwargs):
    prior_std = kwargs.get("prior_std", 1)
    # return BayesianVNet(input_shape, kernel_size, activation, padding, prior_std)
    return BayesianVNet(in_channels=input_shape, base_channels=16, num_classes=1)

class FeatureToLogits(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

class VoxelWiseLinearHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)   # [B*D*H*W, C]
        logits = self.linear(x)                      # [B*D*H*W, 1]
        logits = logits.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        return logits