
import torch
import torch.nn as nn
import torch.nn.functional as F
from groupnorm import GroupNormalization
from utils import normal_prior

class BayesianConv2dFlipout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=F.relu, padding=1, prior_fn=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = F.relu
        # prior_fn is not used in this placeholder, but would be for a full Bayesian layer
    def forward(self, x):
        return self.activation(self.conv(x))

def down_stage(in_channels, out_channels, kernel_size=3, activation=F.relu, padding=1):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(),

        GroupNormalization(num_channels=out_channels),

        
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(),

        GroupNormalization(num_channels=out_channels),


    ]
    return nn.Sequential(*layers)

def up_stage_CG(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
    return nn.Sequential(
        BayesianConv2dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
        # nn.ReLU(),

        GroupNormalization(num_channels=out_channels),

        
    )

def up_stage_GCG(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
    return nn.Sequential(

        GroupNormalization(num_channels=in_channels),

        BayesianConv2dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
        # nn.ReLU(),

        GroupNormalization(num_channels=out_channels),

        
    )


def up_sample():
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
    )

def end_stage(in_channels, out_channels, prior_fn, kernel_size=3, activation=F.relu, padding=1):
    return nn.Sequential(
        BayesianConv2dFlipout(in_channels, out_channels, kernel_size, activation=activation, padding=padding, prior_fn=prior_fn),
        # nn.ReLU(),
        BayesianConv2dFlipout(1, 1, 1, activation=torch.sigmoid, padding='valid', prior_fn=prior_fn),
        # nn.ReLU(),
    )

class BayesianUNet(nn.Module):
    def __init__(self, input_shape=(1, 280, 280), kernel_size=3, activation=F.relu, padding=1, prior_std=1):
        super().__init__()
        prior_fn = normal_prior(prior_std)
        self.down1 = down_stage(input_shape, 16, kernel_size, activation, padding)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down2 = down_stage(16, 32, kernel_size, activation, padding)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down3 = down_stage(32, 64, kernel_size, activation, padding)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down4 = down_stage(64, 128, kernel_size, activation, padding)
        # self.up1 = up_stage(128, 64, 64, prior_fn, kernel_size, activation, padding)
        # self.up2 = up_stage(64, 32, 32, prior_fn, kernel_size, activation, padding)
        # self.up3 = up_stage(32, 16, 16, prior_fn, kernel_size, activation, padding)
        self.ups_1 = up_sample()
        self.up_1_cg1 = up_stage_CG(128, 64, prior_fn, 2, activation, padding)
        self.up_1_gcg = up_stage_GCG(128, 64, prior_fn, 3, activation, padding)
        self.up_1_cg2 = up_stage_CG(64, 64, prior_fn, 3, activation, padding)

        self.ups_2 = up_sample()
        self.up_2_cg1 = up_stage_CG(64, 32, prior_fn, 2, activation, padding)
        self.up_2_gcg = up_stage_GCG(64, 32, prior_fn, 3, activation, padding)
        self.up_2_cg2 = up_stage_CG(32, 32, prior_fn, 3, activation, padding)

        self.ups_3 = up_sample()
        self.up_3_cg1 = up_stage_CG(32, 16, prior_fn, 2, activation, padding)
        self.up_3_gcg = up_stage_GCG(32, 16, prior_fn, 3, activation, padding)
        self.up_3_cg2 = up_stage_CG(16, 16, prior_fn, 3, activation, padding)

        # self.up2 = up_stage(64, 32, prior_fn, kernel_size, activation, padding)
        # self.up3 = up_stage(32, 16, prior_fn, kernel_size, activation, padding)
        self.end = end_stage(16, 1, prior_fn, kernel_size, activation, 'same')

    def forward(self, x):
        # print('input', x.shape)
        c1 = self.down1(x.permute(0, 3, 1, 2))
        # c1 = self.down1(x)
        # print('c1', c1.shape)
        p1 = self.pool1(c1)
        # print('p1', p1.shape)
        c2 = self.down2(p1)
        # print('c2', c2.shape)
        p2 = self.pool2(c2)
        # print('p2', p2.shape)
        c3 = self.down3(p2)
        # print('c3', c3.shape)
        p3 = self.pool3(c3)
        # print('p3', p3.shape)
        c4 = self.down4(p3)
        # print('c41', c4.shape)

        # print('###################')

        c4 = self.ups_1(c4)
        # print('c42', c4.shape)
        c4 = self.up_1_cg1(c4)
        # print('c43', c4.shape)
        merge_1 = torch.cat([c3, c4], dim=1)
        # print('merge_1', merge_1.shape)
        c5 = self.up_1_gcg(merge_1)
        # print('c51', c5.shape)
        c5 = self.up_1_cg2(c5)
        # print('c52', c5.shape)

        # print('###################')

        c5 = self.ups_2(c5)
        # print('c53', c5.shape)
        c5 = self.up_2_cg1(c5)
        # print('c54', c5.shape)
        merge_2 = torch.cat([c2, c5], dim=1)
        # print('merge_2', merge_2.shape)
        c5 = self.up_2_gcg(merge_2)
        # print('c55', c5.shape)
        c5 = self.up_2_cg2(c5)
        # print('c56', c5.shape)

        # print('###################')

        c6 = self.ups_3(c5)
        # print('c61', c6.shape)
        c6 = self.up_3_cg1(c6)
        # print('c62', c6.shape)
        merge_3 = torch.cat([c1, c6], dim=1)
        # print('merge_3', merge_3.shape)
        c7 = self.up_3_gcg(merge_3)
        # print('c71', c7.shape)
        c7 = self.up_3_cg2(c7)
        # print('c72', c7.shape)


        # u1_s2 = self.ups_2(u1_cat)
        # u1 = self.up1_2(u1_s2)
        # u1_cat = torch.cat([c2, u1], dim=1)

        # u1_s3 = self.ups_3(u1_cat)
        # u1 = self.up1_3(u1_s3)
        # u1_cat = torch.cat([c1, u1], dim=1)

        # u2 = self.up2_1(u1_cat)
        # u2_cat = torch.cat([c2, u2], dim=1)
        # u3 = self.up3_1(u2_cat)
        # u3_cat = torch.cat([c1, u3], dim=1)


        # u1 = self.up1(c4)
        # u1_cat = torch.cat([c3, u1], dim=1)
        # u2 = self.up2(u1_cat)
        # u2_cat = torch.cat([c2, u2], dim=1)
        # u3 = self.up3(u2_cat)
        # u3_cat = torch.cat([c1, u3], dim=1)
        out = self.end(c7)
        # print(out.shape)
        return out

def bayesian_unet(input_shape=(1, 280, 280), kernel_size=3, activation=F.relu, padding=1, **kwargs):
    prior_std = kwargs.get("prior_std", 1)
    return BayesianUNet(input_shape, kernel_size, activation, padding, prior_std)
