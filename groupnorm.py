# import torch
# import torch.nn as nn

# class GroupNormalization(nn.Module):
#     def __init__(self, num_channels, groups=4, eps=1e-5, affine=True):
#         super(GroupNormalization, self).__init__()
#         assert num_channels % groups == 0, "num_channels must be divisible by groups"
#         self.groups = groups
#         self.num_channels = num_channels
#         self.eps = eps
#         self.affine = affine
#         if self.affine:
#             self.weight = nn.Parameter(torch.ones(num_channels))
#             self.bias = nn.Parameter(torch.zeros(num_channels))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#     def forward(self, x):
#         # return x
#         N, C = x.shape[:2]
#         G = self.groups
#         if C % G != 0:
#             # Adjust the number of groups to the largest factor of C that is <= G
#             for g in range(G, 0, -1):
#                 if C % g == 0:
#                     G = g
#                     break
#             else:
#                 # If no suitable group size is found, set G to 1
#                 G = 1
#         # 将空间维度合并为一组
#         spatial_dims = x.shape[2:]
#         x = x.contiguous().view(N, G, C // G, -1)
#         mean = x.mean(dim=(2, 3), keepdim=True)
#         var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
#         x = (x - mean) / (var + self.eps).sqrt()
#         x = x.view(N, C, *spatial_dims)
#         if self.affine:
#             w_shape = (1, C) + (1,) * len(spatial_dims)
#             b_shape = (1, C) + (1,) * len(spatial_dims)
#             w = self.weight.view(*w_shape)
#             b = self.bias.view(*b_shape)
#             x = x * w + b
#         return x

#     def extra_repr(self):
#         return '{num_channels}, groups={groups}, eps={eps}, affine={affine}'.format(**self.__dict__)


# import torch
# import torch.nn as nn

# class GroupNormalization(nn.Module):
#     """
#     PyTorch implementation of Group Normalization
#     Equivalent to the Keras version you provided.
#     """

#     def __init__(self,
#                  num_groups=4,
#                  num_channels=None,
#                  eps=1e-5,
#                  affine=True):
#         """
#         Args:
#             num_groups (int): number of groups for Group Normalization
#             num_channels (int): number of channels in the input (needed for weight/bias)
#             eps (float): epsilon to avoid divide-by-zero
#             affine (bool): whether to include learnable scale (gamma) and shift (beta)
#         """
#         super(GroupNormalization, self).__init__()
#         assert num_channels is not None, "num_channels must be specified"

#         if num_channels % num_groups != 0:
#             raise ValueError("num_channels must be divisible by num_groups")

#         self.num_groups = num_groups
#         self.num_channels = num_channels
#         self.eps = eps
#         self.affine = affine

#         if self.affine:
#             self.weight = nn.Parameter(torch.ones(num_channels))
#             self.bias = nn.Parameter(torch.zeros(num_channels))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)

#     def forward(self, x):
#         N, C = x.shape[:2]
#         # reshape to (N, num_groups, C // num_groups, ...)
#         x = x.view(N, self.num_groups, C // self.num_groups, *x.shape[2:])
#         mean = x.mean(dim=(2, *range(3, x.dim())), keepdim=True)
#         var = x.var(dim=(2, *range(3, x.dim())), keepdim=True, unbiased=False)
#         x = (x - mean) / torch.sqrt(var + self.eps)
#         x = x.view(N, C, *x.shape[3:])  # reshape back

#         if self.affine:
#             # broadcast weight and bias
#             w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
#             b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
#             x = x * w + b

#         return x


import torch
import torch.nn as nn

class GroupNormalization(nn.Module):
    """Group normalization layer (PyTorch implementation)

    Group Normalization divides the channels into groups and computes within
    each group the mean and variance for normalization. GN's computation is
    independent of batch sizes, and its accuracy is stable in a
    wide range of batch sizes.

    This implementation is a wrapper around PyTorch's native `nn.GroupNorm`
    to maintain an interface similar to the original Keras implementation.

    Arguments
        num_channels: Integer, the number of channels of the input tensor.
            This is a required argument in PyTorch.
        groups: Integer, the number of groups for Group Normalization.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.

    Input shape
        (N, C, *) where N is the batch size, C is the number of channels,
        and * represents any number of additional spatial dimensions.
        The channel dimension is expected to be at axis 1.

    Output shape
        Same shape as input.

    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """
    def __init__(self,
                 num_channels,
                 groups=4,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 **kwargs):
        super(GroupNormalization, self).__init__()

        if num_channels % groups != 0:
            print('num_channels', num_channels)
            raise ValueError(f"Number of channels ({num_channels}) must be a multiple of the number of groups ({groups}).")

        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.num_channels = num_channels
        
        # In PyTorch, 'affine' controls both scale and center parameters.
        # We handle the individual enabling/disabling below.
        self.affine = self.scale or self.center

        # Use the highly optimized native PyTorch GroupNorm layer
        self.norm = nn.GroupNorm(
            num_groups=self.groups,
            num_channels=self.num_channels,
            eps=self.epsilon,
            affine=self.affine
        )
        
        # If affine is enabled, we might need to disable grad for one of them
        if self.affine:
            if not self.scale:
                # self.norm.weight corresponds to gamma (scale)
                # Set requires_grad to False to prevent it from being trained.
                # The weight is initialized to 1 by default, so it acts as an identity multiplication.
                self.norm.weight.requires_grad = False
            
            if not self.center:
                # self.norm.bias corresponds to beta (center)
                # Set requires_grad to False to prevent it from being trained.
                # The bias is initialized to 0 by default, so it acts as an identity addition.
                self.norm.bias.requires_grad = False


    def forward(self, inputs):
        """
        Forward pass for the GroupNormalization layer.
        """
        return self.norm(inputs)

    def __repr__(self):
        """
        Provides a string representation of the layer.
        """
        return (f"GroupNormalization(num_channels={self.num_channels}, groups={self.groups}, "
                f"epsilon={self.epsilon}, center={self.center}, scale={self.scale})")


# if __name__ == "__main__":
#     # --- Example Usage ---
    
#     # Define input tensor with shape (N, C, H, W)
#     # Batch size N=2, Channels C=8, Height H=10, Width W=10
#     # The number of channels (8) must be divisible by the number of groups.
#     input_tensor = torch.randn(2, 8, 10, 10)

#     # Case 1: Use 4 groups for normalization
#     print("--- Case 1: Using 4 groups ---")
#     gn_layer = GroupNormalization(num_channels=8, groups=4)
#     print("Layer:", gn_layer)
    
#     output_tensor = gn_layer(input_tensor)
    
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)
#     # The output shape is the same as the input shape
#     assert input_tensor.shape == output_tensor.shape
#     print("-" * 20)

#     # Case 2: Use 2 groups, disable scale (gamma) but keep center (beta)
#     print("--- Case 2: Using 2 groups, no scale ---")
#     gn_layer_no_scale = GroupNormalization(num_channels=8, groups=2, scale=False)
#     print("Layer:", gn_layer_no_scale)
    
#     # Check if the weight (gamma) parameter is frozen
#     print(f"Weight (gamma) is trainable: {gn_layer_no_scale.norm.weight.requires_grad}")
#     print(f"Bias (beta) is trainable: {gn_layer_no_scale.norm.bias.requires_grad}")

#     output_tensor_no_scale = gn_layer_no_scale(input_tensor)
    
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor_no_scale.shape)
#     assert input_tensor.shape == output_tensor_no_scale.shape
#     print("-" * 20)
    
#     # --- Example in a simple model ---
#     model = nn.Sequential(
#         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
#         GroupNormalization(num_channels=16, groups=8),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#         GroupNormalization(num_channels=32, groups=16),
#         nn.ReLU()
#     )
    
#     print("--- Example Model ---")
#     print(model)
    
#     test_image = torch.randn(4, 3, 64, 64) # Batch of 4 images
#     output = model(test_image)
#     print("\nModel input shape:", test_image.shape)
#     print("Model output shape:", output.shape)