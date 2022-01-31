import numpy as np

import torch
from torch.nn import functional as F


class Hyperparameters():
    def __init__(self,
                 base_dim: int,
                 res_blocks: int,
                 bottleneck: bool,
                 skip: bool,
                 weight_norm: bool,
                 coupling_bn: bool,
                 affine: bool,
                 scale_reg: float):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine
        self.scale_reg = scale_reg


def preprocessor(n_bits=8):
    def preprocess(img):
        # number of different values/colours per dimension
        n_bins = 2.0 ** n_bits

        img = img * 255

        if n_bits < 8:
            img = torch.floor(img / 2 ** (8 - n_bits))

        img = img / n_bins - 0.5
        # add noise to make the space continuous
        img = img + torch.rand_like(img) / n_bins

        return img

    return preprocess


def bits_per_dim(loss, img_shape):
    dim = np.prod(img_shape)
    bpd = loss / (np.log(2) * dim)

    return bpd


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def pre_process(x, data_info, noise=False):
    n_bits = data_info["n_bits"]
    n_bins = 2.0 ** data_info["n_bits"]
    x = x * 255

    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))

    x = x / n_bins

    constraint = 0.9
    B, C, H, W = x.shape

    if noise:
        # dequantization
        noise = torch.distributions.Uniform(0., 1.).sample((B, C, H, W)).to(x.device)
        x = x + noise / n_bins

    # restrict data
    x *= 2.  # [0, 2]
    x -= 1.  # [-1, 1]
    x *= constraint  # [-0.9, 0.9]
    x += 1.  # [0.1, 1.9]
    x /= 2.  # [0.05, 0.95]

    # logit data
    logit_x = torch.log(x) - torch.log(1. - x)

    # log-determinant of Jacobian from the transform
    pre_logit_scale = torch.tensor(
        np.log(constraint) - np.log(1. - constraint))
    log_det = F.softplus(logit_x) + F.softplus(-logit_x) \
              - F.softplus(-pre_logit_scale)

    return logit_x, torch.sum(log_det, dim=(1, 2, 3))


def post_process(x):
    constraint = 0.9

    x = 1. / (torch.exp(-x) + 1.)  # [0.05, 0.95]
    x *= 2.  # [0.1, 1.9]
    x -= 1.  # [-0.9, 0.9]
    x /= constraint  # [-1, 1]
    x += 1.  # [0, 2]
    x /= 2.  # [0, 1]
    return x
