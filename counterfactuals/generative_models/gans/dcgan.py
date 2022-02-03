"""
code adapted from https://github.com/AKASHKADEL/dcgan-mnist
"""

import torch.nn as nn
import torch
from counterfactuals.generative_models.base import GenerativeModel


class dcGAN(GenerativeModel):
    def __init__(self, nc=1, nz=100, ngf=32, ndf=32, data_info=None, find_z=None):
        super(dcGAN, self).__init__(g_model_type="GAN", data_info=data_info)
        self.nz = nz
        self.find_z = find_z
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = torch.randn(x.shape[0], self.nz, 1, 1).to(x.device)

        return self.find_z(self.generator, z, x)

    def decode(self, z):
        output = self.generator(z)
        return output

    def sample(self, batchsize):
        return torch.randn(batchsize, self.nz, 1, 1)
