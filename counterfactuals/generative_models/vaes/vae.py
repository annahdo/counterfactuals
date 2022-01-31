import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar, Dict, Tuple

from counterfactuals.generative_models.base import GenerativeModel, Tensor


class VAE_CelebA(GenerativeModel):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 data_info: Dict = None,
                 **kwargs) -> None:
        super(VAE_CelebA, self).__init__(g_model_type="VAE", data_info=data_info)

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def _encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input * 2. - 1.
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def encode(self, input: Tensor) -> Tensor:
        mu, log_var = self._encode(input)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        result = (result + 1.) / 2.
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0] * 2. - 1.
        input = args[1] * 2. - 1.
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def sample(self,
               num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space
        :param num_samples: (Int) Number of samples
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        return z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAE_MNIST(GenerativeModel):
    def __init__(self,
                 z_dim: int = 10,
                 data_info: Dict = None):
        super(VAE_MNIST, self).__init__(g_model_type="VAE", data_info=data_info)

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),

        )
        self.mu = nn.Linear(32 * 3 * 3, z_dim)
        self.logvar = nn.Linear(32 * 3 * 3, z_dim)

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32 * 3 * 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        temp = self.encoder(x).view(x.shape[0], -1)
        mu = self.mu(temp)
        logvar = self.logvar(temp)

        return mu, logvar

    def encode(self, x: Tensor) -> Tensor:
        mu, logvar = self._encode(x)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        temp = self.fc(z).view(z.shape[0], 32, 3, 3)
        return self.decoder(temp)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(logvar) * torch.randn(logvar.shape)

    def sample(self, num_samples: int) -> Tensor:
        return torch.randn([num_samples, self.z_dim])

    def reconstruction_loss(self, x_org: Tensor, x_r: Tensor) -> Tensor:
        return ((x_org - x_r) ** 2).sum()

    def KL_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return (torch.exp(logvar) ** 2 + mu ** 2 - logvar - 1 / 2).sum()

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        reconstruction_loss = ((input - recons) ** 2).sum()
        KL_divergence = (torch.exp(log_var) ** 2 + mu ** 2 - log_var - 1 / 2).sum()
        loss = reconstruction_loss + KL_divergence

        return loss
