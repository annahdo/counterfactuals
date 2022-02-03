import torch
import torch.distributions as distributions
from typing import Tuple, Dict

from counterfactuals.generative_models.base import GenerativeModel
from counterfactuals.generative_models.flows.glow import Glow
from counterfactuals.generative_models.flows.realnvp import realNVP
from counterfactuals.generative_models.flows.utils import Hyperparameters
from counterfactuals.generative_models.gans.dcgan import dcGAN
from counterfactuals.generative_models.gans.pgan import pGAN
from counterfactuals.generative_models.gans.utils import make_find_z_fun
from counterfactuals.generative_models.vaes.vae import VAE_CelebA, VAE_MNIST


def get_generative_model(generative_model_type: str,
                         data_info: Dict,
                         device: str) -> Tuple[GenerativeModel, str]:
    """
    Select and create generative model based on type (Flow, GAN or VAE) and data set
    """
    data_shape, n_bits, data_set = data_info["data_shape"], data_info["n_bits"], data_info["data_set"]

    if generative_model_type == "Flow":
        if data_set in ["CelebA", "CheXpert", "Mall"]:

            generative_model = Glow(
                in_channel=data_shape[0], n_flow=32, n_block=4,
                affine=False,
                conv_lu=True,
                data_info=data_info)
            return generative_model, "Glow"

        elif data_set == "MNIST":

            hps = Hyperparameters(base_dim=64, res_blocks=4, bottleneck=False, skip=True,
                                  weight_norm=True, coupling_bn=True, affine=True, scale_reg=5e-5)

            prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
            generative_model = realNVP(prior, hps, data_info)
            return generative_model, "realNVP"
        else:
            assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    elif generative_model_type == "GAN":
        if data_set == "MNIST":
            generative_model = dcGAN(data_info=data_info, find_z=make_find_z_fun(max_steps=3000, lr=0.1, diff=1e-3))
            return generative_model, "dcGAN"
        elif data_set == "CelebA":
            generative_model = pGAN(data_info=data_info, find_z=make_find_z_fun(max_steps=2000, lr=0.1, diff=1e-3))
            return generative_model, "pGAN"
        else:
            assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    elif generative_model_type == "VAE":
        if data_set == "MNIST":
            generative_model = VAE_MNIST(data_info=data_info)
            return generative_model, "cVAE"

        elif data_set == "CelebA":
            generative_model = VAE_CelebA(in_channels=3, latent_dim=128, data_info=data_info)
            return generative_model, "cVAE"
        else:
            assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    else:
        assert False, f"ERROR: Generative model type {generative_model_type} unknown."
