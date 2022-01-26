import abc
import torch
import torch.nn as nn
import torch.distributions as distributions


from counterfactuals.generative_models.flows.glow import Glow
from counterfactuals.generative_models.flows.realnvp import RealNVP
from counterfactuals.generative_models.flows.utils import Hyperparameters

def get_generative_model(generative_model_type, data_info, device):
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

            hps = Hyperparameters(
                base_dim=64,
                res_blocks=4,
                bottleneck=0,
                skip=1,
                weight_norm=1,
                coupling_bn=1,
                affine=1,
                scale_reg=5e-5)

            prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
            generative_model = RealNVP(data_info=data_info, prior=prior, hps=hps)

            return generative_model, "RealNVP"
        else:
            assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    elif generative_model_type == "GAN":
        assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    elif generative_model_type == "VAE":
        assert False, f"ERROR: Combination {generative_model_type} with data_set {data_set} not implemented"

    else:
        assert False, f"ERROR: Generative model type {generative_model_type} unknown."

