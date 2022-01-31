import abc
import torch.nn as nn
from typing import TypeVar, Dict

Tensor = TypeVar('torch.tensor')


class GenerativeModel(nn.Module):
    """
    Base class for all generative models (VAEs, GANs, Flows)
    """

    def __init__(self,
                 g_model_type: str,
                 data_info: Dict):
        super().__init__()

        self.g_model_type = g_model_type
        self.data_info = data_info
        self.data_set = data_info["data_set"]

    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass
