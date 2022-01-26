import abc
import torch.nn as nn

class GenerativeModel(nn.Module):
    def __init__(self, type, data_set):
        super().__init__()

        self.type = type
        self.data_set = data_set

    @abc.abstractmethod
    def encode(self, x):
        pass

    @abc.abstractmethod
    def decode(self, z):
        pass

