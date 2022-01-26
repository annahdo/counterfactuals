import torch.nn as nn
import torch
from normf.nn.base import NeuralNet
import torchvision


class MNIST_CNN(NeuralNet):

    def __init__(self, in_channels=1, num_classes=10):
        """CNN Builder."""
        super(MNIST_CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


class CNN(NeuralNet):
    """CNN."""

    def __init__(self, in_channels=3, num_classes=2, flattened_size=16384):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


class CelebA_CNN(CNN):
    """CNN."""

    def __init__(self, in_channels=3, num_classes=2, flattened_size=16384):
        """CNN Builder."""
        super(CelebA_CNN, self).__init__(in_channels=in_channels, num_classes=num_classes,
                                         flattened_size=flattened_size)


class CheXpert_CNN(CNN):
    def __init__(self, in_channels=1, num_classes=2, flattened_size=65536):
        """CNN Builder."""
        super(CheXpert_CNN, self).__init__(in_channels=in_channels, num_classes=num_classes,
                                           flattened_size=flattened_size)
