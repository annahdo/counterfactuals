import os
import numpy as np

import torch
import torchvision
from typing import TypeVar, Tuple

Tensor = TypeVar('torch.tensor')


def torch_to_image(tensor: Tensor,
                   mean: np.array = np.array([0.]),
                   std: np.array = np.array([1.])) -> np.array:
    """
    Helper function to convert torch tensor containing input data into image.
    """
    tensor = tensor.contiguous().squeeze().detach().cpu()

    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)

    img = tensor.squeeze().numpy()
    if img.shape == 3:
        img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    else:
        img = img * std + mean

    return np.clip(img, 0, 1)


def expl_to_image(heatmap: Tensor) -> np.array:
    """
    Helper image to convert torch tensor containing a heatmap into image.
    """
    img = heatmap.squeeze().data.cpu().numpy()

    img = img / np.max(np.abs(img))  # divide by maximum
    img = np.maximum(-1, img)
    img = np.minimum(1, img) * 0.5  # clamp to -1 and divide by two -> range [-0.5, 0.5]
    img = img + 0.5

    return img


def make_dir(directory_name: str) -> str:
    """
    create directories and return directory name with slash at the end
    """
    if not directory_name:
        return ''
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if directory_name[-1] != '/':
        directory_name = directory_name + '/'

    return directory_name


def save_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    loss: float = None,
                    epoch: int = None,
                    acc: float = None):
    """
    Save checkpoint including model state dict
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    state = {
        'state_dict': state_dict,
        'loss': loss,
        'epoch': epoch,
        'acc': acc,
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    device: str) -> Tuple[float, int, float]:
    """
    load state dict for a model
    return other parameters saved in checkpoint
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: no model found at {checkpoint_path}")
        print('-' * 30)
        return None, None, None
    print(f"Loading checkpoint at {checkpoint_path} ...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    acc = None
    if 'acc' in checkpoint.keys():
        acc = checkpoint['acc']
    return loss, epoch, acc


def get_transforms(data_shape: Tuple) -> torchvision.transforms.Compose:
    """
    transforms for loading an image in RGB or grayscale and making is into a torch tensor
    """
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))])
    if data_shape[0] == 1:
        transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), transforms])
    else:
        transforms = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                                                     transforms])

    return transforms
