import os
import numpy as np
import csv


import matplotlib as mpl

import torch
import torchvision
import re


def num_from_str(test_string):
    temp = re.findall(r'\d+', test_string.split("/")[-1])
    return int(temp[0])



def get_image_files(root, img_ext=[".jpeg", ".jpg", ".png"]):
    files = os.listdir(root)
    paths = [os.path.join(root, file) for file in files if os.path.splitext(file)[1] in img_ext]
    return paths


def torch_to_image(tensor, mean=np.array([0., 0., 0., ]), std=np.array([1., 1., 1.])):
    """
    Helper function to convert torch tensor containing input data into image.
    """
    tensor = tensor.contiguous().squeeze().detach().cpu()

    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)

    img = tensor.squeeze().numpy()
    if len(mean) != 1:
        img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    else:
        img = img * std + mean

    return np.clip(img, 0, 1)


def expl_to_image(heatmap):
    """
    Helper image to convert torch tensor containing a heatmap into image.
    """
    img = heatmap.squeeze().data.cpu().numpy()

    img = img / np.max(np.abs(img))  # divide by maximum
    img = np.maximum(-1, img)
    img = np.minimum(1, img) * 0.5  # clamp to -1 and divide by two -> range [-0.5, 0.5]
    img = img + 0.5

    return img


def torch_to_numpy(tensor):
    if len(tensor.shape) == 4:
        numpy_array = tensor.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
    else:
        numpy_array = tensor.contiguous().squeeze().detach().cpu().numpy()

    return numpy_array


def get_dir(file_path):
    """
    get parent directory
    """
    return os.path.dirname(file_path)


def make_dir(directory_name):
    if not directory_name:
        return ''
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if directory_name[-1] != '/':
        directory_name = directory_name + '/'

    return directory_name


def save_img(tensors, output_dir='../data/saved_images/', mean=np.array([0.4914, 0.4822, 0.4465]),
             std=np.array([0.2023, 0.1994, 0.2010]),
             name=None):
    output_dir = make_dir(output_dir)
    for i in range(tensors.shape[0]):
        nparray = torch_to_image(tensors[i][None], mean=mean, std=std)
        img_name = name if name else f'img_{i}.png'
        mpl.image.imsave(output_dir + img_name, nparray, vmin=0., vmax=1.0)


def save_checkpoint(checkpoint_path, model, loss, epoch, acc=None):
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


def load_checkpoint(checkpoint_path, model, device):
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: no model found at {checkpoint_path}")
        print('-' * 30)
        return None, None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    acc = None
    if 'acc' in checkpoint.keys():
        acc = checkpoint['acc']

    print(f"Loading checkpoint at {checkpoint_path} ...")
    print(f"acc:      {acc}")
    print(f"loss:     {loss}")
    print(f"epoch:    {epoch}")
    print('-' * 30)

    return loss, epoch, acc


def hinge_loss(y_pred, y):
    y_true = torch.ones(y.shape, dtype=int).to(y.device)
    y_true[y != 1] = -1
    return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))


def moving_avg(current_avg, new_value, idx):
    return current_avg + (new_value - current_avg) / idx


def write_csv(header, columns, output_file):
    with open(output_file, 'w') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(header)
        for i in range(len(columns[0])):
            csvWriter.writerow([column[i] for column in columns])


def read_csv(file):
    import pandas as pd

    data = pd.read_csv(file)

    header = []
    columns = []
    for key in data.keys():
        columns.append(np.array(data[key]))
        header.append(key)

    return header, columns


def get_transforms(data_shape):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))])
    if data_shape[0] == 1:
        transforms = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), transforms])
    else:
        transforms = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                                                     transforms])

    return transforms