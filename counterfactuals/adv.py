from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from typing import TypeVar, Dict

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from counterfactuals.utils import make_dir, get_transforms, torch_to_image, expl_to_image
from counterfactuals.plot import plot_grid_part
from counterfactuals.generative_models.base import GenerativeModel
from counterfactuals.classifiers.base import NeuralNet

Tensor = TypeVar('torch.tensor')

matplotlib.use('Agg')


def adv_attack(g_model: GenerativeModel,
               classifier: NeuralNet,
               device: str,
               attack_style: str,
               data_info: Dict,
               num_steps: int,
               lr: float,
               save_at: float,
               target_class: int,
               image_path: str,
               result_dir: str,
               maximize: bool) -> None:
    """
    prepare adversarial attack in X or Z
    run attack
    save resulting adversarial example/counterfactual
    """
    # load image
    transforms = get_transforms(data_info["data_shape"])
    x = transforms(Image.open(image_path)).to(device)

    # define parameters that will be optimized
    params = []
    if attack_style == "z":
        # define z as params for derivative wrt to z
        z = g_model.encode(x)
        z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
        x_org = x.detach().clone()
        z_org = [z_i.clone() for z_i in z] if isinstance(z, list) else z.clone()

        if type(z) == list:
            for z_part in z:
                z_part.requires_grad = True
                params.append(z_part)
        else:
            z.requires_grad = True
            params.append(z)
    else:
        # define x as params for derivative wrt x
        x_org = x.clone()
        x.requires_grad = True
        params.append(x)
        z = None

    print("\nRunning counterfactual search in Z ..." if attack_style == 'z'
          else "Running conventional adv attack in X ...")
    optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.0)

    # run the adversarial attack
    x_prime = run_adv_attack(x, z, optimizer, classifier, g_model, target_class,
                             attack_style, save_at, num_steps, maximize)

    if x_prime is None:
        print("Warning: Maximum number of iterations exceeded! Attack did not reach target value, returned None.")
        return

    # save results
    result_dir = make_dir(result_dir)
    image_name = image_path.split('/')[-1].split('.')[0]
    data_shape = data_info["data_shape"]
    cmap_img = "jet" if data_shape[0] == 3 else "gray"

    # calculate heatmap as difference dx between original and adversarial/counterfactual
    # TODO: dx to original or projection?
    heatmap = torch.abs(x_org - x_prime).sum(dim=0).sum(dim=0)

    all_images = [torch_to_image(x_org)]
    titles = ["$x$", "$x^\prime$", "$\delta x$"]
    cmaps = [cmap_img, cmap_img, 'coolwarm']
    if attack_style == 'z':
        all_images.append(torch_to_image(g_model.decode(z_org)))
        titles = ["$x$", "$g(g^{-1}(x))$", "$x^\prime$", "$\delta x$"]
        cmaps = [cmap_img, cmap_img, cmap_img, 'coolwarm']

    all_images.append(torch_to_image(x_prime))
    all_images.append(expl_to_image(heatmap))

    _ = plot_grid_part(all_images, titles=titles, images_per_row=4, cmap=cmaps)
    plt.subplots_adjust(wspace=0.03, hspace=0.01, left=0.03, right=0.97, bottom=0.01, top=0.95)

    g_model_name = f"_{type(g_model).__name__}" if g_model is not None else ""
    plt.savefig(result_dir + f'overview_{image_name}_{attack_style}{g_model_name}_save_at_{save_at}.png')


def run_adv_attack(x: Tensor,
                   z: Tensor,
                   optimizer: Optimizer,
                   classifier: NeuralNet,
                   g_model: GenerativeModel,
                   target_class: int,
                   attack_style: str,
                   save_at: float,
                   num_steps: int,
                   maximize: bool) -> Tensor:
    """
    run optimization process on x or z for num_steps iterations
    early stopping when save_at is reached
    if not return None
    """
    target = torch.LongTensor([target_class]).to(x.device)

    softmax = torch.nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    with tqdm(total=num_steps) as progress_bar:
        for step in range(num_steps):
            optimizer.zero_grad()

            if attack_style == "z":
                x = g_model.decode(z)

            # assert that x is a valid image
            x.data = torch.clip(x.data, min=0.0, max=1.0)

            if "UNet" in type(classifier).__name__:
                _, regression = classifier(x)
                # minimize negative regression to maximize regression
                loss = -regression if maximize else regression

                progress_bar.set_postfix(regression=regression.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                if (maximize and regression.item() > save_at) or (not maximize and regression.item() < save_at):
                    return x

            else:
                prediction = classifier(x)
                acc = softmax(prediction)[torch.arange(0, x.shape[0]), target]
                loss = loss_fn(prediction, target)

                progress_bar.set_postfix(acc_target=acc.item(), loss=loss.item(), step=step + 1)
                progress_bar.update()

                # early stopping
                if acc > save_at:
                    return x

            loss.backward()
            optimizer.step()

    return None
