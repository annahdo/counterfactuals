import torch.optim as optim
import torch
from tqdm import tqdm
from typing import TypeVar
from counterfactuals.generative_models.base import GenerativeModel

Tensor = TypeVar('torch.tensor')
LossFun = TypeVar('torch.nn._Loss')


def make_find_z_fun(max_steps: int = 5000,
                    lr: float = 0.1,
                    diff: float = 1e-3,
                    loss_fun: LossFun = torch.nn.MSELoss()):
    """
    initializes a function with which one can find the GAN latent representations for a given image

    :param max_steps: maximum number of iterations during optimization
    :param lr: learning rate
    :param diff: early stopping loss between image
    :param loss_fun: loss function to optimize during search process
    :return: function
    """

    def find_z(generator: GenerativeModel,
               z: Tensor,
               img: Tensor) -> Tensor:
        """
        find the latent representation of an image, so that when decoding the latent representation decoded=GAN(latent)
        the decoded image is very close to the source image

        :param generator: GAN model
        :param z: initial latent representation (random)
        :param img: image
        :return: latent representation that corresponds to image
        """
        z = z.clone()
        z.requires_grad = True
        optimizer = optim.Adam([z], lr=lr)

        print("Optimizing latent representation ...")

        with tqdm(total=max_steps) as progress_bar:
            for step in range(max_steps):
                optimizer.zero_grad()
                x = generator.forward(z)
                x = torch.clip(x, min=0, max=1)

                loss = loss_fun(x, img)

                progress_bar.set_postfix(loss=loss.item(), step=step + 1)
                progress_bar.update()

                if loss < diff:
                    break
                loss.backward()

                optimizer.step()

        return z.detach()

    return find_z
