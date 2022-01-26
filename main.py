import click
from argparse import Namespace
import torch
import ast
import os
from PIL import Image
from matplotlib import pyplot as plt

import counterfactuals.adv
import counterfactuals.classifiers.cnn as classifiers
from counterfactuals.utils import make_dir, load_checkpoint, get_transforms, get_dir, torch_to_image
from counterfactuals.data import get_data_info
from counterfactuals.plot import plot_grid_part
from counterfactuals.generative_models.factory import get_generative_model


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.group()
def cli():
    pass


@click.group(chain=True)
@click.pass_context
def main(ctx):
    ctx.ensure_object(Namespace)
    ctx.obj.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@main.command()
@click.option('--name', default='CelebA', type=click.Choice(['MNIST', 'CelebA', 'CheXpert', 'Mall']))
@click.pass_context
def data_set(ctx, name):
    print("-" * 30)
    print(f"DATASET: {name}")

    data_info = get_data_info(name)
    ctx.obj.data_info = data_info


@main.command()
@click.option('--type', default='Flow', type=click.Choice(['Flow', 'GAN', 'VAE']))
@click.option('--path', default=None, help='Path to trained model')
@click.pass_context
def generative_model(ctx, type, path):
    g_model, g_model_type = get_generative_model(type, ctx.obj.data_info, device=ctx.obj.device)
    data_set = ctx.obj.data_info["data_set"]

    print("-" * 30)
    print(f"Generative model: {data_set}_{g_model_type}")
    if not path:
        path = f"checkpoints/generative_models/{data_set}_{g_model_type}.pth"

    _, _, _ = load_checkpoint(path, g_model, ctx.obj.device)

    ctx.obj.generative_model_name = path.split('/')[-1].split('.pth')[0]
    ctx.obj.generative_model = g_model
    ctx.obj.generative_model_path = path
    ctx.obj.generative_model_type = g_model_type


@main.command()
@click.option('--type', default=None)
@click.option('--path', default="checkpoints/classifiers/CelebA_CNN_9.pth", help='Path to trained model')
@click.pass_context
def classifier(ctx, type, path):
    print("-" * 30)
    print("CLASSIFIER")

    data_set = ctx.obj.data_info["data_set"]
    if type is None:
        type = "Mall_Unet" if data_set == "Mall" else data_set + "_CNN"

    classifier = getattr(classifiers, type)()

    if path is None:
        path = f"checkpoints/classifiers/{type}.pth"
        os.makedirs(get_dir(path), exist_ok=True)

    _, _, _ = load_checkpoint(path, classifier, ctx.obj.device)

    classifier.to(ctx.obj.device)
    classifier.eval()

    ctx.obj.classifier_type = type
    ctx.obj.classifier = classifier


@main.command()
@click.option('--attack_style', default='z', type=click.Choice(['z', 'conv']),
              help="Find conventional adversarial examples in X or counterfacluals in Z")
@click.option('--num_steps', default=5000, type=int, help='Maximum number of optimizer steps')
@click.option('--lr', default=1e-2, type=float, help='Learning rate')
@click.option('--save_at', default=0.99, type=float, help='Stop attack when acc of target class reaches this value')
@click.option('--target_class', default=1, type=int,
              help='target class that the modified image should be classified as')
@click.option('--image_path', default="images/CelebA_img_1.png", type=str,
              help='Path to image on which to run the attack')
@click.option('--result_dir', default="results", type=str, help="Directory to save results to")
@click.pass_context
def adv_attack(ctx, attack_style, num_steps, lr, save_at, target_class, image_path, result_dir):
    print("-" * 30)
    print("Running counterfactual search in Z" if attack_style == 'z' else "Running conventional adv attack in X")

    generative_model = ctx.obj.generative_model
    classifier_model = ctx.obj.classifier
    generative_model.eval()
    classifier_model.eval()

    counterfactuals.adv.adv_attack(generative_model, classifier_model, ctx.obj.device,
                                   attack_style, ctx.obj.data_info, num_steps, lr, save_at,
                                   target_class, image_path, result_dir)


if __name__ == '__main__':
    cli.add_command(main)
    cli()
