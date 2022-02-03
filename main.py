import click
from argparse import Namespace
import torch
import ast
import os

import counterfactuals.adv
import counterfactuals.classifiers.cnn as classifiers
import counterfactuals.classifiers.unet as unet
from counterfactuals.utils import load_checkpoint
from counterfactuals.data import get_data_info
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
@click.option('--g_type', default='Flow', type=click.Choice(['Flow', 'GAN', 'VAE']))
@click.option('--path', default=None, help='Path to trained model')
@click.pass_context
def generative_model(ctx, g_type, path):
    g_model, g_model_type = get_generative_model(g_type, ctx.obj.data_info, device=ctx.obj.device)
    data_set = ctx.obj.data_info["data_set"]

    print("-" * 30)
    print(f"GENERATIVE MODEL: {g_type}")
    if not path:
        path = f"checkpoints/generative_models/{data_set}_{g_model_type}.pth"

    _, _, _ = load_checkpoint(path, g_model, ctx.obj.device)

    g_model.to(ctx.obj.device)

    ctx.obj.generative_model_name = path.split('/')[-1].split('.pth')[0]
    ctx.obj.generative_model = g_model
    ctx.obj.generative_model_path = path
    ctx.obj.generative_model_type = g_model_type


@main.command()
@click.option('--path', default=None, help='Path to trained model')
@click.option('--unet_type', default="ultrasmall",
              type=click.Choice(['small', 'ultrasmall']), help='Select U-Net type if using U-Net')
@click.pass_context
def classifier(ctx, path, unet_type):
    print("-" * 30)
    print("CLASSIFIER")

    data_set = ctx.obj.data_info["data_set"]

    if data_set == "Mall":
        c_type = "Mall_UNet"
        kwargs = {'unet_type': unet_type}
        classifier = getattr(unet, c_type)(**kwargs)
    else:
        c_type = data_set + "_CNN"
        classifier = getattr(classifiers, c_type)()

    if path is None:
        path = f"checkpoints/classifiers/{c_type}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)

    _, _, _ = load_checkpoint(path, classifier, ctx.obj.device)

    classifier.to(ctx.obj.device)
    classifier.eval()

    ctx.obj.classifier_type = c_type
    ctx.obj.classifier = classifier


@main.command()
@click.option('--attack_style', default='z', type=click.Choice(['z', 'conv']),
              help="Find conventional adversarial examples in X or counterfacluals in Z")
@click.option('--num_steps', default=5000, type=int, help='Maximum number of optimizer steps')
@click.option('--lr', default=5e-2, type=float, help='Learning rate')
@click.option('--save_at', default=0.99, type=float, help="Stop attack when acc of target class or regression "
                                                          "value reaches this value")
@click.option('--target_class', default=1, type=int,
              help='target class that the modified image should be classified as')
@click.option('--image_path', default="images/CelebA_img_1.png", type=str,
              help='Path to image on which to run the attack')
@click.option('--result_dir', default="results", type=str, help="Directory to save results to")
@click.option('--maximize', default=True, type=bool, help="Set to False if you want to minimize the "
                                                          "regression value. relevant for U-Net only")
@click.pass_context
def adv_attack(ctx, attack_style, num_steps, lr, save_at, target_class, image_path, result_dir, maximize):
    print("-" * 30)
    c_model = ctx.obj.classifier
    c_model.eval()

    if attack_style == "z":
        g_model = ctx.obj.generative_model
        g_model.eval()
    else:
        g_model = None

    counterfactuals.adv.adv_attack(g_model, c_model, ctx.obj.device,
                                   attack_style, ctx.obj.data_info, num_steps, lr, save_at,
                                   target_class, image_path, result_dir, maximize)


if __name__ == '__main__':
    cli.add_command(main)
    cli()
