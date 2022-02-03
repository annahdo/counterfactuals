# Counterfactuals

We create counterfactual explanations for image data by doing gradient ascent in the latent space of a generative model.
As opposed to gradient ascent in the image space, which produces conventional adversarial examples, our method finds counterfactuals that are structurally different to the original image and resemble samples from the target class.

We apply our method on different data sets: 
- MNIST (hand written digits)
- CelebA (celebrity faces)
- CheXpert (chest x-rays)
- Mall (pedestrians in a shopping mall)

using different generative models:
- Normalizing Flows (realNVP and Glow)
- Generative Adversarial Networks (progressive GAN and deep convolutional GAN)
- Variational Auto-Encoder (convolutional VAE)

and different (classification) models leading the gradient ascent updates:
- 10 class classifier for MNIST
- binary classifiers (blond/not-blond for CelebA and cardiomegaly/healthy for CheXpert)
- a U-Net for the number of pedestrians present in Mall images

### Install


### Download
The pretrained networks can be downloaded from

### Run
Conventional adversarial examples can be produced with following code:
```
# conventional adv attacks
# MNIST
python main.py main data-set --name MNIST classifier adv-attack --image_path images/MNIST_img_${IDX}.png --target_class 9 --lr 5e-4 --attack_type conv --num_steps 2000 --save_at 0.99
# CelebA
python main.py main data-set --name CelebA classifier --path checkpoints/classifiers/CelebA_CNN_9.pth adv-attack --image_path images/CelebA_img_${IDX}.png --target_class 1 --lr 5e-3 --num_steps 1000 --save_at 0.99
# CheXpert
python main.py main data-set --name CheXpert classifier adv-attack --image_path images/CheXpert_img_${IDX}.png --target_class 1 --lr 5e-4 --num_steps 1000 --save_at 0.99
# Mall
python main.py main data-set --name Mall classifier --path checkpoints/classifiers/Mall_UNet_ultrasmall.pth adv-attack --image_path images/Mall_img_${IDX}.png --lr 1e-4 --num_steps 5000 --save_at 10 --maximize True
python main.py main data-set --name Mall classifier --path checkpoints/classifiers/Mall_UNet_ultrasmall.pth adv-attack --image_path images/Mall_img_${IDX}.png --lr 1e-4 --num_steps 5000 --save_at 0.01 --maximize False

```
Counterfactuals can be produced with following code:
```
# NORMALIZING FLOWS
# MNIST realNVP
python main.py main data-set --name MNIST classifier generative-model --g_type Flow adv-attack --image_path images/MNIST_img_${IDX}.png --target_class 9 --lr 5e-2 --num_steps 2000 --save_at 0.99
# CelebA Glow
python main.py main data-set --name CelebA classifier --path checkpoints/classifiers/CelebA_CNN_9.pth generative-model --g_type Flow adv-attack --image_path images/CelebA_img_${IDX}.png --target_class 1 --lr 7e-4 --num_steps 1000 --save_at 0.99
# CheXpert Glow
python main.py main data-set --name CheXpert classifier generative-model --g_type Flow adv-attack --image_path images/CheXpert_img_${IDX}.png --target_class 1 --lr 5e-3 --num_steps 1000 --save_at 0.99
# Mall Glow
python main.py main data-set --name Mall classifier --path checkpoints/classifiers/Mall_UNet_ultrasmall.pth generative-model --g_type Flow adv-attack --image_path images/Mall_img_${IDX}.png --lr 5e-3 --num_steps 5000 --save_at 10 --maximize True
python main.py main data-set --name Mall classifier --path checkpoints/classifiers/Mall_UNet_ultrasmall.pth generative-model --g_type Flow adv-attack --image_path images/Mall_img_${IDX}.png --lr 5e-3 --num_steps 5000 --save_at 0.01 --maximize False

# GAN
# MNIST dcGAN
python main.py main data-set --name MNIST classifier generative-model --g_type Flow adv-attack --image_path images/MNIST_img_${IDX}.png --target_class 9 --lr 1e-2 --num_steps 1000
# CelebA pGAN
python main.py main data-set --name CelebA classifier --path checkpoints/classifiers/CelebA_CNN_9.pth generative-model --g_type Flow adv-attack --image_path images/CelebA_img_${IDX}.png --target_class 1 --lr 1e-2 --num_steps 1000

# VAE
# MNIST
python main.py main data-set --name MNIST classifier generative-model --g_type Flow adv-attack --image_path images/MNIST_img_${IDX}.png --target_class 9 --num_steps 1000 --lr 1e-2

```

When using this Code please cite

```
@inproceedings{dombrowski2021diffeomorphic,
  title={Diffeomorphic Explanations with Normalizing Flows},
  author={Dombrowski, Ann-Kathrin and Gerken, Jan E and Kessel, Pan},
  booktitle={ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
  year={2021}
}
```