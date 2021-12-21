import os
import torch
import wandb
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision.utils import save_image


class DeterministicWarmup(object):
    """ Linear deterministic warm-up """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t


def log_images_toy(x_recon: torch.Tensor, x_sample: torch.Tensor, epoch: int) -> None:
    """Log reconstruction and sample to wandb"""
    name = f'./log_images/img_{epoch}.png'

    # ensure both tensors are on cpu
    if x_recon.is_cuda or x_sample.is_cuda:
        x_recon = x_recon.detach().cpu()
        x_sample = x_sample.detach().cpu()

    # create plots
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(x_recon[:, 0], x_recon[:, 1], '.')
    ax[0].set_title('Reconstruction')
    ax[1].plot(x_sample[:, 0], x_sample[:, 1], '.')
    ax[1].set_title('Sample', fontsize=16)
    f.savefig(name, transparent=True, bbox_inches='tight')
    plt.close()

    # log plots
    wandb.log({
        "visualization": wandb.Image(name)
    }, commit=True)

    # remove logged images
    os.remove(name)


def lambda_lr(n_epochs, offset, delay):
    """
    Creates learning rate step function for LambdaLR scheduler.
    Stepping starts after "delay" epochs and will reduce LR to 0 when "n_epochs" has been reached
    Offset is used continuing training models.
    """
    if (n_epochs - delay) == 0:
        raise Exception("Error: delay and n_epochs cannot be equal!")
    return lambda epoch: 1 - max(0, epoch + offset - delay)/(n_epochs - delay)


def bce_loss(r, x):
    """ Binary Cross Entropy Loss """
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def plt_img_save(img, name='log_image.png'):
    N = img.shape[0]
    if N >= 16:
        f, ax = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            idx = i*2
            for j in range(2):
                ax[j, i].imshow(np.reshape(img[idx+j, :], (28, 28)), cmap='gray')
                ax[j, i].axis('off')
    else:
        f, ax = plt.subplots(1, N, figsize=(16, 4))
        for i in range(N):
            ax[i].imshow(np.reshape(img[i, :], (28, 28)), cmap='gray')
            ax[i].axis('off')
    
    f.savefig(name, transparent=True, bbox_inches='tight')
    plt.close()


def log_images(x_recon, x_sample, epoch):
    convert_img(x_recon, "recon", epoch)
    convert_img(x_sample, "sample", epoch)

    # Log the images to wandb
    name_1 = f"recon{epoch}.png"
    name_2 = f"sample{epoch}.png"
    wandb.log({
        "Reconstruction": wandb.Image(name_1),
        "Sample": wandb.Image(name_2)
    }, commit=True)

    # Delete the logged images
    os.remove(name_1)
    os.remove(name_2)


def log_image_flow(x_sample, epoch):
    # Log the images to wandb
    name = f"sample{epoch}.jpg"
    save_image(x_sample, fp=name, nrow=8)
    wandb.log({
        "Sample": wandb.Image(name)
    }, commit=True)

    # Delete the logged images
    os.remove(name)


def convert_img(img, img_name, epoch):
    name_jpg = img_name + str(epoch) + '.jpg'
    name_png = img_name + str(epoch) + '.png'

    # Save batch as single image
    save_image(img, name_jpg)

    # Load image
    imag = image.imread(name_jpg)[:, :, 0]

    # Delete image
    os.remove(name_jpg)

    # Save image as proper plots
    plt_img_save(imag, name=name_png)
