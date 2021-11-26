import os
import torch
import wandb
import matplotlib.pyplot as plt


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


def log_images(x_recon: torch.Tensor, x_sample: torch.Tensor, epoch: int) -> None:
    """Log reconstruction and sample to wandb"""
    name = f'./log_images/img_{epoch}.png'

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
