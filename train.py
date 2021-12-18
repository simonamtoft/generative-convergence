import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

from models import DRAW, VariationalAutoencoder, LadderVAE, \
    Flow, AffineCouplingBijection, ActNormBijection, Reverse, \
    ElementwiseParams, StandardNormal

from lib import seed_everything, get_args
from trainers import train_draw, train_vae #, train_flow


CONFIG = {
    'optimizer': 'adam',
    'batch_size': 128,
    'lr': 1e-3,
    'lr_decay': {
        'n_epochs': 1000,
        'delay': 150,
        'offset': 0,
    },
    'kl_warmup': 100,
}

DIRS = ['saved_models', 'log_images', 'losses']
WANDB_NAME = "generative-convergence-mnist"


def setup_and_train(config: dict, mute: bool, x_shape: list) -> tuple:
    if 'vae' in config['model']:
        config['as_beta'] = True

        # load lvae or vae
        if config['model'] == 'lvae':
            config['h_dim'] = [128, 128, 128]
            config['z_dim'] = [2, 2, 2]
            model = LadderVAE(config, x_shape).to(config['device'])
        else:
            config['h_dim'] = [128, 128, 128]
            config['z_dim'] = 2
            model = VariationalAutoencoder(config, x_shape).to(config['device'])

        # perform training
        print(json.dumps(config, sort_keys=False, indent=4) + '\n')
        train_losses, val_losses = train_vae(train_loader, val_loader, model, config, mute, WANDB_NAME)
    if config['model'] == 'draw':
        config['h_dim'] = 256
        config['z_dim'] = 32
        config['T'] = 10
        config['N'] = 12
        config['attention'] = 'base'

        # Instantiate model
        model = DRAW(config, x_shape).to(config['device'])

        # perform training
        print(json.dumps(config, sort_keys=False, indent=4) + '\n')
        train_losses, val_losses = train_draw(train_loader, val_loader, model, config, mute, WANDB_NAME)
    return train_losses, val_losses


# Define transformation
def tmp_lambda(x):
    return torch.bernoulli(x)


if __name__ == '__main__':
    # add arguments to config
    config, args = get_args([], CONFIG)

    # create necessary folders
    for directory in DIRS:
        if not os.path.isdir(f'./{directory}'):
            print(f'Creating directory "{directory}"...')
            os.mkdir(f'./{directory}')

    # check if cuda
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        config['device'] = 'cpu'
        kwargs = {}

    # get train and validation data
    data_transform = Compose([
        ToTensor(),
        Lambda(tmp_lambda)
    ])
    train_data = MNIST('./', download=True, transform=data_transform)

    # split into training and validation sets
    train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000])

    # Setup data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        **kwargs
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        **kwargs
    )

    # get shape of input
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    x_shape = images.shape[2:4]

    # mute weight and biases prints
    os.environ["WANDB_SILENT"] = "true"

    # train using different seeds
    seeds = np.random.randint(0, 1e6, args['n_runs'])
    losses = {'train': [], 'val': []}
    for i in range(args['n_runs']):
        seed = seeds[i]
        print(f"\nTraining with seed {seed} ({i+1}/{args['n_runs']})")
        seed_everything(seed)
        train, val = setup_and_train(config, args['mute'], x_shape)
        losses['train'].append(train)
        losses['val'].append(val)
    print('\nFinished all training runs...')

    # save loss results to file
    filename = f'./losses/{config["model"]}_mnist_{args["n_runs"]}.json'
    print(f'Saving losses to file {filename}')
    with open(filename, 'w') as f:
        json.dump(losses, f)
    print('\nDone!')


