import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, \
    Omniglot
from torchvision.transforms import Compose, ToTensor, \
    Lambda, Normalize, CenterCrop, Resize
from torchvision.transforms.functional import invert

from models import DRAW, VariationalAutoencoder, LadderVAE, \
    Flow, AffineCouplingBijection, ActNormBijection, Reverse, \
    ElementwiseParams, StandardNormal, Squeeze2d, Augment, \
    UniformDequantization, DenseNet, ElementwiseParams2d, \
    StandardUniform, Slice, ActNormBijection2d, Conv1x1, \
    DRAW2, DRAW3, DRAW4

from lib import seed_everything, get_args
from trainers import train_draw, train_vae, train_flow

DATA_FOLDER = './data/'

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

DIRS = ['saved_models', 'log_images', 'losses', 'data']
# WANDB_NAME = "generative-convergence-mnist"
WANDB_NAME = "draw-tests"


def setup_and_train(config: dict, mute: bool, x_shape: list, train_loader: DataLoader, val_loader: DataLoader) -> tuple:
    if 'vae' in config['model']:
        config['as_beta'] = True
        x_dim = x_shape[0] * x_shape[1]

        # load lvae or vae
        if config['model'] == 'lvae':
            config['h_dim'] = [512, 256, 256]
            config['z_dim'] = [64, 64, 64]
            model = LadderVAE(config, x_dim).to(config['device'])
        else:
            config['h_dim'] = [512, 256, 128, 64]
            config['z_dim'] = 64
            model = VariationalAutoencoder(config, x_dim).to(config['device'])

        # perform training
        print(json.dumps(config, sort_keys=False, indent=4) + '\n')
        train_losses, val_losses = train_vae(train_loader, val_loader, model, config, mute, WANDB_NAME)
    elif 'draw' in config['model']:
        config['h_dim'] = 256
        config['z_dim'] = 32
        config['T'] = 10
        config['N'] = 12       # only used for filterbank...
        config['attention'] = 'base'

        # Instantiate model
        if config['model'][-1] == 'w':
            print("DRAW")
            model = DRAW(config, x_shape).to(config['device'])
        elif config['model'][-1] == '2':
            print("DRAW2")
            model = DRAW2(config, x_shape).to(config['device'])
        elif config['model'][-1] == '3':
            print("DRAW3")
            model = DRAW3(config, x_shape).to(config['device'])
        elif config['model'][-1] == '4':
            print("DRAW4")
            model = DRAW4(config, x_shape).to(config['device'])

        # perform training
        print(json.dumps(config, sort_keys=False, indent=4) + '\n')
        train_losses, val_losses = train_draw(train_loader, val_loader, model, config, mute, WANDB_NAME)
    elif config['model'] == 'flow':
        def net(channels):
            return nn.Sequential(
                DenseNet(
                    in_channels=channels//2,
                    out_channels=channels,
                    num_blocks=1,
                    mid_channels=64,
                    depth=8,
                    growth=16,
                    dropout=0.0,
                    gated_conv=True,
                    zero_init=True
                ),
                ElementwiseParams2d(2)
            )

        model = Flow(
            base_dist=StandardNormal((8, 7, 7)),
            transforms=[
                UniformDequantization(num_bits=1),
                Augment(StandardUniform((1, 28, 28)), x_size=1),
                AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                Squeeze2d(), Slice(StandardNormal((4, 14, 14)), num_keep=4),
                AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                Squeeze2d(), Slice(StandardNormal((8, 7, 7)), num_keep=8),
                AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
                AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
                AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
                AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
            ]
        ).to(config['device'])
        
        # alter some training params
        config['lr'] = 1e-4

        # perform training
        print(json.dumps(config, sort_keys=False, indent=4) + '\n')
        train_losses, val_losses = train_flow(train_loader, val_loader, model, config, mute, WANDB_NAME)
    return train_losses, val_losses


# binarize transformation
def tmp_lambda(x):
    return torch.bernoulli(x)


if __name__ == '__main__':
    # add arguments to config
    config, args = get_args(['mnist', 'omniglot'], CONFIG, data_default='mnist')

    # create necessary folders
    for directory in DIRS:
        if not os.path.isdir(f'./{directory}'):
            print(f'Creating directory "{directory}"...')
            os.mkdir(f'./{directory}')

    # setup data arguments
    kwargs = {}

    # enable cuda if available
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        kwargs['num_workers'] = 4
        kwargs['pin_memory'] = True
    else:
        config['device'] = 'cpu'

    # Set the standard shape of the data
    data_dim = 28
    data_shape = [data_dim, data_dim]

    # Define data transform
    data_transform = [
        ToTensor(),
        Lambda(tmp_lambda)
    ]

    # load in the data
    if config['dataset'] == 'mnist':
        # get MNIST data
        train_data = MNIST(DATA_FOLDER, download=True, transform=Compose(data_transform))

        # define split
        data_split = [50000, 10000]
    elif config['dataset'] == 'omniglot':
        # Add resize transformation
        data_transform.append(Resize(data_dim))
        data_transform.append(CenterCrop(data_dim))
        data_transform.append(invert)

        # get Omniglot data
        train_data = Omniglot(DATA_FOLDER, download=True, transform=Compose(data_transform))

        # define split
        data_split = [15424, 3856]
    
    # split into training and validation sets
    train_set, val_set = torch.utils.data.random_split(train_data, data_split)

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

    # mute weight and biases prints
    if args['mute']:
        os.environ["WANDB_SILENT"] = "true"

    # train using different seeds
    seeds = np.random.randint(0, 1e6, args['n_runs'])
    losses = {'train': [], 'val': []}
    for i in range(args['n_runs']):
        seed = seeds[i]
        print(f"\nTraining with seed {seed} ({i+1}/{args['n_runs']})")
        seed_everything(seed)
        train, val = setup_and_train(config, args['mute'], data_shape, train_loader, val_loader)
        losses['train'].append(train)
        losses['val'].append(val)
    print('\nFinished all training runs...')

    # save loss results to file
    filename = f'./losses/{config["model"]}_mnist_{args["n_runs"]}.json'
    print(f'Saving losses to file {filename}')
    with open(filename, 'w') as f:
        json.dump(losses, f)
    print('\nDone!')


