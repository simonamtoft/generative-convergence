import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib import get_ffjord_data, get_toy_names, \
    seed_everything, get_args
from models import DRAW, VariationalAutoencoder, LadderVAE, \
    Flow, AffineCouplingBijection, ActNormBijection, Reverse, \
    ElementwiseParams, StandardNormal
from trainers import train_vae, train_draw, train_flow


CONFIG = {
    'optimizer': 'adam',
    'train_samples': 4096*1000,
    'val_samples': 4096*1000,
    'batch_size': 4096,
    'lr': 1e-3,
    'lr_decay': {
        'n_epochs': 4000,
        'delay': 200,
        'offset': 0,
    },
    'kl_warmup': 100,
}

DIRS = ['saved_models', 'log_images', 'losses']


def setup_and_train(config: dict, mute: bool) -> tuple:
    if 'vae' in config['model']:
        config['as_beta'] = True

        # dimensions of data.
        # input is (x, y) values (2) while 
        # output of decoder should be 2 means and variances (4)
        x_dim = [2, 4] 

        # load lvae or vae
        if config['model'] == 'lvae':
            config['h_dim'] = [128, 128, 128]
            config['z_dim'] = [2, 2, 2]
            model = LadderVAE(config, x_dim).to(config['device'])
        else:
            config['h_dim'] = [128, 128, 128]
            config['z_dim'] = 2
            model = VariationalAutoencoder(config, x_dim).to(config['device'])

        # perform training
        train_losses, val_losses = train_vae(train_loader, val_loader, model, config, mute)
    elif config['model']  == 'flow':
        # instantiate model
        def net():
            return nn.Sequential(
                nn.Linear(1, 200), nn.ReLU(),
                nn.Linear(200, 100), nn.ReLU(),
                nn.Linear(100, 2), ElementwiseParams(2)
            )

        model = Flow(
            base_dist=StandardNormal((2,)),
            transforms=[
                AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
                AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
                AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
                AffineCouplingBijection(net()), ActNormBijection(2),
            ]
        ).to(config['device'])

        # perform training
        train_losses, val_losses = train_flow(train_loader, val_loader, model, config, mute)
    elif config['model']  == 'draw':
        # define some model specific config
        config['attention'] = 'base'
        config['h_dim'] = 256
        config['z_dim'] = 32
        config['T'] = 10
        config['N'] = 12

        # instantiate model
        model = DRAW(config, [1, 2]).to(config['device'])

        # perform training
        train_losses, val_losses = train_draw(train_loader, val_loader, model, config, mute)
    return train_losses, val_losses



if __name__ == '__main__':
    # add arguments to config
    config, args = get_args(get_toy_names(), CONFIG)

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
    train_data = get_ffjord_data(config['dataset'], config['train_samples'])
    val_data = get_ffjord_data(config['dataset'], config['val_samples'])

    # Setup data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        **kwargs
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        **kwargs
    )

    # mute weight and biases prints
    os.environ["WANDB_SILENT"] = "true"
    
    # print some stuff
    print(f"\nTraining of {config['model']} model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')

    # train using different seeds
    seeds = np.random.randint(0, 1e6, args['n_runs'])
    losses = {'train': [], 'val': []}
    for i in range(args['n_runs']):
        seed = seeds[i]
        print(f"\nTraining with seed {seed} ({i+1}/{args['n_runs']})")
        seed_everything(seed)
        train, val = setup_and_train(config, args['mute'])
        losses['train'].append(train)
        losses['val'].append(val)
        print(train, val)
    print('\nFinished all training runs...')

    # save loss results to file
    filename = f'./losses/{config["model"]}_{config["dataset"]}.json'
    print(f'Saving losses to file {filename}')
    with open(filename, 'w') as f:
        json.dump(losses, f)
    print('\nDone!')
