import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib import get_ffjord_data, get_toy_names
from models import DRAW, VariationalAutoencoder, LadderVAE, \
    Flow, AffineCouplingBijection, ActNormBijection, Reverse, \
    ElementwiseParams, StandardNormal
from trainers import train_vae, train_draw, train_flow


CONFIG = {
    'optimizer': 'adam',
    'train_samples': 128*1000,
    'val_samples': 128*1000,
    'batch_size': 128,
    'lr': 1e-3,
    'lr_decay': {
        'n_epochs': 4000,
        'delay': 200,
        'offset': 0,
    },
    'kl_warmup': 100,
}

DIRS = ['saved_models', 'log_images']


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description="Model training script.")
    parser.add_argument(
        '-m', 
        help='Pick which model to train (default: vae).', 
        default='vae',
        type=str,
        choices=['lvae', 'vae', 'draw', 'flow'],
        dest='model'
    )
    parser.add_argument(
        '-d', 
        help='Pick which dataset to fit to (default: 8gaussians).', 
        default='8gaussians',
        type=str,
        choices=get_toy_names(),
        dest='dataset'
    )
    parser.add_argument(
        '-e', 
        help='Pick number of epochs to train over (default: 100).', 
        default=100,
        type=int,
        dest='epochs'
    )
    parser.add_argument(
        '-mute', 
        help='Mute tqdm outputs. (mainly for bsub submit runs)', 
        action='store_true'
    )
    parser.add_argument(
        '-n',
        help='Pick number of models to train.', 
        default=1,
        type=int,
        dest='runs'
    )
    args = parser.parse_args()

    # create necessary folders
    for directory in DIRS:
        if not os.path.isdir(f'./{directory}'):
            print(f'Creating directory "{directory}"...')
            os.mkdir(f'./{directory}')

    # check if cuda
    has_cuda = torch.cuda.is_available()
    
    # add arguments to config
    config = CONFIG
    config['model'] = args.model
    config['dataset'] = args.dataset
    config['device'] = 'cuda' if has_cuda else 'cpu'
    config['epochs'] = args.epochs

    # get train and validation data
    train_data = get_ffjord_data(config['dataset'], config['train_samples'])
    val_data = get_ffjord_data(config['dataset'], config['val_samples'])

    # Setup data loaders
    kwargs = {'num_workers': 4, 'pin_memory': True} if has_cuda else {} # 4
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

    # define and train model
    if 'vae' in config['model']:
        config['as_beta'] = True

        # dimensions of data.
        # input is (x, y) values (2) while 
        # output of decoder should be 2 means and variances (4)
        x_dim = [2, 4] 

        # load lvae or vae
        if config['model'] == 'lvae':
            config['h_dim'] = [512, 256, 256]
            config['z_dim'] = [64, 32, 32]
            model = LadderVAE(config, x_dim).to(config['device'])
        else:
            config['h_dim'] = [512, 256, 128, 64]
            config['z_dim'] = 32
            model = VariationalAutoencoder(config, x_dim).to(config['device'])

        # perform training
        train_vae(train_loader, val_loader, model, config, args.mute)
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
        train_flow(train_loader, val_loader, model, config, args.mute)
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
        train_draw(train_loader, val_loader, model, config, args.mute)
