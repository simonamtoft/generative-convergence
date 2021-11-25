import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib import get_ffjord_data

from models import DRAW, VariationalAutoencoder
from trainers import train_vae, train_draw, train_flow


# Define config
CONFIG = {
    'dataset': '8gaussians',
    'train_samples': 1024,
    'val_samples': 1024,
    'batch_size': 64,
    'epochs': 10,
    'lr': 1e-3,
    'optimizer': 'adam',
}


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description="Model training script.")
    parser.add_argument('-m', '--model', help='Pick which model to train (default: vae).', default='vae', type=str, choices=['vae', 'draw', 'flow'])
    args = parser.parse_args()

    # set device
    # has_cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if has_cuda else 'cpu')

    # start config
    config = CONFIG
    config['model'] = args.model
    config['device'] = 'cpu'

    # get data
    train_data = get_ffjord_data(config['dataset'], config['train_samples'])
    val_data = get_ffjord_data(config['dataset'], config['val_samples'])

    # Setup data loader
    # kwargs = {'num_workers': 4, 'pin_memory': True} if has_cuda else {}
    kwargs = {}
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
    if config['model'] == 'vae':
        # define some model specific config
        config['h_dim'] = [512, 256, 128, 64]
        config['z_dim'] = 64
        config['as_beta'] = True

        # instantiate model
        model = VariationalAutoencoder(config, 2).to(config['device'])

        # perform training
        train_vae(train_loader, val_loader, model, config)
    elif config['model']  == 'flow':
        # instantiate model
        model = None

        # perform training
        train_flow(train_loader, val_loader, model, config)
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
        train_draw(train_loader, val_loader, model, config)

