import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.gen_data import get_ffjord_data
import model_trainers

from draw.draw import DRAW
from vae.vae import VariationalAutoencoder

# pick model to train
# - vae, draw, flow
MODEL_NAME = 'vae' 

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define config
config = {
    'dataset': '8gaussians',
    'train_samples': 1024,
    'val_samples': 1024,
    'batch_size': 64,
    'epochs': 10,
    'lr': 1e-3,
    'model': MODEL_NAME,
}

# get data
train_data = get_ffjord_data(config['dataset'], config['train_samples'])
val_data = get_ffjord_data(config['dataset'], config['val_samples'])

# Setup data loader
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = DataLoader(
    train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    **kwargs
)
val_loader = DataLoader(
    val_data,
    batch_size=config['batch_size'],
    shuffle=True,
    **kwargs
)


# define and train model
if MODEL_NAME == 'vae':
    # define some model specific config
    config['h_dim'] = [512, 256, 128, 64]
    config['z_dim'] = 64
    config['as_beta'] = True

    # instantiate model
    x_shape = None #???????????
    model = VariationalAutoencoder(config, x_shape).to(device)

    # perform training
    model_trainers.train_vae(train_loader, model, config)
elif MODEL_NAME == 'flow':
    model = None
    model_trainers.train_flow(train_loader, model, config)
elif MODEL_NAME == 'draw':
    # define some model specific config
    config['attention'] = 'base'
    config['h_dim'] = 256
    config['z_dim'] = 32
    config['T'] = 10
    config['N'] = 12

    # instantiate model
    x_shape = None #???????????
    model = DRAW(config, x_shape).to(device)

    # perform training
    model_trainers.train_draw(train_loader, model, config)

