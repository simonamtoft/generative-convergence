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
has_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if has_cuda else 'cpu')
device = 'cpu'

# Define config
config = {
    'dataset': '8gaussians',
    'train_samples': 1024,
    'val_samples': 1024,
    'batch_size': 64,
    'epochs': 10,
    'lr': 1e-3,
    'optimizer': 'adam',
    'model': MODEL_NAME,
    'device': device,
}

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
if MODEL_NAME == 'vae':
    # define some model specific config
    config['h_dim'] = [512, 256, 128, 64]
    config['z_dim'] = 64
    config['as_beta'] = True

    # instantiate model
    model = VariationalAutoencoder(config, 2).to(device)

    # perform training
    model_trainers.train_vae(train_loader, val_loader, model, config)
elif MODEL_NAME == 'flow':
    model = None
    model_trainers.train_flow(train_loader, val_loader, model, config)
elif MODEL_NAME == 'draw':
    # define some model specific config
    config['attention'] = 'base'
    config['h_dim'] = 256
    config['z_dim'] = 32
    config['T'] = 10
    config['N'] = 12

    # instantiate model
    model = DRAW(config, [2, 1]).to(device)

    # perform training
    model_trainers.train_draw(train_loader, val_loader, model, config)

