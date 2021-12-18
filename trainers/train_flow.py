import os
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader

from lib import lambda_lr


def train_flow(train_loader: DataLoader, val_loader: DataLoader, model, config: dict, mute: bool, wandb_name: str):
    """ Train a Flow model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=wandb_name, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])
    
    # Set learning rate scheduler
    # if "lr_decay" in config:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
    )

    # train and validate
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(config['epochs']), desc='Training Flow', disable=mute):
        # Training Epoch
        model.train()
        losses = []
        for x, _ in iter(train_loader):
            # pass through model and get loss
            x = Variable(x).to(config['device'])
            loss = -model.log_prob(x).mean()

            # update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update losses
            losses.append(loss.item())
        
        # log training
        loss = np.array(losses).mean()
        train_losses.append(loss)
        wandb.log({
            'loss_train': loss
        }, commit=False)

        # Update scheduler
        # if "lr_decay" in config:
        scheduler.step()

        # Evaluate on validation set
        with torch.no_grad():
            model.eval()
            losses = []
            for x in iter(val_loader):
                # pass through model and get loss
                x = Variable(x).to(config['device'])
                loss = -model.log_prob(x).mean()

                # update losses
                losses.append(loss.item())
            
            # Log validation stuff
            loss = np.array(losses).mean()
            val_losses.append(loss)
            wandb.log({
                'loss_val': loss,
            }, commit=False)

        # Sampling
        samples = model.sample(config['batch_size']).cpu().numpy()

        # create and log plot
        name = f'./log_images/flow_sampling_{epoch+1}.png'
        plt.figure()
        plt.plot(samples[:, 0], samples[:, 1], '.')
        plt.title('Samples')
        plt.savefig(name, transparent=True, bbox_inches='tight')
        plt.close()
        wandb.log({
            "sampling": wandb.Image(name)
        }, commit=True)
        os.remove(name)
        
    # Finalize training
    wandb.finish()
    return train_losses, val_losses