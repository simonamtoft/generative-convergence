import os
import json
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, Adamax

from .config import PROJECT_NAME
from .train_utils import DeterministicWarmup, log_images, \
    lambda_lr


def train_flow(train_loader, val_loader, model, config, mute):
    """ Train a Flow model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=PROJECT_NAME, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])
    
    # Set learning rate scheduler
    if "lr_decay" in config:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
        )

    print(f"\nTraining of flow model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
    for epoch in tqdm(range(config['epochs']), desc='Training Flow', disable=mute):
        # Training Epoch
        model.train()
        losses = []
        for x in iter(train_loader):
            # pass through model and get loss
            x = x.to(config['device'])
            loss = -model.log_prob(x).mean()

            # update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update losses
            losses.append(loss.item())

        # log training
        wandb.log({
            'loss_train': torch.tensor(losses).mean()
        }, commit=False)

        # Update scheduler
        if "lr_decay" in config:
            scheduler.step()

        # Evaluate on validation set
        with torch.no_grad():
            model.eval()
            losses = []
            for x in iter(val_loader):
                # pass through model and get loss
                x = x.to(config['device'])
                loss = -model.log_prob(x).mean()

                # update losses
                losses.append(loss.item())

            # Log validation stuff
            wandb.log({
                'loss_val': torch.tensor(losses).mean(),
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

    # Save final model 
    # <<<< THIS HAS ISSUES WITH A TRANSFORM DEFINED >>>>
    # torch.save(model, './saved_models/flow_model.pt')
    # wandb.save('./saved_models/flow_model.pt')

    # Finalize logging
    wandb.finish()
    print('\nTraining finished!')