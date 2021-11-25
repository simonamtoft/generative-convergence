import json
import wandb
import torch
from tqdm import tqdm
from torch.optim import Adam, Adamax

from .config import PROJECT_NAME


def train_flow(train_loader, val_loader, model, config):
    """ Train a FLOW model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=PROJECT_NAME, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])

    print(f"\nTraining of flow model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
    for _ in tqdm(range(config['epochs']), desc='Training FLOW'):
        # Training Epoch
        loss_sum = 0.0
        for _, x in enumerate(train_loader):
            # pass through model and get loss
            loss = -model.log_prob(x.to(config['device'])).mean()

            # update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update losses
            loss_sum += loss.detach().cpu().item()

        wandb.log({
            'loss': loss_sum / len(train_loader)
        }, commit=True)

    # Save final model
    torch.save(model, './saved_models/flow_model.pt')
    wandb.save('./saved_models/flow_model.pt')

    # Finalize logging
    wandb.finish()
    print('\nTraining finished!')