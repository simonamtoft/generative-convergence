import json
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, Adamax
from torch.distributions.normal import Normal

from lib import DeterministicWarmup, log_images, \
    lambda_lr


def get_normal(x_params: torch.Tensor) -> Normal:
    x_mu = x_params[:, 0:2]
    x_log_var = x_params[:, 2:]
    p = Normal(x_mu, x_log_var)
    return p


def train_draw(train_loader, val_loader, model, config, mute, wandb_name):
    """ Train a DRAW model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=wandb_name, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])
    
    # linear deterministic warmup
    gamma = DeterministicWarmup(n=config['kl_warmup'], t_max=1)

    # Set learning rate scheduler
    # if "lr_decay" in config:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
    )

    # Define loss function
    bce_loss = nn.BCELoss(reduction='none').to(config['device'])

    # train and validate
    train_losses = {'recon': [], 'kl': [], 'elbo': []}
    val_losses = {'recon': [], 'kl': [], 'elbo': []}
    for epoch in tqdm(range(config['epochs']), desc='Training DRAW', disable=mute):
        # Training Epoch
        model.train()
        loss_recon = []
        loss_kl = []
        loss_elbo = []
        alpha = next(gamma)
        for x, _ in iter(train_loader):
            batch_size = x.size(0)

            # Pass through model
            x = x.view(batch_size, -1).to(config['device'])
            x_hat, kld = model(x)
            x_hat = torch.sigmoid(x_hat)

            # Compute losses
            recon = torch.mean(bce_loss(x_hat, x).sum(1))
            kl = torch.mean(kld.sum(1))
            loss = recon + alpha * kl

            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            loss_recon.append(recon.item())
            loss_kl.append(kl.item())
            loss_elbo.append(-loss.item())
        
        # get mean losses
        loss_recon = np.array(loss_recon).mean()
        loss_kl = np.array(loss_kl).mean()
        loss_elbo = np.array(loss_elbo).mean()

        # Log train stuff
        train_losses['recon'].append(loss_recon)
        train_losses['kl'].append(loss_kl)
        train_losses['elbo'].append(loss_elbo)
        wandb.log({
            'recon_train': loss_recon,
            'kl_train': loss_kl,
            'elbo_train': loss_elbo
        }, commit=False)

        # Update scheduler
        # if "lr_decay" in config:
        scheduler.step()

        # Evaluate on validation set
        with torch.no_grad():
            model.eval()
            loss_recon = []
            loss_kl = []
            loss_elbo = []
            for x, _ in iter(val_loader):
                batch_size = x.size(0)

                # Pass through model
                x = x.view(batch_size, -1).to(config['device'])
                x_hat, kld = model(x)
                x_hat = torch.sigmoid(x_hat)

                # Compute losses
                recon = torch.mean(bce_loss(x_hat, x).sum(1))
                kl = torch.mean(kld.sum(1))
                loss = recon + alpha * kl

                # save losses
                loss_recon.append(recon.item())
                loss_kl.append(kl.item())
                loss_elbo.append(-loss.item())
            
            # get mean losses
            loss_recon = np.array(loss_recon).mean()
            loss_kl = np.array(loss_kl).mean()
            loss_elbo = np.array(loss_elbo).mean()

            # Log validation losses
            val_losses['recon'].append(loss_recon)
            val_losses['kl'].append(loss_kl)
            val_losses['elbo'].append(loss_elbo)
            wandb.log({
                'recon_val': loss_recon,
                'kl_val': loss_kl,
                'elbo_val': loss_elbo
            }, commit=False)

            # Sample from model
            x_sample = model.sample()

            # Log images to wandb
            log_images(x_hat, x_sample, epoch)

    # Finalize training
    torch.save(model, './saved_models/draw_model.pt')
    wandb.save('./saved_models/draw_model.pt')
    wandb.finish()
    return train_losses, val_losses