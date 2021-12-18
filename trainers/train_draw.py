import json
import wandb
import torch
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

    # train and validate
    train_losses = {'recon': [], 'kl': [], 'elbo': []}
    val_losses = {'recon': [], 'kl': [], 'elbo': []}
    for _ in tqdm(range(config['epochs']), desc='Training DRAW', disable=mute):
        # Training Epoch
        model.train()
        loss_recon = []
        loss_kl = []
        loss_elbo = []
        alpha = next(gamma)
        for x in iter(train_loader):
            batch_size = x.size(0)

            # Pass through model
            x = x.view(batch_size, -1).to(config['device'])
            x_params, kld = model(x)
            # x_params = torch.sigmoid(x_params)

            # define likelihood distribution
            p = get_normal(x_params)

            # Compute losses
            recon = -torch.mean(p.log_prob(x).sum(1))
            kl = torch.mean(kld)
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
        recon_train = np.array(recon_train).mean()
        kld_train = np.array(kld_train).mean()
        elbo_train = np.array(elbo_train).mean()

        # Log train stuff
        train_losses['recon'].append(recon_train)
        train_losses['kl'].append(kld_train)
        train_losses['elbo'].append(elbo_train)
        wandb.log({
            'recon_train': recon_train,
            'kl_train': kld_train,
            'elbo_train': elbo_train
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
            for x in iter(val_loader):
                batch_size = x.size(0)

                # Pass through model
                x = x.view(batch_size, -1).to(config['device'])
                x_params, kld = model(x)
                # x_params = torch.sigmoid(x_params)

                # define likelihood distribution
                p = get_normal(x_params)

                # Compute losses
                recon = -torch.mean(p.log_prob(x).sum(1))
                kl = torch.mean(kld)
                loss = recon + alpha * kl

                # save losses
                loss_recon.append(recon.item())
                loss_kl.append(kl.item())
                loss_elbo.append(-loss.item())
            
            # get mean losses
            recon_val = np.array(recon_val).mean()
            kld_val = np.array(kld_val).mean()
            elbo_val = np.array(elbo_val).mean()

            # Log validation losses
            val_losses['recon'].append(recon_val)
            val_losses['kl'].append(kld_val)
            val_losses['elbo'].append(elbo_val)
            wandb.log({
                'recon_train': recon_val,
                'kl_train': kld_val,
                'elbo_train': elbo_val
            }, commit=False)

    # Finalize training
    torch.save(model, './saved_models/draw_model.pt')
    wandb.save('./saved_models/draw_model.pt')
    wandb.finish()
    return train_losses, val_losses