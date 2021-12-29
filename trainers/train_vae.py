import wandb
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from lib import DeterministicWarmup, log_images, \
    lambda_lr, bce_loss

SAVE_NAME = 'vae_model.pt'


def train_vae(train_loader: DataLoader, val_loader: DataLoader, model, config: dict, mute: bool, wandb_name: str):
    # Initialize a new wandb run
    wandb.init(project=wandb_name, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    elif config['optimizer']  == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))

    # Set learning rate scheduler
    # if "lr_decay" in config:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
    )
    
    # linear deterministic warmup
    gamma = DeterministicWarmup(n=config['kl_warmup'], t_max=1)

    # Run training and validation
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(config['epochs']), desc=f"Training {config['model']}", disable=mute):
        # Train Epoch
        model.train()
        alpha = next(gamma)
        elbo_train = []
        kld_train = []
        recon_train = []
        for x, _ in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(config['device'])
            x_hat, kld = model(x)

            # Compute losses
            recon = torch.mean(bce_loss(x_hat, x))
            kl = torch.mean(kld)
            loss = recon + alpha * kl
            elbo = -(recon + kl)

            # filter nan losses
            if not torch.isnan(loss):
                # Update gradients
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # save losses
            elbo_train.append(torch.mean(elbo).item())
            kld_train.append(torch.mean(kl).item())
            recon_train.append(torch.mean(recon).item())

        # get mean losses
        recon_train = np.array(recon_train).mean()
        kld_train = np.array(kld_train).mean()
        elbo_train = np.array(elbo_train).mean()
        
        # Log train losses
        train_losses.append(elbo_train)
        wandb.log({
            'recon_train': recon_train,
            'kl_train': kld_train,
            'loss_train': elbo_train
        }, commit=False)

        # Update scheduler
        # if "lr_decay" in config:
        scheduler.step()

        # Validation epoch
        model.eval()
        with torch.no_grad():
            elbo_val = []
            kld_val = []
            recon_val = []
            for x, _ in iter(val_loader):
                batch_size = x.size(0)

                # Pass batch through model
                x = x.view(batch_size, -1)
                x = Variable(x).to(config['device'])
                x_hat, kld = model(x)

                # Compute losses
                recon = torch.mean(bce_loss(x_hat, x))
                kl = torch.mean(kld)
                loss = recon + alpha * kl
                elbo = -(recon + kl)

                # save losses
                elbo_val.append(torch.mean(elbo).item())
                kld_val.append(torch.mean(kld).item())
                recon_val.append(torch.mean(recon).item())
        
        # get mean losses
        recon_val = np.array(recon_val).mean()
        kld_val = np.array(kld_val).mean()
        elbo_val = np.array(elbo_val).mean()

        # Log validation losses
        val_losses.append(elbo_val)
        wandb.log({
            'recon_val': recon_val,
            'kl_val': kld_val,
            'loss_val': elbo_val
        }, commit=False)

        # Sample from model
        if isinstance(config['z_dim'], list):
            x_mu = Variable(torch.randn(16, config['z_dim'][0])).to(config['device'])
        else:
            x_mu = Variable(torch.randn(16, config['z_dim'])).to(config['device'])
        x_sample = model.sample(x_mu)

        # Log images to wandb
        log_images(x_hat, x_sample, str(epoch) + config['model'])
    
    # Finalize training
    torch.save(model, f"./saved_models/{config['model']}_model.pt")
    wandb.save(f"./saved_models/{config['model']}_model.pt")
    wandb.finish()
    return train_losses, val_losses