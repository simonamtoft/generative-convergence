import wandb
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal

from lib import DeterministicWarmup, log_images_toy, \
    lambda_lr


def get_normal(x_params: torch.Tensor) -> Normal:
    x_mu = x_params[:, 0:2]
    x_log_var = x_params[:, 2:]
    p = Normal(x_mu, x_log_var)
    return p


def train_vae(train_loader: DataLoader, val_loader: DataLoader, model, config: dict, mute: bool, wandb_name: str) -> tuple:
    """ Train a Standard VAE model and log training information to wandb.
        Also perform an evaluation on a validation set.
        Trains on toy data using a Normal distribution as the likelihood function
    """
    # Initialize a new wandb run
    wandb.init(project=wandb_name, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    elif config['optimizer']  == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    
    # linear deterministic warmup
    gamma = DeterministicWarmup(n=config['kl_warmup'], t_max=1)

    # Set learning rate scheduler
    # if "lr_decay" in config:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
    )

    # train and validate
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(config['epochs']), desc=f"Training {config['model']}", disable=mute):
        # Training epoch
        model.train()
        elbo_train = []
        kld_train = []
        recon_train = []
        alpha = next(gamma)
        for x in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(config['device'])
            x_params, kld = model(x)

            # define likelihood distribution
            p = get_normal(x_params)

            # Compute losses
            recon = -torch.mean(p.log_prob(x).sum(1))
            kl = torch.mean(kld)
            loss = recon + alpha * kl
            elbo = -(recon + kl)

            # filter nan losses
            if not torch.isnan(loss):
                # Update gradients
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # perform clipping of model parameters
                clip_grad_norm_(model.parameters(), max_norm=1)

            # save losses
            elbo_train.append(torch.mean(elbo).item())
            kld_train.append(torch.mean(kl).item())
            recon_train.append(torch.mean(recon).item())
        
        # get sample for reconstruction
        x_recon = p.sample((1,))[0]

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
        with torch.no_grad():
            model.eval()
            elbo_val = []
            kld_val = []
            recon_val = []
            for x in iter(val_loader):
                batch_size = x.size(0)

                # Pass batch through model
                x = x.view(batch_size, -1)
                x = Variable(x).to(config['device'])
                x_params, kld = model(x)

                # define likelihood distribution
                p = get_normal(x_params)

                # Compute losses
                recon = -torch.mean(p.log_prob(x).sum(1))
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
                'recon_train': recon_val,
                'kl_train': kld_val,
                'loss_train': elbo_val
            }, commit=False)

            # Sample from model
            if isinstance(config['z_dim'], list):
                z_mu = Variable(torch.randn(config['batch_size'], config['z_dim'][-1])).to(config['device'])
            else:
                z_mu = Variable(torch.randn(config['batch_size'], config['z_dim'])).to(config['device'])
            x_params = model.sample(z_mu)
            p = get_normal(x_params)
            x_sample = p.sample((1,))[0]
            
            # log sample and reconstruction
            log_images_toy(x_recon, x_sample, str(epoch+1) + config['model'])

    # Finalize training
    torch.save(model, f"./saved_models/{config['model']}_model.pt")
    wandb.save(f"./saved_models/{config['model']}_model.pt")
    wandb.finish()
    return train_losses, val_losses