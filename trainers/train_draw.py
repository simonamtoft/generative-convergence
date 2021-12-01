import json
import wandb
import torch
from tqdm import tqdm
from torch.optim import Adam, Adamax
from torch.distributions.normal import Normal

from .config import PROJECT_NAME
from .train_utils import DeterministicWarmup


def get_normal(x_params: torch.Tensor) -> Normal:
    x_mu = x_params[:, 0:2]
    x_log_var = x_params[:, 2:]
    p = Normal(x_mu, x_log_var)
    return p


def train_draw(train_loader, val_loader, model, config):
    """ Train a DRAW model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=PROJECT_NAME, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])
    
    # linear deterministic warmup
    gamma = DeterministicWarmup(n=50, t_max=1)

    # train and validate
    print(f"\nTraining of DRAW model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
    for _ in tqdm(range(config['epochs']), desc='Training DRAW'):
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
        
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(loss_recon).mean(),
            'kl_train': torch.tensor(loss_kl).mean(),
            'elbo_train': torch.tensor(loss_elbo).mean()
        }, commit=False)


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
            
            # Log validation stuff
            wandb.log({
                'recon_val': torch.tensor(loss_recon).mean(),
                'kl_val': torch.tensor(loss_kl).mean(),
                'elbo_val': torch.tensor(loss_elbo).mean()
            }, commit=True)

    # Save final model
    torch.save(model, './saved_models/draw_model.pt')
    wandb.save('./saved_models/draw_model.pt')

    # Finalize logging
    wandb.finish()
    print('\nTraining finished!')