import json
import wandb
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import Adam, Adamax

from .config import PROJECT_NAME
from .train_utils import DeterministicWarmup


def train_vae(train_loader, val_loader, model, config):
    """ Train a Standard VAE model and log training information to wandb.
        Also perform an evaluation on a validation set."""
    # Initialize a new wandb run
    wandb.init(project=PROJECT_NAME, config=config)
    wandb.watch(model)

    # specify optimizer
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer']  == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config['lr'])
    
    # linear deterministic warmup
    gamma = DeterministicWarmup(n=50, t_max=1)
    
    # train and validate
    print(f"\nTraining of VAE model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
    for _ in tqdm(range(config['epochs']), desc='Training VAE'):
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
            x_hat, kld = model(x)

            # Compute losses
            # recon = torch.mean(loss_func(x_hat, x))
            recon = torch.mean(dist.log_prob(x_hat))
            kl = torch.mean(kld)
            loss = recon + alpha * kl

            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            elbo_train.append(torch.mean(-loss).item())
            kld_train.append(torch.mean(kl).item())
            recon_train.append(torch.mean(recon).item())
    
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(recon_train).mean(),
            'kl_train': torch.tensor(kld_train).mean(),
            'elbo_train': torch.tensor(elbo_train).mean()
        }, commit=False)

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
                print(x.shape)
                x = Variable(x).to(config['device'])
                x_hat, kld = model(x)

                # Compute losses
                # recon = torch.mean(loss_func(x_hat, x))
                recon = torch.mean(dist.log_prob(x_hat))
                kl = torch.mean(kld)
                loss = recon + alpha * kl

                # save losses
                elbo_val.append(torch.mean(-loss).item())
                kld_val.append(torch.mean(kld).item())
                recon_val.append(torch.mean(recon).item())
        
            # Log validation stuff
            wandb.log({
                'recon_val': torch.tensor(recon_val).mean(),
                'kl_val': torch.tensor(kld_val).mean(),
                'elbo_val': torch.tensor(elbo_val).mean()
            }, commit=True)

    # Save final model
    torch.save(model, './saved_models/vae_model.pt')
    wandb.save('./saved_models/vae_model.pt')

    # Finalize logging
    wandb.finish()
    print('\nTraining finished!')