import json
import wandb
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam, Adamax

# define loss function for VAEs (recon)
loss_func = torch.nn.CrossEntropyLoss()

# define string of project to log to on wandb
PROJECT_NAME = 'generative-convergence'


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

    train_start_print(config, 'flow')
    for _ in range(config['epochs']):
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
    
    train_start_print(config, 'vae')
    for _ in range(config['epochs']):
        # Training epoch
        model.train()
        elbo_train = []
        kld_train = []
        recon_train = []
        for x in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(config['device'])
            x_hat, kld = model(x)

            # Compute losses
            recon = torch.mean(loss_func(x_hat, x))
            kl = torch.mean(kld)
            # loss = recon + alpha * kl
            loss = recon + kl

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
                x = Variable(x).to(config['device'])
                x_hat, kld = model(x)

                # Compute losses
                recon = torch.mean(loss_func(x_hat, x))
                kl = torch.mean(kld)
                # loss = recon + alpha * kl
                loss = recon + kl

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
    
    train_start_print(config, 'draw')
    for _ in range(config['epochs']):
        # Training Epoch
        model.train()
        loss_recon = []
        loss_kl = []
        loss_elbo = []
        for x in iter(train_loader):
            batch_size = x.size(0)

            # Pass through model
            x = x.view(batch_size, -1).to(config['device'])
            x_hat, kld = model(x)
            x_hat = torch.sigmoid(x_hat)

            # compute losses
            reconstruction = torch.mean(loss_func(x_hat, x).sum(1))
            kl = torch.mean(kld.sum(1))
            # loss = reconstruction + alpha * kl
            loss = reconstruction + kl

            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            loss_recon.append(reconstruction.item())
            loss_kl.append(kl.item())
            loss_elbo.append(-loss.item())
        
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(loss_recon).mean(),
            'kl_train': torch.tensor(loss_kl).mean(),
            'elbo_train': torch.tensor(loss_elbo).mean()
        }, commit=True)


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
                x_hat, kld = model(x)
                x_hat = torch.sigmoid(x_hat)
                
                # Compute losses
                reconstruction = torch.mean(loss_func(x_hat, x).sum(1))
                kl = torch.mean(kld.sum(1))
                # loss = reconstruction + alpha * kl 
                loss = reconstruction + kl

                # save losses
                loss_recon.append(reconstruction.item())
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


def train_start_print(config, model_name):
    print(f"\nTraining of {model_name} model will run on device: {config['device']}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
