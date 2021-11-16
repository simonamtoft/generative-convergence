import json
import wandb
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam, Adamax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_flow(train_loader, model, config):
    # Initialize a new wandb run
    wandb.init(project='generative-convergence', config=config)
    wandb.watch(model)

    # specify optimizer
    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config.lr)

    # model training
    train_start_print(config, 'flow')
    for _ in range(config.epochs):
        loss_sum = 0.0
        for _, x in enumerate(train_loader):
            # pass through model and get loss
            loss = -model.log_prob(x.to(device)).mean()

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
    torch.save('flow_model.pt')
    wandb.save('flow_model.pt')

    # Finalize training
    wandb.finish()



def train_vae(train_loader, model, config):
    # Initialize a new wandb run
    wandb.init(project='generative-convergence', config=config)
    wandb.watch(model)

    # specify optimizer
    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config.lr)
    
    # model training
    train_start_print(config, 'vae')
    model.train()
    for _ in range(config.epochs):
        elbo_train = []
        kld_train = []
        recon_train = []
        for x, _ in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(device)
            x_hat, kld = model(x)

            # Compute losses
            recon = torch.mean(bce_loss(x_hat, x))
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
        }, commit=True)

    # Save final model
    torch.save('vae_model.pt')
    wandb.save('vae_model.pt')

    # Finalize training
    wandb.finish()


def train_draw(train_loader, model, config):
    # Initialize a new wandb run
    wandb.init(project='generative-convergence', config=config)
    wandb.watch(model)

    # specify optimizer
    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamax':
        optimizer = Adamax(model.parameters(), lr=config.lr)
    
    # model training
    train_start_print(config, 'draw')
    model.train()
    for _ in range(config.epochs):
        loss_recon = []
        loss_kl = []
        loss_elbo = []
        for x, i in tqdm(train_loader, disable=True):
            batch_size = x.size(0)

            # Pass through model
            x = x.view(batch_size, -1).to(device)
            x_hat, kld = model(x)
            x_hat = torch.sigmoid(x_hat)

            # compute losses
            reconstruction = torch.mean(bce_loss(x_hat, x).sum(1))
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

    # Save final model
    torch.save('draw_model.pt')
    wandb.save('draw_model.pt')

    # Finalize training
    wandb.finish()


def train_start_print(config, model_name):
    print(f"\nTraining of {model_name} model will run on device: {device}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))
