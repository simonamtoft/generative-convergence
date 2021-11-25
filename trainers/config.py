import torch

# define string of project to log to on wandb
PROJECT_NAME = 'generative-convergence'

# define loss function for VAEs (recon)
loss_func = torch.nn.CrossEntropyLoss()

