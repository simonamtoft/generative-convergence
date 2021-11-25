import math
import torch
import torch.nn.functional as F

def log_gaussian(x, mu, log_var):
    """
        Returns the log pdf of a normal distribution parametrised
        by mu and log_var evaluated at x.
    
    Inputs:
        x       : the point to evaluate
        mu      : the mean of the distribution
        log_var : the log variance of the distribution
    Returns:
        log(N(x | mu, sigma))
    """
    log_pdf = (
        - 0.5 * math.log(2 * math.pi) 
        - log_var / 2 
        - (x - mu)**2 / (2 * torch.exp(log_var))
    )
    return torch.sum(log_pdf, dim=-1)


def log_standard_gaussian(x):
    """
        Returns the log pdf of a standard normal distribution N(0, 1)
    
    Inputs:
        x   : the point to evaluate
    Returns:
        log(N(x | 0, I))
    """
    log_pdf = (
        -0.5 * math.log(2 * math.pi) 
        - x ** 2 / 2
    )

    return torch.sum(log_pdf, dim=-1)
