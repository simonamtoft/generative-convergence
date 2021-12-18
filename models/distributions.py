import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sum_except_batch, mean_except_batch


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


class Distribution(nn.Module):
    """Distribution base class."""

    def log_prob(self, x):
        """Calculate log probability under the distribution.
        Args:
            x: Tensor, shape (batch_size, ...)
        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, num_samples):
        """Generates samples from the distribution.
        Args:
            num_samples: int, number of samples to generate.
        Returns:
            samples: Tensor, shape (num_samples, ...)
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, num_samples):
        """Generates samples from the distribution together with their log probability.
        Args:
            num_samples: int, number of samples to generate.
        Returns:
            samples: Tensor, shape (num_samples, ...)
            log_prob: Tensor, shape (num_samples,)
        """
        samples = self.sample(num_samples)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def forward(self, *args, mode, **kwargs):
        '''
        To allow Distribution objects to be wrapped by DataParallelDistribution,
        which parallelizes .forward() of replicas on subsets of data.
        DataParallelDistribution.log_prob() calls DataParallel.forward().
        DataParallel.forward() calls Distribution.forward() for different
        data subsets on each device and returns the combined outputs.
        '''
        if mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        else:
            raise RuntimeError("Mode {} not supported.".format(mode))


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)


class ConditionalDistribution(Distribution):
    """ConditionalDistribution base class"""

    def log_prob(self, x, context):
        """Calculate log probability under the distribution.
        Args:
            x: Tensor, shape (batch_size, ...).
            context: Tensor, shape (batch_size, ...).
        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, context):
        """Generates samples from the distribution.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns:
            samples: Tensor, shape (batch_size, ...).
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, context):
        """Generates samples from the distribution together with their log probability.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns::
            samples: Tensor, shape (batch_size, ...).
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()


class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('one', torch.ones(1))

    def log_prob(self, x):
        lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype))
        ub = mean_except_batch(x.le(self.one).type(self.one.dtype))
        return torch.log(lb*ub)

    def sample(self, num_samples):
        return torch.rand((num_samples,) + self.shape, device=self.zero.device, dtype=self.zero.dtype)
