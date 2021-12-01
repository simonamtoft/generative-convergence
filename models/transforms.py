import torch
import torch.nn as nn

from collections.abc import Iterable

from .utils import sum_except_batch


class Transform(nn.Module):
    """Base class for Transform"""

    has_inverse = True

    @property
    def bijective(self):
        raise NotImplementedError()

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        raise NotImplementedError()

    @property
    def lower_bound(self):
        return self.stochastic_forward

    def forward(self, x):
        """
        Forward transform.
        Computes `z <- x` and the log-likelihood contribution term `log C`
        such that `log p(x) = log p(z) + log C`.
        Args:
            x: Tensor, shape (batch_size, ...)
        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z):
        """
        Inverse transform.
        Computes `x <- z`.
        Args:
            z: Tensor, shape (batch_size, ...)
        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()


class Bijection(Transform):
    """Base class for Bijection"""

    bijective = True
    stochastic_forward = False
    stochastic_inverse = False
    lower_bound = False


class CouplingBijection(Bijection):
    """Transforms each input variable with an invertible elementwise bijection.
    This input variables are split in two parts. The second part is transformed conditioned on the first part.
    The coupling network takes the first part as input and outputs trasnformations for the second part.
    Args:
        coupling_net: nn.Module, a coupling network such that for x = [x1,x2]
            elementwise_params = coupling_net(x1)
        split_dim: int, dimension to split the input (default=1).
        num_condition: int or None, number of parameters to condition on.
            If None, the first half is conditioned on:
            - For even inputs (1,2,3,4), (1,2) will be conditioned on.
            - For odd inputs (1,2,3,4,5), (1,2,3) will be conditioned on.
    """

    def __init__(self, coupling_net, split_dim=1, num_condition=None):
        super(CouplingBijection, self).__init__()
        assert split_dim >= 1
        self.coupling_net = coupling_net
        self.split_dim = split_dim
        self.num_condition = num_condition

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return torch.split(input, split_proportions, dim=self.split_dim)
        else:
            return torch.chunk(input, 2, dim=self.split_dim)

    def forward(self, x):
        id, x2 = self.split_input(x)
        elementwise_params = self.coupling_net(id)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = torch.cat([id, z2], dim=self.split_dim)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            id, z2 = self.split_input(z)
            elementwise_params = self.coupling_net(id)
            x2 = self._elementwise_inverse(z2, elementwise_params)
            x = torch.cat([id, x2], dim=self.split_dim)
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()


class AffineCouplingBijection(CouplingBijection):
    '''
    Affine coupling bijection.
    Args:
        coupling_net: nn.Module, a coupling network such that for x = [x1,x2]
            elementwise_params = coupling_net(x1)
        split_dim: int, dimension to split the input (default=1).
        num_condition: int or None, number of parameters to condition on.
            If None, the first half is conditioned on:
            - For even inputs (1,2,3,4), (1,2) will be conditioned on.
            - For odd inputs (1,2,3,4,5), (1,2,3) will be conditioned on.
        scale_fn: callable, the transform to obtain the scale.
    '''

    def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s: torch.exp(s)):
        super(AffineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        assert callable(scale_fn)
        self.scale_fn = scale_fn

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        z = scale * x + shift
        ldj = sum_except_batch(torch.log(scale))
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        x = (z - shift) / scale
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift


class _ActNormBijection(Bijection):
    '''
    Base class for activation normalization [1].
    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def __init__(self, num_features, data_dep_init=True, eps=1e-6):
        super(_ActNormBijection, self).__init__()
        self.num_features = num_features
        self.data_dep_init = data_dep_init
        self.eps = eps

        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.register_params()

    def data_init(self, x):
        self.initialized += 1.
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def forward(self, x):
        if self.training and not self.initialized: self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)
        ldj = torch.sum(-self.log_scale).expand([x.shape[0]]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z):
        return self.shift + z * torch.exp(self.log_scale)

    def register_params(self):
        '''Register parameters shift and log_scale'''
        raise NotImplementedError()

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        raise NotImplementedError()

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise NotImplementedError()


class ActNormBijection(_ActNormBijection):
    '''
    Activation normalization [1] for inputs on the form (B,D).
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def register_params(self):
        '''Register parameters shift and log_scale'''
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.num_features)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features)))

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return 1


class Permute(Bijection):
    """
    Permutes inputs on a given dimension using a given permutation.
    Args:
        permutation: Tensor or Iterable, shape (dim_size)
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, permutation, dim=1):
        super(Permute, self).__init__()
        assert isinstance(dim, int), 'dim must be an integer'
        assert dim >= 1, 'dim must be >= 1 (0 corresponds to batch dimension)'
        assert isinstance(permutation, torch.Tensor) or isinstance(permutation, Iterable), 'permutation must be a torch.Tensor or Iterable'
        if isinstance(permutation, torch.Tensor):
            assert permutation.ndimension() == 1, 'permutation must be a 1D tensor, but was of shape {}'.format(permutation.shape)
        else:
            permutation = torch.tensor(permutation)

        self.dim = dim
        self.register_buffer('permutation', permutation)

    @property
    def inverse_permutation(self):
        return torch.argsort(self.permutation)

    def forward(self, x):
        return torch.index_select(x, self.dim, self.permutation), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, z):
        return torch.index_select(z, self.dim, self.inverse_permutation)


class Reverse(Permute):
    """
    Reverses inputs on a given dimension.
    Args:
        dim_size: int, number of elements on dimension dim
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, dim_size, dim=1):
        super(Reverse, self).__init__(torch.arange(dim_size - 1, -1, -1), dim)