import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul
import numpy as np

from collections.abc import Iterable

from .utils import sum_except_batch
from .distributions import ConditionalDistribution


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
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
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
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
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


class Squeeze2d(Bijection):
    """
    A bijection defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.
    Introduced in the RealNVP paper [1].
    Args:
        factor: int, the factor to squeeze by (default=2).
        ordered: bool, if True, squeezing happens imagewise.
                       if False, squeezing happens channelwise.
                       For more details, see example (default=False).
    Source implementation:
        Based on `squeeze_nxn`, `squeeze_2x2`, `squeeze_2x2_ordered`, `unsqueeze_2x2` in:
        https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_utils.py
    Example:
        Input x of shape (1, 2, 4, 4):
        [[[[ 1  2  1  2]
           [ 3  4  3  4]
           [ 1  2  1  2]
           [ 3  4  3  4]]
          [[10 20 10 20]
           [30 40 30 40]
           [10 20 10 20]
           [30 40 30 40]]]]
        Standard output z of shape (1, 8, 2, 2):
        [[[[ 1  1]
           [ 1  1]]
          [[ 2  2]
           [ 2  2]]
          [[ 3  3]
           [ 3  3]]
          [[ 4  4]
           [ 4  4]]
          [[10 10]
           [10 10]]
          [[20 20]
           [20 20]]
          [[30 30]
           [30 30]]
          [[40 40]
           [40 40]]]]
        Ordered output z of shape (1, 8, 2, 2):
        [[[[ 1  1]
           [ 1  1]]
          [[10 10]
           [10 10]]
          [[ 4  4]
           [ 4  4]]
          [[40 40]
           [40 40]]
          [[ 2  2]
           [ 2  2]]
          [[20 20]
           [20 20]]
          [[ 3  3]
           [ 3  3]]
          [[30 30]
           [30 30]]]]
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(self, factor=2, ordered=False):
        super(Squeeze2d, self).__init__()
        assert isinstance(factor, int)
        assert factor > 1
        self.factor = factor
        self.ordered = ordered

    def _squeeze(self, x):
        assert len(x.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
        batch_size, c, h, w = x.shape
        assert h % self.factor == 0, 'h = {} not multiplicative of {}'.format(h, self.factor)
        assert w % self.factor == 0, 'w = {} not multiplicative of {}'.format(w, self.factor)
        t = x.view(batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor)
        if not self.ordered:
            t = t.permute(0, 1, 3, 5, 2, 4).contiguous()
        else:
            t = t.permute(0, 3, 5, 1, 2, 4).contiguous()
        z = t.view(batch_size, c * self.factor ** 2, h // self.factor, w // self.factor)
        return z

    def _unsqueeze(self, z):
        assert len(z.shape) == 4, 'Dimension should be 4, but was {}'.format(len(z.shape))
        batch_size, c, h, w = z.shape
        assert c % (self.factor ** 2) == 0, 'c = {} not multiplicative of {}'.format(c, self.factor ** 2)
        if not self.ordered:
            t = z.view(batch_size, c // self.factor ** 2, self.factor, self.factor, h, w)
            t = t.permute(0, 1, 4, 2, 5, 3).contiguous()
        else:
            t = z.view(batch_size, self.factor, self.factor, c // self.factor ** 2, h, w)
            t = t.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = t.view(batch_size, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x):
        z = self._squeeze(x)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._unsqueeze(z)
        return x


class Surjection(Transform):
    """Base class for Surjection"""

    bijective = False

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        return not self.stochastic_forward


class UniformDequantization(Surjection):
    '''
    A uniform dequantization layer.
    This is useful for converting discrete variables to continuous [1, 2].
    Forward:
        `z = (x+u)/K, u~Unif(0,1)^D`
        where `x` is discrete, `x \in {0,1,2,...,K-1}^D`.
    Inverse:
        `x = Quantize(z, K)`
    Args:
        num_bits: int, number of bits in quantization,
            i.e. 8 for `x \in {0,1,2,...,255}^D`
            or 5 for `x \in {0,1,2,...,31}^D`.
    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    '''

    stochastic_forward = True

    def __init__(self, num_bits=8):
        super(UniformDequantization, self).__init__()
        self.num_bits = num_bits
        self.quantization_bins = 2**num_bits
        self.register_buffer('ldj_per_dim', -torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)))

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def forward(self, x):
        u = torch.rand(x.shape, device=self.ldj_per_dim.device, dtype=self.ldj_per_dim.dtype)
        z = (x.type(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape)
        return z, ldj

    def inverse(self, z):
        z = self.quantization_bins * z
        return z.floor().clamp(min=0, max=self.quantization_bins-1).long()


class Augment(Surjection):
    '''
    A simple augmentation layer which augments the input with additional elements.
    This is useful for constructing augmented normalizing flows [1, 2].
    References:
        [1] Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models,
            Huang et al., 2020, https://arxiv.org/abs/2002.07101
        [2] VFlow: More Expressive Generative Flows with Variational Data Augmentation,
            Chen et al., 2020, https://arxiv.org/abs/2002.09741
    '''
    stochastic_forward = True

    def __init__(self, encoder, x_size, split_dim=1):
        super(Augment, self).__init__()
        assert split_dim >= 1
        self.encoder = encoder
        self.split_dim = split_dim
        self.x_size = x_size
        self.cond = isinstance(self.encoder, ConditionalDistribution)

    def split_z(self, z):
        split_proportions = (self.x_size, z.shape[self.split_dim] - self.x_size)
        return torch.split(z, split_proportions, dim=self.split_dim)

    def forward(self, x):
        if self.cond: z2, logqz2 = self.encoder.sample_with_log_prob(context=x)
        else:         z2, logqz2 = self.encoder.sample_with_log_prob(num_samples=x.shape[0])
        z = torch.cat([x, z2], dim=self.split_dim)
        ldj = -logqz2
        return z, ldj

    def inverse(self, z):
        x, z2 = self.split_z(z)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth, dropout):
        super(DenseLayer, self).__init__()

        layers = []

        layers.extend([
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        ])

        if dropout > 0.:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Conv2d(in_channels, growth, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 3,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        h = self.conv(x)
        a, b, c = torch.chunk(h, chunks=3, dim=1)
        return a + b * torch.sigmoid(c)


class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):

        layers = [DenseLayer(in_channels+i*growth, growth, dropout) for i in range(depth)]

        if gated_conv:
            layers.append(GatedConv2d(in_channels+depth*growth, out_channels, kernel_size=1, padding=0))
        else:
            layers.append(nn.Conv2d(in_channels+depth*growth, out_channels, kernel_size=1, padding=0))

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        super(DenseBlock, self).__init__(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):
        super(ResidualDenseBlock, self).__init__()

        self.dense = DenseBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                depth=depth,
                                growth=growth,
                                dropout=dropout,
                                gated_conv=gated_conv,
                                zero_init=zero_init)

    def forward(self, x):
        return x + self.dense(x)


class DenseNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False):

        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)] +\
                 [ResidualDenseBlock(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)] +\
                 [nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)]

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        super(DenseNet, self).__init__(*layers)


class ElementwiseParams2d(nn.Module):
    '''
    Move elementwise parameters to last dimension.
    Ex.: For image of shape (B,C,H,W) with P elementwise parameters,
    the input takes shape (B,P*C,H,W) while the output takes shape (B,C,H,W,P).
    Args:
        num_params: int, number of elementwise parameters P.
        mode: str, mode of channels (see below), one of {'interleaved', 'sequential'} (default='interleaved').
    Mode:
        Ex.: For C=3 and P=2, the input is assumed to take the form along the channel dimension:
        - interleaved: [R G B R G B]
        - sequential: [R R G G B B]
        while the output takes the form [R G B].
    '''

    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams2d, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 4, 'Expected input of shape (B,C,H,W)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            channels = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * channels , height, width)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, channels) + x.shape[2:])
                # x.shape = (bs, num_params, channels, height, width)
                x = x.permute([0,2,3,4,1])
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (channels, self.num_params) + x.shape[2:])
                # x.shape = (bs, channels, num_params, height, width)
                x = x.permute([0,1,3,4,2])
            # x.shape = (bs, channels, height, width, num_params)
        return x


class Slice(Surjection):
    '''
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    stochastic_forward = False

    def __init__(self, decoder, num_keep, dim=1):
        super(Slice, self).__init__()
        assert dim >= 1
        self.decoder = decoder
        self.dim = dim
        self.num_keep = num_keep
        self.cond = isinstance(self.decoder, ConditionalDistribution)

    def split_input(self, input):
        split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x):
        z, x2 = self.split_input(x)
        if self.cond: ldj = self.decoder.log_prob(x2, context=z)
        else:         ldj = self.decoder.log_prob(x2)
        return z, ldj

    def inverse(self, z):
        if self.cond: x2 = self.decoder.sample(context=z)
        else:         x2 = self.decoder.sample(num_samples=z.shape[0])
        x = torch.cat([z, x2], dim=self.dim)
        return x


class _ActNormBijection(Bijection):
    '''
    Base class for activation normalization [1].
    References:
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
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
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
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


class ActNormBijection2d(_ActNormBijection):
    '''
    Activation normalization [1] for inputs on the form (B,C,H,W).
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    References:
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def register_params(self):
        '''Register parameters shift and log_scale'''
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.num_features, 1, 1)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features, 1, 1)))

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        x_mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
        x_std = torch.std(x, dim=[0, 2, 3], keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return x.shape[2:4].numel()


class Conv1x1(Bijection):
    """
    Invertible 1x1 Convolution [1].
    The weight matrix is initialized as a random rotation matrix
    as described in Section 3.2 of [1].
    Args:
        num_channels (int): Number of channels in the input and output.
        orthogonal_init (bool): If True, initialize weights to be a random orthogonal matrix (default=True).
        slogdet_cpu (bool): If True, compute slogdet on cpu (default=True).
    Note:
        torch.slogdet appears to run faster on CPU than on GPU.
        slogdet_cpu is thus set to True by default.
    References:
        [1] Glow: Generative Flow with Invertible 1x1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    """
    def __init__(self, num_channels, orthogonal_init=True, slogdet_cpu=True):
        super(Conv1x1, self).__init__()
        self.num_channels = num_channels
        self.slogdet_cpu = slogdet_cpu
        self.weight = nn.Parameter(torch.Tensor(num_channels, num_channels))
        self.reset_parameters(orthogonal_init)

    def reset_parameters(self, orthogonal_init):
        self.orthogonal_init = orthogonal_init

        if self.orthogonal_init:
            nn.init.orthogonal_(self.weight)
        else:
            bound = 1.0 / np.sqrt(self.num_channels)
            nn.init.uniform_(self.weight, -bound, bound)

    def _conv(self, weight, v):
        
        # Get tensor dimensions
        _, channel, *features = v.shape
        n_feature_dims = len(features)
        
        # expand weight matrix
        fill = (1,) * n_feature_dims
        weight = weight.view(channel, channel, *fill)

        if n_feature_dims == 1:
            return F.conv1d(v, weight)
        elif n_feature_dims == 2:
            return F.conv2d(v, weight)
        elif n_feature_dims == 3:
            return F.conv3d(v, weight)
        else:
            raise ValueError(f'Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d')

    def _logdet(self, x_shape):
        b, c, *dims = x_shape
        if self.slogdet_cpu:
            _, ldj_per_pixel = torch.slogdet(self.weight.to('cpu'))
        else:
            _, ldj_per_pixel = torch.slogdet(self.weight)
        ldj = ldj_per_pixel * reduce(mul, dims)
        return ldj.expand([b]).to(self.weight.device)

    def forward(self, x):
        z = self._conv(self.weight, x)
        ldj = self._logdet(x.shape)
        return z, ldj

    def inverse(self, z):
        weight_inv = torch.inverse(self.weight)
        x = self._conv(weight_inv, z)
        return x
