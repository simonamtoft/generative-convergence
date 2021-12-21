import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from .stochastic import GaussianSample
from .distributions import log_gaussian, log_standard_gaussian


class Encoder(nn.Module):
    """ Inference Network
    Encode the observation `x` into the parameters of the posterior distribution
    `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
    
    Infer the probability distribtuion p(z|x) from the data by
    fitting a variational distribtuion q(z|x).
    
    Inputs:
        dims (array) :  Dimensions of the networks on the form 
                        [input_dim, [hidden_dims], latent_dim] 
    Returns:
        Tuple of (z, mu, log(sigma^2))
    """
    def __init__(self, dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()

        # Setup network dimensions
        [x_dim, h_dim, z_dim] = dims
        neuron_dims = [x_dim, *h_dim]

        # Define the hidden layer as a stack of linear layers
        linear_layers = [
            nn.Linear(neuron_dims[i-1], neuron_dims[i]) 
            for i in range(1, len(neuron_dims))
        ]
        self.hidden = nn.ModuleList(linear_layers)

        # Define sampling function
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class BetaDecoder(nn.Module):
    """Generative network for the Beta VAE.
    Generate samples from the original distribution p(x).
    """
    def __init__(self, dims):
        super(BetaDecoder, self).__init__()

        # Setup network dimensions
        [z_dim, h_dim, x_dim] = dims
        neuron_dims = [z_dim, *h_dim]

        # Define hidden layer as a stack of linear and Tanh layers
        layers = []
        for i in range(1, len(neuron_dims)):
            layers.append(
                nn.Linear(neuron_dims[i - 1], neuron_dims[i])
            )
            layers.append(nn.Tanh())
        self.hidden = nn.Sequential(*layers)

        # reconstruction and activation function
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.activation(self.reconstruction(x))


class Decoder(nn.Module):
    """Generative network.
    Generate samples from the original distribution p(x).
    
    Inputs:
        dims (array) :  Dimensions of the networks on the form
                        [latent_dim, [hidden_dims], input_dim] 
    Returns:
        decoded x
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()

        # Setup network dimensions
        [z_dim, h_dim, x_dim] = dims
        neuron_dims = [z_dim, *h_dim]

        # Define the hidden layer as a stack of linear layers
        linear_layers = [
            nn.Linear(neuron_dims[i-1], neuron_dims[i]) 
            for i in range(1, len(neuron_dims))
        ]
        self.hidden = nn.ModuleList(linear_layers)

        # Define reconstruction layer and activation function
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    """
        Variational Autoencoder model consisting of the encoder + decoder.
    Inputs:
        dims (array) :  Dimensions of the networks on the form
                        [input_dim, [hidden_dims], latent_dim]
         
    """
    def __init__(self, config, x_dim, x_list=True):
        super(VariationalAutoencoder, self).__init__()
        # setup network dimensions
        h_dim = config['h_dim']
        z_dim = config['z_dim']
        
        if isinstance(x_dim, list):
            enc_dims = [x_dim[0], h_dim, z_dim]
            dec_dim = [z_dim, list(reversed(h_dim)), x_dim[1]]
        else:
            enc_dims = [x_dim, h_dim, z_dim]
            dec_dim = [z_dim, list(reversed(h_dim)), x_dim]

        # Define encoder 
        self.encoder = Encoder(enc_dims)
        
        # Define decoder
        if config['as_beta']:
            self.decoder = BetaDecoder(dec_dim)
        else: 
            self.decoder = Decoder(dec_dim)

        # zero out the biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _kld(self, z, q_param, p_param=None):
        """Compute KL-divergence of some element z.    
        Inputs:
            z           : sample from the q distribution
            q_param     : (mu, log_var) of the q distribution.
            p_param     : (mu, log_var) of the p distribution.
        
        Returns:
            KL-divergence of q||p 
        """

        # Define q distribution
        (mu, log_var) = q_param
        qz = log_gaussian(z, mu, log_var)

        # Define p distribution
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        
        kl = qz - pz
        return kl

    def forward(self, x):
        """
            Run datapoint through model to reconstruct input
        Inputs
            x       :   input data
        Returns
            x_mu    :   reconstructed input
            kld     :   computed kl-divergence
        """
        # fit q(z|x) to x
        z, z_mu, z_log_var = self.encoder(x)

        # compute KL-divergence
        kld = self._kld(z, (z_mu, z_log_var))

        # reconstruct input via. decoder
        x_mu = self.decoder(z)
        return x_mu, kld

    def sample(self, z):
        """ Sampling from model
        Given a z ~ N(0, I) it generates a sample from the 
        learned distribution based on p_theta(x|z)
        """
        return self.decoder(z)
