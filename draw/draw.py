import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .attention import BaseAttention, FilterbankAttention


class DRAW(nn.Module):
    def __init__(self, config, x_shape):
        super(DRAW, self).__init__()
        self.h_dim = config['h_dim']
        self.x_dim = x_shape[0] * x_shape[1]
        self.z_dim = config['z_dim']
        self.T = config['T']
        self.N = config['N']

        # instantiate distribution layers
        self.variational = nn.Linear(self.h_dim, 2*self.z_dim)
        self.observation = nn.Linear(self.x_dim, self.x_dim)

        # define attention module
        if config['attention'] == 'base':
            self.attention = BaseAttention(self.h_dim, self.x_dim)
            enc_dim = 2*self.x_dim + self.h_dim
        elif config['attention'] == 'filterbank':
            self.attention = FilterbankAttention(self.h_dim, self.x_dim, x_shape, self.N)
            enc_dim = 2*self.N**2 + self.h_dim
        else:
            raise Exception(f"Error: Attention module '{config['attention']}' not implemented.")

        # Recurrent encoder/decoder using LSTM
        self.encoder = nn.LSTMCell(enc_dim, self.h_dim)
        self.decoder = nn.LSTMCell(self.z_dim, self.h_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # define initial state as zeros
        h_enc = x.new_zeros((batch_size, self.h_dim))
        h_dec = x.new_zeros((batch_size, self.h_dim))
        c_enc = x.new_zeros((batch_size, self.h_dim))
        c_dec = x.new_zeros((batch_size, self.h_dim))
        
        # prior
        p_mu = x.new_zeros((batch_size, self.z_dim))
        p_std = x.new_ones((batch_size, self.z_dim))
        self.prior = Normal(p_mu, p_std)

        # initial canvas
        canvas = x.new_zeros((batch_size, self.x_dim))

        # initial loss
        kl = 0

        for _ in range(self.T):
            # calculate error
            x_hat = x - torch.sigmoid(canvas)

            # use attention to read image: N x N
            r_t = self.attention.read(x, x_hat, h_dec)

            # pass throuygh encoder
            h_enc, c_enc = self.encoder(
                torch.cat([r_t, h_dec], dim=1),
                [h_enc, c_enc]
            )

            # sample from distribution
            q_mu, q_log_std = torch.split(
                self.variational(h_enc), self.z_dim, dim=1
            )
            Q_t = Normal(q_mu, torch.exp(q_log_std)) 
            z_t = Q_t.rsample()

            # pass through decoder
            h_dec, c_dec = self.decoder(z_t, [h_dec, c_dec])

            # write on canvas: A x B
            canvas += self.attention.write(h_dec)

            # add loss
            kl += kl_divergence(Q_t, self.prior)

        # reconstruction
        x_mu = self.observation(canvas)
        return [x_mu, kl]

    def sample(self, z=None):
        """ Generate a sample from the distribution """
        if z is None:
            z = self.prior.sample()
        batch_size = z.size(0)

        # initial 
        canvas = z.new_zeros((batch_size, self.x_dim))
        h_dec = z.new_zeros((batch_size, self.h_dim))
        c_dec = z.new_zeros((batch_size, self.h_dim))

        for _ in range(self.T):
            h_dec, c_dec = self.decoder(z, [h_dec, c_dec])
            canvas += self.attention.write(h_dec)
        
        x_mu = self.observation(canvas)
        return x_mu