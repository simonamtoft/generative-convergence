import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseAttention(nn.Module):
    """ No attention module """
    def __init__(self, h_dim, x_dim):
        super(BaseAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.write_head = nn.Linear(h_dim, x_dim)

    def read(self, x, x_hat, h):
        return torch.cat([x, x_hat], dim=1)

    def write(self, h_dec):
        return self.write_head(h_dec)


class FilterbankAttention(nn.Module):
    """ Attention module using Filterbank matrices """
    def __init__(self, h_dim, x_dim, x_shape, N):
        super(FilterbankAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.N = N
        self.A = x_shape[0]
        self.B = x_shape[1]
        self.max_dim = torch.max(torch.tensor(x_shape))
        self.W_read = nn.Linear(h_dim, 5)
        self.W_write = nn.Linear(h_dim, self.N**2)
    
    def compute_F(self, params):
        # Unpack params
        gt_X = params[0]
        gt_Y = params[1]
        var = torch.exp(params[2] + 1e-8)
        dt = torch.exp(params[3])

        # calculate grid center
        g_X = ((self.A + 1) * (gt_X + 1) / 2).item()
        g_Y = ((self.B + 1) * (gt_Y + 1) / 2).item()

        # calculate stride
        d = (dt * (self.max_dim - 1) / (self.N - 1)).item()
        
        # compute filters
        F_X = torch.zeros((self.N, self.A)).to(device)
        F_Y = torch.zeros((self.N, self.B)).to(device)

        # construct mean vectors
        sub_val = (- self.N*0.5 - 0.5) * d
        add_val = (self.N-1 - self.N*0.5 - 0.5) * d
        mu_X = torch.linspace(
            g_X + sub_val, 
            g_X + add_val,
            self.N
        ).to(device)
        mu_Y = torch.linspace(
            g_Y + sub_val, 
            g_Y + add_val,
            self.N
        ).to(device)

        # Compute filter matrices
        for a in range(self.A):
            F_X[:, a] = torch.exp( -(a - mu_X)**2 / (2 * var))

        for b in range(self.B):
            F_Y[:, b] = torch.exp( -(b - mu_Y)**2 / (2 * var))

        # normalize filters (should each sum to 1)
        F_X = F_X / torch.sum(F_X)
        F_Y = F_Y / torch.sum(F_Y)
        return F_X, F_Y

    def read(self, x, x_hat, h):
        """ Performs the read operation with attention """
        params = self.W_read(h)[0]
        gamma = torch.exp(params[4])

        # reshape x
        x = torch.reshape(x, (-1, self.A, self.B))
        x_hat = torch.reshape(x_hat, (-1, self.A, self.B))

        # compute filterbank matrices
        F_X, F_Y = self.compute_F(params)

        # filter x and x_hat
        x_filt = gamma * torch.matmul(torch.matmul(F_Y, x), F_X.T)
        x_hat_filt = gamma * torch.matmul(torch.matmul(F_Y, x_hat), F_X.T)

        # reshape back again
        x_filt = torch.reshape(x_filt, (-1, self.N**2))
        x_hat_filt = torch.reshape(x_hat_filt, (-1, self.N**2))
        return torch.cat([x_filt, x_hat_filt], dim=1)

    def write(self, h_dec):
        params = self.W_read(h_dec)[0]
        F_X, F_Y = self.compute_F(params)
        w_t = torch.reshape(self.W_write(h_dec), (-1, self.N, self.N))
        c_new = torch.matmul(torch.matmul(F_Y.T, w_t), F_X) / torch.exp(params[4])
        return torch.reshape(c_new, (-1, self.A*self.B))
