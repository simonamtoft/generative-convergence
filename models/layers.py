import torch
import torch.nn as nn


class ElementwiseParams(nn.Module):
    '''
    Move elementwise parameters to last dimension.
    Ex.: For an input of shape (B,D) with P elementwise parameters,
    the input takes shape (B,P*D) while the output takes shape (B,D,P).
    Args:
        num_params: int, number of elementwise parameters P.
        mode: str, mode of channels (see below), one of {'interleaved', 'sequential'} (default='interleaved').
    Mode:
        Ex.: For D=3 and P=2, the input is assumed to take the form along dimension 1:
        - interleaved: [1 2 3 1 2 3]
        - sequential: [1 1 2 2 3 3]
        while the output takes the form [1 2 3].
    '''

    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 2, 'Expected input of shape (B,D)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            dims = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * dims)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, dims))
                # x.shape = (bs, num_params, dims)
                x = x.permute([0,2,1])
                # x.shape = (bs, dims, num_params)
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (dims, self.num_params))
                # x.shape = (bs, dims, num_params)
        return x
