'''torch.nn.Module for mixed type outputs'''

import torch

class MixedActivation(torch.nn.Module):
    '''
    Mix of column types in output layer.
    '''

    def __init__(self, col_types):

        super().__init__()
        self.col_types = col_types
        self.act_fns = _get_act_fn(col_types)

    def __repr__(self):
        return f"MixedActivation(features={self.col_types}, act_fns={self.act_fns})"

    def forward(self,x):
        '''Forward pass through MAL'''
        out = []

        c = 0
        for i, col in enumerate(self.col_types):
            if isinstance(col, int):
                out.append(self.act_fns[i](x[:,c:c+col]))
                c += col
            else:
                out.append(self.act_fns[i](x[:,c]).reshape(x.shape[0],-1))
                c += 1
        return torch.cat(out, dim = 1)

def _get_act_fn(col_types):
    '''Get activation functions for each column type'''
    act_fns = []
    for col in col_types:
        if col in ['num','bin'] or isinstance(col, int):
            act_fns.append(torch.nn.Identity())
        elif col == 'pos':
            act_fns.append(torch.nn.ReLU())
        else:
            raise ValueError(f"Unknown column type: {col}")
    return act_fns
