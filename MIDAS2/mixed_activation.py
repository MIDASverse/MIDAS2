"""torch.nn.Module for mixed type outputs"""

import torch
from torch import nn
import torch.nn.functional as F


class MixedActivation(torch.nn.Module):
    """
    Mix of column types in output layer.
    """

    def __init__(self, col_types, device=None):

        super().__init__()
        self.col_types = col_types

        pos = []
        c = 0
        for spec in col_types:
            if spec == "pos":
                pos.append(c)
                c += 1
            elif isinstance(spec, int):  # categorical block
                c += spec
            else:  # 'num' / 'bin'
                c += 1

        # moves with .to(device)
        self.register_buffer(
            "pos_idx", torch.tensor(pos, dtype=torch.long, device=device)
        )

    def __repr__(self):
        return f"MixedActivation(features={self.col_types}, act_fns={self.act_fns})"

    def forward(self, x):
        """Forward pass through MAL"""
        if self.pos_idx.numel() == 0:
            return x

        out = x.clone()
        out[:, self.pos_idx] = F.relu(out[:, self.pos_idx])

        return out


# def _get_act_fn(col_types):
#     """Get activation functions for each column type"""
#     act_fns = []
#     for col in col_types:
#         if col in ["num", "bin"] or isinstance(col, int):
#             act_fns.append(torch.nn.Identity())
#         elif col == "pos":
#             act_fns.append(torch.nn.ReLU())
#         else:
#             raise ValueError(f"Unknown column type: {col}")
#     return act_fns
