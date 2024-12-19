'''Define custom Dataset class for handling missing data.'''

import torch
import pandas as pd
from .processing import _format_cols

class Dataset(torch.utils.data.Dataset):
    '''
    Custom PyTorch Dataset class for handling missing data.
    
    If col_types is not provided, column types are inferred from the data.
    
    '''
    def __init__(
        self,
        data: pd.DataFrame,
        col_types: list = None,
        type_dict: dict = None,
        col_names: list[str] = None,
    ):
        super().__init__()
        self.mask = ~data.isnull().to_numpy() # mask of observed values

        if col_types is None:
            self.col_names = data.columns
            data, self.type_dict, self.col_types = _format_cols(data)
        else:
            self.col_types = col_types
            self.type_dict = type_dict
            self.col_names = col_names

        self.data = data.to_numpy()
        self.mask_expand = ~data.isnull().to_numpy()
        self.data[~self.mask_expand] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        x = self.data[index].copy()
        x_mask = self.mask[index]
        x_mask_expand = self.mask_expand[index]
        x[~x_mask_expand] = 0 # set missing to 0

        return x.astype('float32'), x_mask.astype('bool') # return mask to remove from loss function
