"""Define custom Dataset class for handling missing data."""

import torch
import numpy as np
import pandas as pd
from .processing import _format_cols, _format_cols_test


class Dataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for handling missing data.

    If col_types is not provided, column types are inferred from the data.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        col_types: list = None,
        type_dict: dict = None,
        col_names: list[str] = None,
        test_format: bool = False,
    ):
        super().__init__()
        mask_np = ~data.isnull().to_numpy()  # mask of observed values

        if col_types is None:
            self.col_names = data.columns
            data, self.type_dict, self.col_types = _format_cols(data)
        else:
            self.col_types = col_types
            self.type_dict = type_dict
            self.col_names = col_names

            if test_format:
                data = _format_cols_test(
                    data, self.col_types, self.type_dict, self.col_names
                )

        mask_expand_np = ~data.isnull().to_numpy()
        data_np = data.to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
        data_np[~mask_expand_np] = 0.0

        self.data = torch.as_tensor(data_np, dtype=torch.float32)
        self.mask = torch.as_tensor(mask_np, dtype=torch.bool)
        self.mask_expand = mask_expand_np

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        # x = self.data[index].copy()
        # x_mask = self.mask[index]
        # x_mask_expand = self.mask_expand[index]
        # x[~x_mask_expand] = 0  # set missing to 0

        # return x.astype("float32"), x_mask.astype(
        #     "bool"
        # )  # return mask to remove from loss function

        return self.data[index], self.mask[index]
