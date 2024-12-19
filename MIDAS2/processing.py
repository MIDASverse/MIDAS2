"""Data processing utils for MIDAS2."""

import pandas as pd
import numpy as np


def _process_missing(X):
    """
    Set missings to 0 and return missing matrix.

    """
    data = X.copy()
    data = np.nan_to_num(data, 0)
    missings = np.isnan(X)
    return data, missings


def _format_cols(data: pd.DataFrame):
    """
    Format input pandas DataFrame for use in MIDAS model.

    Parameters:
    data: pd.DataFrame
    verbose: bool. Print column information post-transformation.
        This is useful for checking the auto-conversion has worked.


    """

    # hash maps for column types
    type_dict = {}

    # sanity check
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    # store orderings
    col_names = []
    col_types = []

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category")

        if data[col].dtype.name == "category":

            cats = data[col].cat.categories

            if len(cats) == 2:
                data[col] = data[col].cat.codes
                col_types.append("bin")
                col_names += [col]
            else:
                na_temp = data[col].isnull()
                tmp_df = pd.get_dummies(data[col], prefix=col, dtype=float)
                tmp_df[na_temp] = np.nan
                data = data.drop(col, axis=1, inplace=False)
                data = pd.concat([data, tmp_df], axis=1)
                col_types.append(tmp_df.shape[1])
                col_names += list(tmp_df.columns.values)

            type_dict[col] = cats

        elif data[col].dtype.name == "boolean":
            col_types.append("bin")
            col_names += [col]

        elif data[col].dtype.name[:5] == "float" or data[col].dtype.name[:3] == "int":

            if set(data[col].astype(float).unique()) == {0.0, 1.0}:
                data[col] = data[col].astype("category")
                type_dict[col] = data[col].cat.categories
                data[col] = data[col].cat.codes
                col_types.append("bin")
            elif data[col].min() >= 0:
                type_dict[col] = "positive"
                col_types.append("pos")
            else:
                type_dict[col] = "numeric"
                col_types.append("num")

            col_names += [col]
        else:
            raise ValueError(
                f"Column {col} has unrecognized type. Please convert to numeric or categorical."
            )

    return [data.loc[:, col_names], type_dict, col_types]
