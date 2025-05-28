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

    data = data.copy()

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

        if data[col].dtype == "boolean":
            data[col] = 1 * data[col]
            col_types.append("bin")
            col_names += [col]
            type_dict[col] = [False, True]

        elif data[col].dtype.name == "category":

            cats = data[col].cat.categories
            na_tmp = data[col].isnull()

            # handle binary categorical columns
            if len(cats) == 2:
                tmp_col = 1 * data[col].eq(cats[1])
                tmp_col[na_tmp] = np.nan
                data[col] = tmp_col
                col_types.append("bin")
                col_names += [col]
            else:
                tmp_df = pd.get_dummies(data[col], prefix=col, dtype=float)
                tmp_df[na_tmp] = np.nan
                data = data.drop(col, axis=1, inplace=False)
                data = pd.concat([data, tmp_df], axis=1)
                col_types.append(tmp_df.shape[1])
                col_names += list(tmp_df.columns.values)

            type_dict[col] = cats

        elif data[col].dtype.name[:5] == "float" or data[col].dtype.name[:3] == "int":

            unq_vals = data[col].astype(float).unique()
            unq_vals = unq_vals[~np.isnan(unq_vals)]  # remove NaNs for comparison

            # handle binary columns
            if set(unq_vals) == {0.0, 1.0}:
                type_dict[col] = [0.0, 1.0]
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


def _format_cols_test(data: pd.DataFrame, col_types, type_dict, col_names):
    """
    Format input pandas DataFrame for use in MIDAS model.

    Parameters:
    data: pd.DataFrame
    verbose: bool. Print column information post-transformation.
        This is useful for checking the auto-conversion has worked.


    """
    # sanity check
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    data = data.copy()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category")

        if data[col].dtype == "boolean":
            data[col] = 1 * data[col]

        elif data[col].dtype.name == "category":

            cats = type_dict[col]
            na_tmp = data[col].isnull()

            if len(cats) == 2:
                tmp_col = 1 * data[col].eq(cats[1])
                tmp_col[na_tmp] = np.nan
                data[col] = tmp_col
            else:

                cats_df = pd.DataFrame(
                    np.zeros((data.shape[0], len(cats))),
                    columns=[f"{col}_{cat}" for cat in cats],
                )

                for i, cat in enumerate(data[col]):
                    cats_df.iloc[i, cats.get_loc(str(cat))] = (
                        1 if not na_tmp[i] else np.nan
                    )
                cats_df.loc[na_tmp, :] = np.nan
                data = data.drop(col, axis=1, inplace=False)
                data = pd.concat([data, cats_df], axis=1)

        elif data[col].dtype.name[:5] == "float" or data[col].dtype.name[:3] == "int":
            pass

        else:
            raise ValueError(
                f"Column {col} has unrecognized type. Please convert to numeric or categorical."
            )

    return data.loc[:, col_names]


# def _format_cols_test(data: pd.DataFrame, col_types, type_dict, col_names):
#     """
#     Format input pandas DataFrame for use in MIDAS model.

#     Parameters:
#     data: pd.DataFrame
#     verbose: bool. Print column information post-transformation.
#         This is useful for checking the auto-conversion has worked.


#     """
#     # sanity check
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("Data must be a pandas DataFrame")

#     # Create a new DataFrame to store the formatted data
#     formatted_data = pd.DataFrame()

#     # Process each column according to its type from training data
#     for i, col_type in enumerate(col_types):
#         if col_type == "bin":
#             # Binary column
#             if col_names[i] in data.columns:
#                 # If column exists in test data, process it
#                 if (
#                     data[col_names[i]].dtype == "object"
#                     or data[col_names[i]].dtype == "boolean"
#                 ):
#                     data[col_names[i]] = data[col_names[i]].astype("category")

#                 if data[col_names[i]].dtype.name == "category":
#                     # Ensure categories match training data
#                     orig_cats = type_dict.get(col_names[i], None)
#                     if orig_cats is not None:
#                         # Map values to match training categorical encoding
#                         data[col_names[i]] = data[col_names[i]].map(
#                             lambda x: (
#                                 np.nan
#                                 if pd.isna(x)
#                                 else (
#                                     np.where(orig_cats == x)[0][0]
#                                     if x in orig_cats
#                                     else np.nan
#                                 )
#                             )
#                         )
#                     else:
#                         # Use cat.codes for binary variables
#                         data[col_names[i]] = data[col_names[i]].cat.codes

#                 formatted_data[col_names[i]] = data[col_names[i]]
#             else:
#                 # Column doesn't exist, fill with NaNs
#                 formatted_data[col_names[i]] = np.nan

#         elif col_type == "pos" or col_type == "num":
#             # Numeric column
#             if col_names[i] in data.columns:
#                 formatted_data[col_names[i]] = data[col_names[i]]
#             else:
#                 formatted_data[col_names[i]] = np.nan

#         else:
#             # One-hot encoded categorical column
#             # Find the original column name (removing the prefix from dummy variables)
#             prefix = col_names[i].split("_")[0]
#             if prefix in data.columns:
#                 # Create dummies with the same categories as in training
#                 if data[prefix].dtype == "object" or data[prefix].dtype == "boolean":
#                     data[prefix] = data[prefix].astype("category")

#                 # Create dummy variables for all categories from training
#                 dummies = pd.get_dummies(data[prefix], prefix=prefix, dtype=float)

#                 # Add missing columns (categories in training but not in test)
#                 for col in col_names[i : i + col_type]:
#                     if col not in dummies.columns:
#                         dummies[col] = 0

#                 # Add to formatted data in correct order
#                 for j in range(col_type):
#                     col = col_names[i + j]
#                     if col in dummies.columns:
#                         formatted_data[col] = dummies[col]
#                     else:
#                         formatted_data[col] = 0

#                 # Skip the other dummy columns we've just processed
#                 i += col_type - 1
#             else:
#                 # Original column doesn't exist, add all dummy columns as zeros
#                 for j in range(col_type):
#                     formatted_data[col_names[i + j]] = 0

#                 # Skip the other dummy columns we've just processed
#                 i += col_type - 1
#     # Ensure columns are in the correct order
#     return formatted_data[col_names]
