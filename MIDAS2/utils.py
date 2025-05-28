import numpy as np
import pandas as pd


def imp_mean(imputations, pandas=False):
    """
    For a generator imputation object, calculate mean of imputed values across m datasets.

    """

    imps = list(imputations)
    if pandas:
        pd_df = pd.DataFrame(np.mean(imps, axis=0))
        pd_df.columns = imps[0].columns
        return pd_df
    else:
        return np.mean(imps, axis=0)
