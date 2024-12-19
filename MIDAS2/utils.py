import numpy as np


def imp_mean(imputations):
    """
    For a generator imputation object, calculate mean of imputed values across m datasets.

    """

    imps = list(imputations)
    return np.mean(imps, axis=0)
