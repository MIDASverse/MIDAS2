"""Run MAR-1 simulation from Lall and Robinson (2022)"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from MIDAS2 import model as md


## Data generation as per King et al (2001)
def MAR1(n: int):
    """
    Generate data from a multivariate normal distribution with MAR-1 covariances.
    See King et al (2001) for details.
    """
    dta = np.random.multivariate_normal(
        mean=[0, 0, 0, 0, 0],
        cov=[
            [
                1,
                -0.12,
                -0.1,
                0.5,
                0.1,
            ],
            [
                -0.12,
                1,
                0.1,
                -0.6,
                0.1,
            ],
            [
                -0.1,
                0.1,
                1,
                -0.5,
                0.1,
            ],
            [
                0.5,
                -0.6,
                -0.5,
                1,
                0.1,
            ],
            [0.1, 0.1, 0.1, 0.1, 1],
        ],
        size=n,
    )
    return pd.DataFrame(dta, columns=["x" + str(i) for i in range(1, 6)])


def make_missing(data: pd.DataFrame):
    """
    Add MAR-1 missingness to generated data.
    """
    n = data.shape[0]
    M = np.ndarray(data.shape, dtype=bool)
    U1 = np.random.uniform(0, 1, n)

    # Y and X4 are MCAR:
    M[:, 0] = U1 < 0.85
    M[:, 4] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 and X2 are MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data.iloc[:, 3] < -1, U2 < 0.9], axis=0)

    U3 = np.random.uniform(0, 1, n)
    M[:, 2] = ~np.all([data.iloc[:, 3] < -1, U3 < 0.9], axis=0)

    data[~M] = np.nan

    return data


if __name__ == "__main__":

    np.random.seed(89)
    torch.manual_seed(89)

    train_data = MAR1(5000)
    missing_data = make_missing(train_data.copy())

    # train imputer
    MIDAS_model = md.MIDAS()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("midas_training"):
            MIDAS_model.fit(
                missing_data, epochs=250, batch_size=256, lr=0.01, verbose=False
            )

        with record_function("midas_imputation"):
            imputed_data = MIDAS_model.transform(m=10)
