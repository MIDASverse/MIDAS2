"""Test MIDAS2 pipeline"""

import random
import torch
import numpy as np
import pandas as pd

from MIDAS2 import model as md

np.random.seed(89)
torch.manual_seed(89)
random.seed(89)

n = 20

data = pd.DataFrame(
    {
        "xn": np.random.randn(n).astype("float32"),
        "xc": np.random.choice(["a", "b", "c"], n),
        "xb": np.random.choice(["x", "y"], n),
        "xb2": np.random.choice([0, 1], n).astype("float32"),
        "xn2": np.random.randint(0, 100, n).astype("float32"),
    }
)

data["xc2"] = np.where(data["xn"] > 0, "a", "b")

for col in data.columns:
    data[col] = np.where(np.random.uniform(0, 1, n) < 0.1, np.nan, data[col])

data.iloc[0, 1] = np.nan

mod = md.MIDAS()
mod.fit(data, epochs=10, batch_size=2, seed=89)

a = mod.transform(m=5)

for df in a:
    print(df.loc[0, "xc"])
