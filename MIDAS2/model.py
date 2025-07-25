"""Main class for MIDAS imputer"""

import random
from typing import Generator

import torch
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from scipy.special import expit  # for sigmoid function

from .mixed_activation import MixedActivation
from .dataset import Dataset
from .custom_loss import _masked_loss, MixedLoss


class MIDAS(torch.nn.Module):
    """Multiple Imputation using Denoising Autoencoders (MIDAS)

    Parameters:

    hidden_layers: The number of nodes in each hidden encoder layer.
        The node sizes are reversed for the decoder portion.
    dropout_prob: The dropout probability for each hidden layer

    Notes:

    MIDAS(2) follows the sklearn pipeline. You first declare an
    imputation model, .fit() it to your data, then .transform(m) to
    return m imputed datasets. You can use .fit_transform()
    to do both at once.
    """

    def __init__(
        self,
        hidden_layers: list[int] = [256, 128, 64],
        dropout_prob: float = 0.5,
        device=None,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.dropout_prob = dropout_prob
        self.seed = None
        self.dataset = None
        self.input_dim = None
        self.encoder = None
        self.decoder = None
        self.mal = None
        self.omit_first = None
        self.device = device

    def forward(self, x):
        """
        Forward pass through the model.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.mal(decoded)
        return output

    def fit(
        self,
        # data handling
        X: pd.DataFrame,
        col_convert: bool = True,
        col_types: list = None,
        type_dict: dict = None,
        # training
        epochs: int = 75,
        batch_size: int = 64,
        lr: float = 0.001,
        num_adj: float = 1,
        cat_adj: float = 1,
        bin_adj: float = 1,
        pos_adj: float = 1,
        # CI testing
        omit_first: bool = False,
        # utils
        verbose: bool = True,
        seed: int = None,
    ) -> None:
        """Fit the MIDAS model to the data.

        Parameters:

            --Data--

            X: The data to be imputed.
            col_convert: If True, column types are inferred from the data.
                If False, you must provide col_types and type_dict.
            col_types: A list of column types.
                If col_convert == False, you must provide col_types.
                Only used, if col_convert == False.
            type_dict: A dictionary of column types.
                If col_convert == False, you must provide type_dict.

            --Training hyperparameters--

            epochs: The number of epochs to train the model.
            batch_size: The batch size for training.
            lr: The learning rate for training.
            num_adj: The loss multiplier for numerical columns.
            cat_adj: The loss multiplier factor for categorical columns.
            bin_adj: The loss multiplier factor for binary columns.
            pos_adj: The loss multiplier factor for positional columns.

            --CI testing--
            omit_first: If True, the first column is omitted from the model *inputs*.
                This is useful for imputing a dataset where you do not want the outcome
                (stored in the first column) to influence the imputation values.

            --Utils--

            verbose: If True, print the loss at each epoch.
            seed: The random seed for reproducibility.

        """

        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        else:
            self.seed = None

        self.omit_first = omit_first

        ### DATA PREPROCESSING ###
        if col_convert:
            self.dataset = Dataset(X)
        elif col_types is not None and type_dict is not None:
            self.dataset = Dataset(X, col_types, type_dict)
        else:
            raise ValueError(
                "If col_convert == False, you must provide col_types and type_dict."
            )

        self.input_dim = self.dataset.data.shape[1]

        self._build_model()

        self._train_model(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_adj=num_adj,
            cat_adj=cat_adj,
            bin_adj=bin_adj,
            pos_adj=pos_adj,
            verbose=verbose,
        )

    def _build_model(self):
        self.encoder = torch.nn.Sequential()

        if self.omit_first:
            prev_dim = self.input_dim - 1
        else:
            prev_dim = self.input_dim

        for i, hidden_dim in enumerate(self.hidden_layers):
            self.encoder.add_module(
                f"layer_{i+1}", torch.nn.Linear(prev_dim, hidden_dim)
            )
            self.encoder.add_module(
                f"dropout_{i+1}", torch.nn.Dropout(self.dropout_prob)
            )
            self.encoder.add_module(f"relu_{i+1}", torch.nn.ReLU())
            prev_dim = hidden_dim

        # create decoder

        self.decoder = torch.nn.Sequential()
        decode_layers = self.hidden_layers
        decode_layers.reverse()

        for i, hidden_dim in enumerate(decode_layers[:-1]):
            self.decoder.add_module(
                f"layer_{i+1}", torch.nn.Linear(hidden_dim, decode_layers[i + 1])
            )
            self.decoder.add_module(
                f"dropout_{i+1}", torch.nn.Dropout(self.dropout_prob)
            )
            self.decoder.add_module(f"relu_{i+1}", torch.nn.ReLU())

        self.decoder.add_module(
            "output", torch.nn.Linear(decode_layers[-1], self.input_dim)
        )

        # mixed activation layer
        self.mal = MixedActivation(self.dataset.col_types)

    def _train_model(
        self,
        epochs,
        batch_size,
        lr,
        corrupt_rate=0.8,
        num_adj=1,
        cat_adj=1,
        bin_adj=1,
        pos_adj=1,
        verbose=True,
    ):
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        loss_fn = MixedLoss(
            self.dataset.col_types,
            num_adj=num_adj,
            cat_adj=cat_adj,
            bin_adj=bin_adj,
            pos_adj=pos_adj,
            device=device,
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            epoch_loss = 0
            for _, (x, mask) in enumerate(dataloader):

                x = x.to(device)
                x_corrupted = x * torch.bernoulli(torch.fill(x, corrupt_rate))
                mask = mask.to(device)
                optimizer.zero_grad()

                if self.omit_first:
                    pred = self(x_corrupted[:, 1:])
                else:
                    pred = self(x_corrupted)

                mixed_losses = loss_fn(pred, x)

                loss = _masked_loss(mixed_losses, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch: {epoch} Loss: {epoch_loss}")

    def transform(
        self,
        X: np.ndarray = None,
        m: int = 5,
        revert_cols: bool = True,
        format_X: bool = False,
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield imputations of X (or missing values) using trained MIDAS model.

        Parameters:

        X: The data to be imputed. If None, the model is applied to the training data.
        m: The number of imputations to generate.
        revert_cols: If False, no transformations are applied post-imputation; binary
            and categorical columns will be returned as logits, not probabilities.
        format_X: If True, the data is formatted to match the training data
            (very important if you are passing in new test data for imputation!)

        Notes:

            1) The imputed values are returned as a generator object.
            2) It is possible to pre-train a MIDAS model and apply it to new data by passing X.

        """

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        device = next(self.parameters()).device

        if X is None:
            X = self.dataset
        else:
            X = Dataset(
                X,
                col_types=self.dataset.col_types,
                type_dict=self.dataset.type_dict,
                col_names=self.dataset.col_names,
                test_format=format_X,
            )

        with torch.no_grad():
            data_np = X.data.cpu().numpy()
            for _ in range(m):
                input_tensor = X.data
                if self.omit_first:
                    input_tensor = input_tensor[:, 1:]

                imputed = self(input_tensor.to(device)).cpu().numpy()

                imputed[X.mask_expand] = data_np[X.mask_expand]

                if revert_cols:

                    imputed = pd.DataFrame(imputed)

                    for i, col in enumerate(X.col_types):
                        if col == "bin":
                            imputed.iloc[:, i] = expit(imputed.iloc[:, i])

                            tmp_bin = np.where(
                                imputed.iloc[:, i] > 0.5,  # hard-coded threshold
                                X.type_dict[X.col_names[i]][1],
                                X.type_dict[X.col_names[i]][0],
                            )

                            imputed.drop(imputed.columns[i], axis=1, inplace=True)
                            imputed.insert(i, X.col_names[i], tmp_bin)

                        elif isinstance(col, int):
                            tmp_cat = [
                                X.type_dict[X.col_names[i]][x]
                                for x in np.argmax(
                                    imputed.iloc[:, i : i + col].to_numpy(), axis=1
                                )
                            ]
                            imputed.drop(
                                [imputed.columns[j] for j in range(i, i + col)],
                                axis=1,
                                inplace=True,
                            )
                            imputed.insert(i, X.col_names[i], tmp_cat)

                    imputed.columns = X.col_names

                yield imputed

    def fit_transform(self, data, m=5, **kwargs) -> Generator[pd.DataFrame, None, None]:
        """Fit the model to the data and return imputed datasets.

        Parameters:

        data: The data to be imputed.
        m: The number of imputations to generate.

        """

        self.fit(data, **kwargs)
        return self.transform(m=m)
