# MIDAS(2)

**Work in progress** implementation of MIDAS in pytorch. Please note that while preliminary testing has been conducted on this implementation, it is still in the early stages of development and we cannot guarantee its performance. Documentation for the main model and methods can be found as docstrings in the `MIDAS2.py` script. 

In addition to migrating to `torch`, this new version adds the following functionality:

* Models can be fit on `X` and used to impute on new data `X'`
* Automatic detection of column-types

## Example usage

The major syntactical difference to **MIDASpy** is that MIDAS2 follows the sklearn API, with fit and transform methods of an imputer object.

```python
from MIDAS2 import model as md

# Create a MIDAS model
mod = md.MIDAS()

# Fit the model to data
mod.fit(X, epochs = 10)

# Multiply impute missing data
X_imputed = mod.transform(X, m = 10)
```

## CHANGELOG

* Renamed the main module from 'MIDAS2' to 'model' (19/12/2024)
* Restructured the package for easier install (19/12/2024)