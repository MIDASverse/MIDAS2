# MIDAS(2)

Work in progress implementation of MIDAS in pytorch.

In addition to migrating to `torch`, this new version adds the following functionality:

* Models can be fit on `X` and used to impute on new data `X'`
* Automatic detection of column-types 

## Example usage

One major difference is that this version more closely follows the sklearn API, with fit and transform methods.

```
import MIDAS2 as md

# Create a MIDAS object
md.MIDAS()

# Fit the model to data
md.fit(X, epochs = 10)

# Multiply impute missing data
X_imputed = md.transform(X, m = 10)
```

