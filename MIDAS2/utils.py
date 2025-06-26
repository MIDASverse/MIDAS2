import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy import stats


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


def combine(
    dfs,
    y,
    ind_vars=None,
    dof_adjust=True,
    incl_constant=True,
    **glm_args,
):
    """
    Function used to run a GLM model across multiple datasets, aggregating the
    results using Rubin's combination rules -- i.e. multiple imputation analysis.

    This function regresses the outcome variable on a linear combination of
    independent variables, given a user-specified model family and link function.
    For example if y_var = 'y' and X_vars = ['x1','x2','x3'], then by default this
    function estimates the model y = a + x1 + x2 + x3, where a is the constant term.
    Note, the constant term is added by default, but can be excluded by setting
    incl_constant = False.

    This function wraps statsmodels.GLM() and allows users to specify linear
    models using GLM families including Gaussian, Binomial, and Poisson.

    The function can be called on the completed dataframes generated from a MIDAS
    model or users can supply their own list of completed datasets to analyse.

    Args:
      df_list: A list of pd.DataFrames. The M completed datasets to be analyzed.

      y_var: String. The name of the outcome variable.

      X_vars: List of strings. The names of the predictor variables.

      dof_adjust: Boolean. Indicates whether to apply the Barnard and Rubin (1999)
      degrees of freedom adjustment for small-samples.

      incl_constant: Boolean. Indicates whether to include an intercept in the null model (the default in
      most generalized linear model software packages).

      **glm_args: Further arguments to be passed to statsmodels.GLM(), e.g., to
      specify model family, offsets, and variance and frequency weights (see the
      statsmodels documentation for full details). If None, a Gaussian (ordinary
      least squares) model will be estimated.

    Returns:
      DataFrame of combined model results"""

    mods_est = []
    mods_var = []
    m = len(dfs)

    for i in range(m):

        df_mod = dfs[i]

        # will only trigger when i = 0 and no ind_vars supplied
        if ind_vars is None:
            ind_vars = df_mod.columns[df_mod.columns != y]

        df_endog = df_mod[y]
        df_exog = df_mod[ind_vars]

        if incl_constant:
            df_exog = sm.add_constant(df_exog)

        ind_model = sm.GLM(df_endog, df_exog, **glm_args)
        ind_results = ind_model.fit()
        mods_est.append(ind_results.params)
        mods_var.append(np.diag(ind_results.cov_params()))

        if i == 0:
            mods_df_resid = ind_results.df_resid
            mods_coef_names = ind_results.model.exog_names

    Q_bar = np.multiply((1 / m), np.sum(np.array(mods_est), 0))
    U_bar = np.multiply((1 / m), np.sum(np.array(mods_var), 0))

    models_demean = list(map(lambda x: np.square(x - Q_bar), mods_est))

    B = np.multiply(1 / (m - 1), np.sum(np.array(models_demean), 0))

    Q_bar_var = U_bar + ((1 + (1 / m)) * B)
    Q_bar_se = np.sqrt(Q_bar_var)

    v_m = (m - 1) * np.square(1 + (U_bar / ((1 + m ** (-1)) * B)))

    if dof_adjust:

        v_complete = mods_df_resid

        gamma = ((1 + m ** (-1)) * B) / Q_bar_var

        v_obs = ((v_complete + 1) / (v_complete + 3)) * v_complete * (1 - gamma)

        v_corrected = ((1 / v_m) + (1 / v_obs)) ** (-1)

        dof = v_corrected

    else:

        dof = v_m

    est = Q_bar
    std_err = Q_bar_se
    stat = est / std_err

    combined_mat = {
        "term": mods_coef_names,
        "estimate": est,
        "std.error": std_err,
        "statistic": stat,
        "df": dof,
        "p.value": (2 * (1 - stats.t.cdf(abs(stat), df=dof))),
    }

    return pd.DataFrame(combined_mat)
