import numpy as np


def predict_y(mdl_data, trans=None):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    Returns
    -------
    Array
        Returns an array containing the linear prediction.

    """

    if trans is None:
        y_e = mdl_data.IV @ mdl_data.CoefResults.betas

    else:
        y_e = trans(mdl_data.IV @ mdl_data.CoefResults.betas) / \
              (1 + trans(mdl_data.IV @ mdl_data.CoefResults.betas))

        # linear_pred mdl_data.IV @ mdl_data.CoefResults.betas
        # y_e = 1 / (1 + np.exp(-linear_pred))

    return y_e


def residuals(mdl_data):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    Returns
    -------
    Array
        Returns an array containing the residuals.

    """
    predicted_y = mdl_data.IV @ mdl_data.CoefResults.betas
    resids = mdl_data.DV - predicted_y
    
    return resids


def _compute_hat_matrix(mdl_data):
    """Compute the hat matrix H = X(X'X)^{-1}X' on-the-fly."""
    x = np.asarray(mdl_data.IV)
    try:
        H = x @ np.linalg.inv(x.T @ x) @ x.T
    except np.linalg.LinAlgError:
        H = x @ np.linalg.pinv(x.T @ x) @ x.T
    return H


def standardized_residuals(mdl_data):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    Returns
    -------
    Array
        Returns an array containing the standardized residuals.

    """
    resids = residuals(mdl_data)

    H = _compute_hat_matrix(mdl_data)
    std_e = np.sqrt(
        (mdl_data.ModelEffects.mse * (1 - np.diag(H))))

    t = resids / np.reshape(std_e, (mdl_data.n, 1))

    return t


def studentized_residuals(mdl_data):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    Returns
    -------
    Array
        Returns an array containing the studentized (jackknifed) residuals.

    """

    d = []

    resid_standardized = standardized_residuals(mdl_data)
    n = mdl_data.n
    k = mdl_data.k - 1

    for i in range(0, n):

        r_i = resid_standardized[i]

        t_i = r_i * np.sqrt(((n - k - 2) / (n - k - 1 - r_i**2)))

        d.append(t_i)

    d = np.array(d).reshape(n, 1)

    return d


def leverage(mdl_data):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    Returns
    -------
    Array
        Returns an array containing the leverage of each observation.

    """

    H = _compute_hat_matrix(mdl_data)
    lev = np.diag(H).reshape(mdl_data.n, 1)

    return lev


def predict(mdl_data, estimate=None, trans=None, decimals=4):
    """


    Parameters
    ----------
    mdl_data : ols or anova data object
        Class object which is returned from the ols or anova command.

    estimate : string
        A string value to indicate which estimate is desired. Available options are:

            estimate in ["y", "xb"] : linear prediction
            estimate in ["residuals", "res", "r"] : residuals
            estimate in ["standardized_residuals", "standardized_r", "rstand"] : standardized residuals
            estimate in ["studentized_residuals", "student_r", "rstud"] : studentized (jackknifed) residuals
            estimate in ["leverage", "lev"] : The leverage of each observation


    Returns
    -------
    Array containing the desired estimate.

    """
    if estimate not in ["y", "xb", "residuals", "res", "r", "standardized_residuals", "standardized_r", "rstand", "studentized_residuals", "student_r", "rstud", "leverage", "lev"]:
        return print("\n", "ERROR: estimate option provided is not supported. Please use help(predict) for supported options.")

    if estimate in ["y", "xb"]:
        est = predict_y(mdl_data, trans=trans)
        return est.round(decimals)

    elif estimate in ["residuals", "res", "r"]:
        est = residuals(mdl_data)
        return est.round(decimals)

    elif estimate in ["standardized_residuals", "standardized_r", "rstand"]:
        est = standardized_residuals(mdl_data)
        return est.round(decimals)

    elif estimate in ["studentized_residuals", "student_r", "rstud"]:
        est = studentized_residuals(mdl_data)
        return est.round(decimals)
    
    elif estimate in ["leverage", "lev"]:
        est = leverage(mdl_data)
        return est.round(decimals)
