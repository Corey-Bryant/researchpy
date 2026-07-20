import numpy as np


def ordinary_least_squares(IV: np.ndarray, DV: np.ndarray) -> np.ndarray:
    """
    Perform Ordinary Least Squares (OLS) regression to estimate coefficients.

    Parameters
    ----------
    IV : np.ndarray
        The design matrix (independent variables).
    DV : np.ndarray
        The response vector (dependent variable).

    Returns
    -------
    np.ndarray
        Estimated coefficients (betas) for the regression model.
    """
    # Calculate the OLS coefficients using the normal equation
    try:
        betas = np.linalg.inv((np.asarray(IV.T) @ np.asarray(IV))) @ np.asarray(IV.T) @ np.asarray(DV)
    except:
        betas = np.linalg.pinv((np.asarray(IV.T) @ np.asarray(IV))) @ np.asarray(IV.T) @ np.asarray(DV)

    return betas
