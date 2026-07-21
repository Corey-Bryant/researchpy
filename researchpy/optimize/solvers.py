from typing import Any

import numpy as np
from scipy.optimize import minimize, OptimizeResult



def _ols_estimation_principal(IV: np.ndarray, DV: np.ndarray) -> np.ndarray:
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




def _mle_estimation_principal(fun: object, x0: object, args: object = (), jac: object = None, method: object = None,
                              callback: object = None, options: object = None, ) -> OptimizeResult:
    """Perform Maximum Likelihood Estimation (MLE) using a specified optimization method.

    This is a wrapper around ``scipy.optimize.minimize`` that allows for flexible optimization of the likelihood function.

    Parameters
    ----------
    fun : callable
        The objective function to minimize.
    x0 : array-like
        Initial guess for the parameters, shape ``(k,)``.
    args : tuple, optional
        Extra arguments passed to the objective function.
    jac : callable, optional
        The gradient of the objective function.
    method : str, optional
        Optimization algorithm to use.
    callback : callable, optional
        User-defined callback function.
    options : dict, optional
        A dictionary of solver options. All methods except TNC accept the following generic options:

        maxiter : int
            Maximum number of iterations to perform. Depending on the method each iteration may use several function evaluations.
            For TNC use maxfun instead of maxiter.

        disp : bool
            Set to True to print convergence messages.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object. Important attributes are:
            ``x`` the solution array,
            ``success`` a Boolean flag indicating if the optimizer exited successfully, and
            ``message`` which describes the cause of the termination.
    """

    try:
        result = minimize(fun=fun, x0=x0, args=args, jac=jac, method=method, options=options, callback=callback)
        return result

    except Exception as e:
        print(f"Optimization failed: {e}")
        raise
