from typing import Any

from numpy import asarray, ndarray, linalg
from scipy.optimize import minimize, OptimizeResult



def ols_estimation_principal(IV: ndarray, DV: ndarray) -> ndarray:
    """
    Perform Ordinary Least-Squares (OLS) regression using the normal equation to estimate coefficients.

    Parameters
    ----------
    IV : ndarray
        The independent variable(s) design matrix.
    DV : ndarray
        The dependent variable response vector.

    Returns
    -------
    ndarray
        Estimated coefficients (betas) for the regression model.
    """
    # Calculate the OLS coefficients using the normal equation
    try:
        betas = linalg.inv((asarray(IV.T) @ asarray(IV))) @ asarray(IV.T) @ asarray(DV)

    except:
        betas = linalg.pinv((asarray(IV.T) @ asarray(IV))) @ asarray(IV.T) @ asarray(DV)

    return betas



def lstsq_estimation_principal(IV: ndarray, DV: ndarray) -> ndarray:
    """
    Perform Least-Squares (LSTSQ) regression using numpy.linalg.lstsq to estimate coefficients, with fallback
    to pseudo-inverse if necessary.

    Parameters
    ----------
    IV : ndarray
        The independent variable(s) design matrix.
    DV : ndarray
        The dependent variable response vector.

    Returns
    -------
    ndarray
        Estimated coefficients (betas) for the regression model.
    """
    # Calculate the LSTSQ coefficients using numpy.linalg.lstsq, with fallback to normal equation using
    # pseudo-inverse if necessary.
    try:
        betas, _, _, _ = linalg.lstsq((asarray(IV.T) @ asarray(IV)), asarray(IV))

    except:
        betas = linalg.pinv((asarray(IV.T) @ asarray(IV))) @ asarray(IV.T) @ asarray(DV)

    return betas



def mle_estimation_principal(fun: object, x0: object, args: object = (), jac: object = None, method: object = None,
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
