import numpy as np
from scipy.special import expit

def log_likelihood(y_e):

    return np.sum(y_e) - np.sum(np.log((1 + y_e)))


def neg_log_likelihood(params, IV, DV, solver_options, distribution_family="binomial", link_function="logit", tracker=None):
    """Negative log-likelihood function for scipy.optimize.

    Parameters
    ----------
    params : array-like
        Current parameter estimates.
    IV : array-like
        Independent variable (design) matrix.
    DV : array-like
        Dependent variable vector.
    solver_options : SolverOptions
        A SolverOptions dataclass instance containing regularization and display settings.
    distribution_family : str
        Distribution family (e.g., "binomial").
    link_function : str
        Link function (e.g., "logit").
    tracker : OptimizationTracker or None
        Optional tracker for monitoring optimization progress.

    Returns
    -------
    float
        The negative log-likelihood value.
    """
    params = np.atleast_2d(params).T  # Ensure params is a column vector
    linear_pred = IV @ params

    # Apply the link function
    if link_function == "logit":
        p = expit(linear_pred)  # Numerically stable sigmoid
    else:
        raise NotImplementedError(f"Link function '{link_function}' is not implemented.")

    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    # Compute log-likelihood based on the distribution family
    if distribution_family == "binomial":
        ll = -np.sum(DV * np.log(p) + (1 - DV) * np.log(1 - p))
    else:
        raise NotImplementedError(f"Distribution family '{distribution_family}' is not implemented.")

    # Add regularization if specified
    if solver_options.regularization == "l2":
        # Don't regularize intercept (first coefficient)
        ll += solver_options.alpha * np.sum(params[1:] ** 2)

    elif solver_options.regularization == "l1":
        ll += solver_options.alpha * np.sum(np.abs(params[1:]))

    # Store the log-likelihood value in the tracker if provided
    if tracker is not None:
        tracker.current_cost = ll
        tracker.current_cost_index += 1

        if solver_options.display:
            print(f"Log-likelihood = {-ll:.5f}")

    return ll



def gradient_neg_log_likelihood(params, IV, DV, solver_options, distribution_family="binomial", link_function="logit"):
    """Gradient of negative log-likelihood.

    Parameters
    ----------
    params : array-like
        Current parameter estimates.
    IV : array-like
        Independent variable (design) matrix.
    DV : array-like
        Dependent variable vector.
    solver_options : SolverOptions
        A SolverOptions dataclass instance containing regularization settings.
    distribution_family : str
        Distribution family (e.g., "binomial").
    link_function : str
        Link function (e.g., "logit").

    Returns
    -------
    ndarray
        Flattened gradient vector.
    """
    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = IV @ params

    # Apply the link function
    if link_function == "logit":
        p = expit(linear_pred)
    else:
        raise NotImplementedError(f"Link function '{link_function}' is not implemented.")

    # Compute gradient based on the distribution family
    if distribution_family == "binomial":
        grad = -IV.T @ (DV - p)
    else:
        raise NotImplementedError(f"Distribution family '{distribution_family}' is not implemented.")

    # Add regularization gradient if specified
    if solver_options.regularization == "l2":
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = 2 * solver_options.alpha * params[1:]  # Don't regularize intercept
        grad += reg_grad

    elif solver_options.regularization == "l1":
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = solver_options.alpha * np.sign(params[1:])
        grad += reg_grad

    return grad.flatten()  # Return flattened gradient for scipy.optimize



