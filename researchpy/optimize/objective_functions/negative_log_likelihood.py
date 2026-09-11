# -*- coding: utf-8 -*-
"""
Shared Likelihood Objective Functions

Likelihood objective functions for optimization routines in ResearchPy. These are designed to be used with
scipy.optimize for fitting statistical models, particularly generalized linear models (GLMs). The functions
delegate core likelihood/gradient math to ``Family`` instances from ``researchpy.models.families``, while
adding regularization support and optional optimization tracking.

Usage:

    from researchpy.models.families import get_family
    from researchpy.optimize.objective_functions import neg_log_likelihood, gradient_neg_log_likelihood

    family = get_family("binomial")
    nll = neg_log_likelihood(params, IV, DV, solver_options, family=family)

"""
import numpy as np

from researchpy.models.families import Family
from researchpy.containers import SolverOptions


def neg_log_likelihood(params: object,
                       IV: object,
                       DV: object,
                       solver_options: SolverOptions,
                       family: Family,
                       tracker: object = None
                       ):
    """Negative log-likelihood function for scipy.optimize.

    Delegates the core likelihood computation to the provided ``Family``
    instance, then applies optional regularization and tracking.

    Parameters
    ----------
    params : array-like
        Current parameter estimates.
    IV : array-like
        Independent variable (design) matrix.
    DV : array-like
        Dependent variable vector.
    solver_options : SolverOptions object
        A SolverOptions dataclass instance containing regularization and display settings.
    family : Family
        A ``researchpy.models.families.Family`` instance that provides ``link_inverse`` and ``log_likelihood`` methods.
    tracker : OptimizationTracker object, or None
        Optional tracker for monitoring optimization progress.

    Returns
    -------
    float
        The negative log-likelihood value.

    Raises
    ------
    ValueError
        If required parameters are not provided or of wrong type.
    """
    if family is None:
        raise ValueError(
            "A Family instance is required. Use get_family() to resolve from a string, "
            "e.g. family=get_family('binomial')."
        )

    params = np.atleast_2d(params).T  # Ensure params is a column vector
    linear_pred = IV @ params

    # Compute fitted values via family inverse link
    mu = family.link_inverse(linear_pred)

    # Compute negative log-likelihood via family
    ll = -family.log_likelihood(DV, mu)

    # Add regularization if specified
    if hasattr(solver_options, "regularization"):
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



def gradient_neg_log_likelihood(params, IV, DV,
                                solver_options, family: Family = None):
    """Gradient of negative log-likelihood.

    Computes the score (gradient) using the canonical GLM formula.
    For canonical links, the gradient simplifies to -X'(y - μ).

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
    family : Family
        A ``researchpy.models.families.Family`` instance that provides
        ``link_inverse`` method.

    Returns
    -------
    ndarray
        Flattened gradient vector.

    Raises
    ------
    ValueError
        If ``family`` is not provided.
    """
    if family is None:
        raise ValueError(
            "A Family instance is required. Use get_family() to resolve from a string, "
            "e.g. family=get_family('binomial')."
        )

    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = IV @ params

    # Compute fitted values via family inverse link
    mu = family.link_inverse(linear_pred)

    # Gradient for canonical link: -X'(y - mu)
    grad = -IV.T @ (DV - mu)

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
