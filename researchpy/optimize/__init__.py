# -*- coding: utf-8 -*-
"""
Researchpy Optimization Module

This module provides the optimization algorithms (a.k.a. estimate principal or estimators) used in ResearchPy for fitting
statistical models. It includes numeric estimation using ordinary least squares, and various optimization algorithms,
such as Newton-Raphson and SciPy's minimize function, as well as a tracker class for monitoring the optimization process.

Usage:
    >>> import researchpy.optimize
    >>> from researchpy.optimize.objective_functions import neg_log_likelihood

"""
from researchpy.optimize.objective_functions import (
    neg_log_likelihood,
    gradient_neg_log_likelihood,
)
from researchpy.optimize.algorithms import (
    IRLS,
)
from researchpy.optimize.solvers import (
    ols_estimation_principal, lstsq_estimation_principal, mle_estimation_principal,
)
from researchpy.optimize._optimize import OptimizationTracker, ModelWarning
from researchpy.optimize.diagnostics import (
    check_conditioning, ConditioningResult, CONDITION_NUMBER_THRESHOLD,
)

__all__ = [
    "neg_log_likelihood",
    "gradient_neg_log_likelihood",
    "ols_estimation_principal",
    "lstsq_estimation_principal",
    "mle_estimation_principal",
    "IRLS",
    "ModelWarning",
    "OptimizationTracker",
    "check_conditioning",
    "ConditioningResult",
    "CONDITION_NUMBER_THRESHOLD",
]
