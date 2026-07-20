# -*- coding: utf-8 -*-
"""
Researchpy Optimization Module

This module provides the optimization solvers (a.k.a. estimate principal or estimators) used in ResearchPy for fitting
statistical models. It includes numeric estimation using ordinary least squares, and various optimization algorithms,
such as Newton-Raphson and SciPy's minimize function, as well as a tracker class for monitoring the optimization process.

Usage:
    >>> import researchpy.optimization
    >>> from researchpy.optimization.objective_functions import neg_log_likelihood

"""


from researchpy.optimization.objective_functions import (
    neg_log_likelihood,
    gradient_neg_log_likelihood
)

from researchpy.optimization.solvers import (
    ordinary_least_squares,
    scipy_minimize,
    newton_raphson,
)

from researchpy.optimization.trackers import OptimizationTracker


__all__ = [
    "neg_log_likelihood",
    "gradient_neg_log_likelihood",
    "ordinary_least_squares",
    "scipy_minimize",
    "newton_raphson",
    "OptimizationTracker",
]
