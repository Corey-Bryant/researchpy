# -*- coding: utf-8 -*-
"""
Researchpy Postestimation Methods

This submodule contains postestimation methods and diagnostics for fitted models including:

- LikelihoodRatioTest: Likelihood ratio test for model comparison (alias: LRTest)

Future methods to be added:
- Wald tests
- Score tests
- Goodness-of-fit tests
- Residual diagnostics
- Influence measures
- etc.

Usage:
    from researchpy.models.postestimation import LikelihoodRatioTest
    # or use the alias
    from researchpy.models.postestimation import LRTest

    # Compare fitted model to null (intercept-only)
    lr_test = LikelihoodRatioTest(fitted_model)
    lr_test.results()

    # Compare two nested models
    lr_test = LikelihoodRatioTest(full_model, restricted_model=reduced_model)
    lr_test.summary()

"""

from researchpy.models.postestimation.likelihood_ratio import LikelihoodRatioTest, LRTest
from researchpy.models.postestimation.predict import (
    predict, leverage, studentized_residuals, standardized_residuals,
    _compute_hat_matrix, residuals, predict_y
)


# Define what gets exported with "from researchpy.models.postestimation import *"
__all__ = [
    "LikelihoodRatioTest",
    "LRTest",
    "predict",
    "leverage",
    "studentized_residuals",
    "standardized_residuals",
    "_compute_hat_matrix",
    "residuals",
    "predict_y"
]

