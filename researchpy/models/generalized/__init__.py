# -*- coding: utf-8 -*-
"""
Researchpy General Regression Models

This submodule contains implementations of generalized regression models including:

- GeneralizedLinearModel: Base class for generalized linear regression models
  (alias: GLM)
- LogisticRegression: Logistic regression for binary outcomes using MLE
  (alias: Logistic)

Future models to be added:
- PoissonRegression: Poisson regression for count data (alias: Poisson)

Usage:
    >>> import researchpy.models.generalized
    >>> from researchpy.models.generalized import *

    >>> from researchpy.models.generalized import GeneralizedLinearModel, LogisticRegression
    # or use the aliases
    >>> from researchpy.models.generalized import GLM, Logistic

    >>> # General regression model
    >>> general_model = GLM("y ~ x1 + x2", data=df, family='binomial', link='logit')
    >>> general_results = general_model.results()

    >>> # Logistic regression
    >>> logit_model = LogisticRegression("outcome ~ predictor1 + predictor2", data=df)
    >>> logit_results = logit_model.results()

"""

from researchpy.models.generalized.general_model import GeneralizedLinearModel, GLM
from researchpy.models.generalized.logistic import LogisticRegression, Logistic

# Define what gets exported with "from researchpy.models.generalized import *"
__all__ = [
    # General model's aliases
    "GeneralizedLinearModel",
    "GLM",
    # Logistic and alias
    "LogisticRegression",
    "Logistic",
]
