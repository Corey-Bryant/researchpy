# -*- coding: utf-8 -*-
"""
Researchpy Models Module

This module provides the code model classes for regression analysis in Researchpy. The module is organized into
submodules for different types of models:

- BaseModel class is the code model that all other models inherit from. This class utilizes the matrix engine from
    researchpy.core.DesignMatrix and provides the foundation for all regression models in Researchpy.
- Linear regression models are implemented in the researchpy.models.linear submodule (e.g., Regress (LM, LinearModel), Anova (ANOVA))
    using the ordinary least squares algorithms (a.k.a. estimator, or estimation principal).
- Generalized linear regression models are implemented in the researchpy.models.generalized submodule (e.g., GeneralizedLinearModel (GLM), LogisticRegression (Logistic))
    using the maximum likelihood solver (a.k.a. estimator, or estimation principal).

"""


from researchpy.models.base import BaseModel
from researchpy.models.linear import (
    Regress, LinearRegression, LM, Anova, ANOVA,
)
from researchpy.models.generalized import (
    GLM, GeneralizedLinearModel, LogisticRegression, Logistic,
)

# Define what gets exported with "from researchpy.models import *"
__all__ = [
    # The base model
    "BaseModel",
    # Family / link abstractions
    #"Family",
    #"BinomialFamily",
    #"PoissonFamily",
    #"GaussianFamily",
    #"get_family",
    # Linear regression models and aliases
    "Regress",
    "LinearRegression",
    "LM",
    "Anova",
    "ANOVA",
    # Generalized linear regression models and aliases
    "GLM",
    "GeneralizedLinearModel",
    "LogisticRegression",
    "Logistic",
]

