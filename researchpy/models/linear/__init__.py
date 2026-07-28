# -*- coding: utf-8 -*-
"""
Researchpy Linear Regression Models

This submodule contains implementations of fixed-effects linear regression models including:

- Regress: Ordinary Least Squares regression (inherits from LinearModel)
  (aliases: LinearRegression, LM)
- Anova: Analysis of Variance (inherits from LinearModel)
  (aliases: ANOVA)

Usage:
    >>> import researchpy.models.linear
    >>> from researchpy.models.linear import *

    >>> from researchpy.models.linear import Regress, Anova
    # or use the aliases
    >>> from researchpy.models.linear import LinearRegression, LM, ANOVA

    >>> # OLS regression
    >>> ols_model = Regress("y ~ x1 + x2", data=df)
    >>> ols_results = ols_model.results()

    >>> # ANOVA
    >>> anova_model = Anova("y ~ C(group)", data=df)
    >>> anova_results = anova_model.results()

"""

from researchpy.models.linear.regress import Regress, LinearRegression, LM
from researchpy.models.linear.anova import Anova, ANOVA

# Define what gets exported with "from researchpy.models.multivariable import *"
__all__ = [
    # Regress and aliases
    "Regress",
    "LinearRegression",
    "LM",
    # ANOVA and alias
    "Anova",
    "ANOVA",
]