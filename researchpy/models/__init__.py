# -*- coding: utf-8 -*-
"""
Researchpy Models Module

This module provides the core model classes for regression analysis in Researchpy.
The module is organized into submodules for different types of models:

- Core model classes (CoreModel, GeneralModel) - base classes for building regression models
- Multivariable regression models (Logistic, etc.) - specific regression implementations
- Postestimation methods (likelihood ratio tests, etc.) - tools for model evaluation

Usage:
    from researchpy.models import CoreModel, GeneralModel
    from researchpy.models.multivariable import Logistic

"""


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency during package initialization."""
    if name == "CoreModel":
        from researchpy.models.base import CoreModel
        return CoreModel
    if name == "GeneralModel":
        from researchpy.models.general_model import GeneralModel
        return GeneralModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define what gets exported with "from researchpy.models import *"
__all__ = [
    "CoreModel",
    "GeneralModel",
]

