# -*- coding: utf-8 -*-
"""
Researchpy Model Module

This module provides the base model class for Researchpy regression models.

DEPRECATION NOTICE:
    The `model` class in this module is maintained for backward compatibility with v0.3.7.
    It will be deprecated in a future version. New code should use:

        from researchpy.models import CoreModel, GeneralModel
        from researchpy.models.multivariable import LogisticRegression
        # or use aliases
        from researchpy.models.multivariable import Logistic

    The new modular structure provides better organization and more functionality.
"""
import warnings
import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .utility import *


class model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.
    .. deprecated::
        The `model` class is deprecated and will be removed in a future version.
        Please use `researchpy.models.CoreModel` or `researchpy.models.GeneralModel` instead.
        Example migration:
            # Old way
            from researchpy.model import model
            # New way
            from researchpy.models import CoreModel
    """


    def __init__(self, formula_like, data = {}, matrix_type = 1):
        # Issue deprecation warning
        warnings.warn(
            "The 'model' class is deprecated and will be removed in a future version. "
            "Please use 'researchpy.models.CoreModel' or 'researchpy.models.GeneralModel' instead. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        # matrix_type = 1 includes intercept
        # matrix_type = 0 does not include the intercept



        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        self.nobs = self.IV.shape[0]

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information  = variable_information(self.IV.design_info.term_names, self.IV.design_info.column_names, data)
        
