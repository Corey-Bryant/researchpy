# -*- coding: utf-8 -*-

from researchpy.version import __version__
from researchpy.utility import *
from researchpy.ttest import ttest
from researchpy.summary import *
from researchpy.correlation import *
from researchpy.crosstab import *
from researchpy.difference_test import *
from researchpy.basic_stats import (
    count, nanvar, nanstd, nansem, value_range,
    kurtosis, skew, confidence_interval, l_ci, u_ci,
)
from researchpy.signrank import *
from researchpy.predict import *

# -- Dataset and Dataset Importers
from researchpy.datasets import (
    fetch_dta, auto, census, citytemp, cancer, lifeexp,
    nlsw88, sp500, systolic, lbw, uslifeexp, voter,
)

# -- These are deprecated and will be removed in future versions
from researchpy.model import model
from researchpy.ols import ols
from researchpy.anova import anova

# -- New modular structure imports
from researchpy.core import as_array, validate_array

from researchpy.models import (
    BaseModel,
    Regress, LinearRegression, LM,
    Anova, ANOVA,
    Logistic, LogisticRegression,
)

# -- Statistical summary subpackage
#from researchpy.statistics import *


# --
__all__ = [
    # Version
    "__version__",

    # Utility
    "rounder", "return_numeric", "as_numeric",
    "patsy_column_cleaner", "patsy_term_cleaner",
    "variable_information", "base_table",

    # T-tests
    "ttest",

    # Summary
    "summary_cont", "summary_cat", "codebook", "summarize",

    # Correlation
    "corr_case", "corr_pair",

    # Crosstab
    "crosstab",

    # Difference test
    "difference_test",

    # Basic stats
    "count", "nanvar", "nanstd", "nansem", "value_range",
    "kurtosis", "skew", "confidence_interval", "l_ci", "u_ci",

    # Sign rank
    "signrank",

    # Predict
    "predict_y", "residuals", "standardized_residuals",
    "studentized_residuals", "leverage", "predict",

    # Datasets
    "fetch_dta", "auto", "census", "citytemp", "cancer", "lifeexp",
    "nlsw88", "sp500", "systolic", "lbw", "uslifeexp", "voter",

    # Legacy (deprecated)
    "model", "ols", "anova",

    # Core
    "as_array", "validate_array",

    # Models
    "BaseModel",
    "Regress", "LinearRegression", "LM",
    "Anova", "ANOVA",
    "Logistic", "LogisticRegression",

    # Statistics subpackage
    #"_get_distribution",
    #"_confidence_interval"
    #"_estimate_confidence_interval",
    #"_compute_pvalue",
]
