# -*- coding: utf-8 -*-

from .version import __version__
from .utility import *
from .ttest import ttest
from .summary import *
from .correlation import *
from .crosstab import *
from .difference_test import *
from .basic_stats import *
from .signrank import *
from .predict import *

# Dataset and Dataset Importers
from . import datasets

# These are deprecated and will be removed in future versions
#from .model import model
#from .ols import ols
#from .anova import anova
from researchpy.model import model
from researchpy.ols import ols
from researchpy.anova import anova


# New modular structure imports
from researchpy.core import as_array, validate_array

from researchpy.models import (
    BaseModel,
    Regress, LinearRegression, LM, Anova, ANOVA,
    GLM, GeneralizedLinearModel, Logistic, LogisticRegression,
)

# Statistical summary subpackage
from researchpy.statistics import *
