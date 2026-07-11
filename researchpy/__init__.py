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
from .model import *
from .ols import ols
from .anova import anova


# New modular structure imports
from .core.data_utils import as_array, validate_array

from .models import CoreModel, GeneralModel
from .models.multivariable import Regress, LinearRegression, LM, Anova, ANOVA, LogisticRegression, Logistic

# Statistical summary subpackage
from .statistics import *
