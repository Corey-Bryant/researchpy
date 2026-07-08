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

# These will be removed once refactoring is complete as they were never published
from .model import *
from .ols import ols
from .anova import anova


# New modular structure imports
#from .core.containerclasses import FitStatistics, ModelResults, TestResults
from .core.data_utils import as_array, validate_array

from .models import CoreModel, GeneralModel
from .models.multivariable import Regress, LinearRegression, LM, Anova, ANOVA, LogisticRegression, Logistic

# Summaries subpackage
from .descriptive import summarize as summarize_new
from . import descriptive

# Statistical summary subpackage
#from .statistics import estable
from .statistics import *

# Dataset and Dataset Importers
from . import datasets
