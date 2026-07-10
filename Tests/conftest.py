"""
Shared test fixtures for ResearchPy golden-value tests.

Datasets are loaded from local dictionaries in stata_datasets.py,
so tests run without any network dependency.

All fixtures are session-scoped: DataFrames are created once and reused
across the entire test session.
"""
import sys
from pathlib import Path

import pytest
import pandas as pd

# Ensure project root is on sys.path so 'Tests.Golden' resolves regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Tests.Golden.stata_datasets import auto as _auto_dict, systolic as _systolic_dict, lbw as _lbw_dict


# ═══════════════════════════════════════════════════════════════════════════
# DATAFRAME FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def auto_df():
    """Full Stata 'auto' DataFrame (74 obs x 12 vars)."""
    return pd.DataFrame(_auto_dict)


@pytest.fixture(scope="session")
def systolic_df():
    """Full Stata 'systolic' DataFrame (58 obs x 3 vars)."""
    return pd.DataFrame(_systolic_dict)


@pytest.fixture(scope="session")
def lbw_df():
    """Full Stata 'lbw' DataFrame (189 obs x 11 vars)."""
    return pd.DataFrame(_lbw_dict)


# ═══════════════════════════════════════════════════════════════════════════
# AUTO: SERIES / SUBSET FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def auto_price(auto_df):
    """auto['price'] as a pandas Series."""
    return auto_df["price"]


@pytest.fixture(scope="session")
def auto_mpg(auto_df):
    """auto['mpg'] as a pandas Series."""
    return auto_df["mpg"]


@pytest.fixture(scope="session")
def auto_price_mpg(auto_df):
    """auto[['price', 'mpg']] as a pandas DataFrame."""
    return auto_df[["price", "mpg"]]


@pytest.fixture(scope="session")
def auto_groupby_foreign_price(auto_df):
    """auto.groupby('foreign')['price'] - Series Groupby object."""
    return auto_df.groupby("foreign")["price"]


@pytest.fixture(scope="session")
def auto_groupby_foreign_price_mpg(auto_df):
    """auto.groupby('foreign')[['price', 'mpg']] - DataFrame Groupby object."""
    return auto_df.groupby("foreign")[["price", "mpg"]]


# ═══════════════════════════════════════════════════════════════════════════
# SYSTOLIC: SERIES / SUBSET FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def systolic_series(systolic_df):
    """systolic['systolic'] as a pandas Series."""
    return systolic_df["systolic"]


@pytest.fixture(scope="session")
def systolic_disease(systolic_df):
    """systolic['disease'] as a pandas Series."""
    return systolic_df["disease"]


@pytest.fixture(scope="session")
def systolic_drug(systolic_df):
    """systolic['drug'] as a pandas Series."""
    return systolic_df["drug"]