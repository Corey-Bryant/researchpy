import pytest
import numpy as np
from researchpy.basic_stats import count, nanvar, nanstd

def test_count():
    # Test counting non-missing observations
    data = np.array([1, 2, 3, np.nan, 4])
    assert count(data) == 4

def test_nanvar():
    # Test variance calculation with missing data
    data = np.array([1, 2, 3, np.nan, 4])
    assert np.isclose(nanvar(data), 1.6667, atol=1e-4)

def test_nanstd():
    # Test standard deviation calculation with missing data
    data = np.array([1, 2, 3, np.nan, 4])
    assert np.isclose(nanstd(data), 1.29099, atol=1e-4)