import pytest
import pandas as pd
from researchpy.ttest import ttest

def test_ttest_independent():
    # Create sample data
    group1 = pd.Series([1, 2, 3, 4, 5])
    group2 = pd.Series([6, 7, 8, 9, 10])

    # Call the function
    result = ttest(group1, group2, equal_variances=True, paired=False)

    # Assertions
    assert result["Test"] == "Independent t-test"
    assert round(result["t-value"], 2) == -5.48  # t-value
    assert round(result["p-value"], 3) == 0.001  # p-value

def test_ttest_paired():
    # Create sample data
    group1 = pd.Series([1, 2, 3, 4, 5])
    group2 = pd.Series([1, 2, 3, 4, 5])

    # Call the function
    result = ttest(group1, group2, equal_variances=True, paired=True)

    # Assertions
    assert result["Test"] == "Paired t-test"
    assert result["t-value"] == 0  # t-value should be zero for identical groups
    assert result["p-value"] == 1  # p-value should be 1 for identical groups
