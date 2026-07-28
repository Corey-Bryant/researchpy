import pytest
import pandas as pd
from researchpy.difference_test import difference_test

def test_difference_test_independent_ttest():
    # Create sample data
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 12, 14, 16]
    })

    # Initialize the difference_test object
    test = difference_test('value ~ group', data, equal_variances=True, independent_samples=True)

    # Conduct the test
    summary, results = test.conduct()

    # Assertions
    assert test.parameters['Test name'] == "Independent samples t-test"
    assert summary.iloc[0, 1] == 2  # Group A count
    assert results.iloc[0, 1] > 0  # t-statistic should be positive

def test_difference_test_paired_ttest():
    # Create sample data
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 12, 10, 12]
    })

    # Initialize the difference_test object
    test = difference_test('value ~ group', data, equal_variances=True, independent_samples=False)

    # Conduct the test
    summary, results = test.conduct()

    # Assertions
    assert test.parameters['Test name'] == "Paired samples t-test"
    assert results.iloc[0, 1] == 0  # t-statistic should be zero for identical values

def test_difference_test_independent():
    # Create sample data
    data = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "value": [1, 2, 3, 4]
    })

    # Call the function
    model = difference_test("value ~ group", data, equal_variances=True, independent_samples=True)
    result = model.conduct()

    # Assertions
    assert result["Test name"] == "Independent samples t-test"

def test_difference_test_paired():
    # Create sample data
    data = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "value": [1, 2, 1, 2]
    })

    # Call the function
    model = difference_test("value ~ group", data, equal_variances=True, independent_samples=False)
    result = model.conduct()

    # Assertions
    assert result["Test name"] == "Paired samples t-test"
