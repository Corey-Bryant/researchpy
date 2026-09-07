import pytest
import pandas as pd
import scipy.stats as st
from researchpy.difference_test import difference_test

def test_difference_test_independent_ttest():
    # Create sample data
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 12, 14, 16]
    })

    # Initialize the difference_test object
    test = difference_test('value ~ group', data, equal_variances=True, independent_samples=True)

    # conduct() returns a tuple: (summary_table, results_table)
    summary, results = test.conduct()

    # Assertions
    assert test.parameters['Test name'] == "Independent samples t-test"
    assert summary.iloc[0, 1] == 2  # Group A count

    # results table: column 0 holds labels, column 1 holds values.
    # Row 0 is the (A - B) mean difference; the t-statistic is on the "t =" row.
    stats = {str(k).strip(): v for k, v in zip(results.iloc[:, 0], results.iloc[:, 1])}
    t_ref, _ = st.ttest_ind([10, 12], [14, 16])
    assert round(stats["t ="], 4) == round(float(t_ref), 4)  # negative: mean(A) < mean(B)

def test_difference_test_paired_ttest():
    # Create sample data
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 12, 10, 12]
    })

    # Initialize the difference_test object
    test = difference_test('value ~ group', data, equal_variances=True, independent_samples=False)

    # conduct() returns a tuple: (summary_table, results_table)
    summary, results = test.conduct()

    # Assertions
    assert test.parameters['Test name'] == "Paired samples t-test"
    # Row 0, column 1 is the (A - B) mean difference; identical values -> 0
    assert results.iloc[0, 1] == 0

def test_difference_test_independent():
    # Create sample data
    data = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "value": [1, 2, 3, 4]
    })

    # The test name is stored on the object's parameters; conduct() returns a tuple.
    model = difference_test("value ~ group", data, equal_variances=True, independent_samples=True)
    summary, results = model.conduct()

    # Assertions
    assert model.parameters["Test name"] == "Independent samples t-test"

def test_difference_test_paired():
    # Create sample data
    data = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "value": [1, 2, 1, 2]
    })

    # The test name is stored on the object's parameters; conduct() returns a tuple.
    model = difference_test("value ~ group", data, equal_variances=True, independent_samples=False)
    summary, results = model.conduct()

    # Assertions
    assert model.parameters["Test name"] == "Paired samples t-test"
