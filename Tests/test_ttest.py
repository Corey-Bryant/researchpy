import pytest
import pandas as pd
import scipy.stats as st
from researchpy.ttest import ttest


def test_ttest_independent():
    # Create sample data
    group1 = pd.Series([1, 2, 3, 4, 5])
    group2 = pd.Series([6, 7, 8, 9, 10])

    # ttest returns a tuple: (summary_table, results_table)
    summary, results = ttest(group1, group2, equal_variances=True, paired=False)

    # The results table's first column is named after the test; rows are
    # ``label -> value`` pairs in columns 0 and 1.
    assert results.columns[0] == "Independent t-test"

    stats = {str(k).strip(): v for k, v in zip(results.iloc[:, 0], results.iloc[:, 1])}

    # Validate against SciPy (researchpy uses scipy.stats.ttest_ind internally)
    t_ref, p_ref = st.ttest_ind(group1, group2)
    assert round(stats["t ="], 4) == round(float(t_ref), 4)
    assert round(stats["Two side test p value ="], 4) == round(float(p_ref), 4)


def test_ttest_paired():
    # Use non-degenerate paired data (identical groups yield an undefined
    # t = 0/0 = NaN because the differences have zero variance).
    group1 = pd.Series([1, 2, 3, 4, 5])
    group2 = pd.Series([2, 4, 5, 7, 8])

    # ttest returns a tuple: (summary_table, results_table)
    summary, results = ttest(group1, group2, equal_variances=True, paired=True)

    assert results.columns[0] == "Paired samples t-test"

    stats = {str(k).strip(): v for k, v in zip(results.iloc[:, 0], results.iloc[:, 1])}

    # Validate against SciPy (researchpy uses scipy.stats.ttest_rel internally)
    t_ref, p_ref = st.ttest_rel(group1, group2)
    assert round(stats["t ="], 4) == round(float(t_ref), 4)
    assert round(stats["Two side test p value ="], 4) == round(float(p_ref), 4)
