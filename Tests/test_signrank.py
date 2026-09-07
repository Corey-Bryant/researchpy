import pytest
import pandas as pd
import scipy.stats as st
from researchpy.signrank import signrank

def test_signrank_initialization():
    # Create sample data
    group1 = [1, 2, 3, 4, 5]
    group2 = [5, 4, 3, 2, 1]

    # Initialize the signrank object
    test = signrank(group1=group1, group2=group2)

    # Assertions
    assert test.group1 == group1
    assert test.group2 == group2

def test_signrank_conduct():
    # Create sample data
    group1 = [1, 2, 3, 4, 5]
    group2 = [5, 4, 3, 2, 1]

    # Initialize the signrank object
    test = signrank(group1=group1, group2=group2)

    # conduct() returns a tuple: (descriptives, variance, results)
    descriptives, variance, results = test.conduct(return_type="Dictionary")

    # The results component holds the z-statistic, w-statistic, and p-value
    assert "z" in results
    assert "w" in results
    assert "pval" in results

    # Symmetric data -> z-statistic of 0 and p-value of 1
    assert round(results["z"], 4) == 0.0
    assert round(results["pval"], 4) == 1.0

    # Validate w/p against SciPy (researchpy uses scipy.stats.wilcoxon internally)
    w_ref, p_ref = st.wilcoxon(group1, group2, zero_method="pratt", correction=False)
    assert round(results["w"], 4) == round(float(w_ref), 4)
    assert round(results["pval"], 4) == round(float(p_ref), 4)

def test_signrank_formula():
    # Formula-based constructor: a 2-level factor splits the DV into groups.
    data = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B"],
        "value": [1, 2, 3, 5, 4, 3],
    })

    test = signrank(formula_like="value ~ group", data=data)

    # Groups are split by factor level (A -> group1, B -> group2)
    assert list(test.group1.ravel()) == [1, 2, 3]
    assert list(test.group2.ravel()) == [5, 4, 3]

    # conduct() returns a tuple: (descriptives, variance, results)
    descriptives, variance, results = test.conduct(return_type="Dictionary")

    assert "z" in results
    assert "w" in results
    assert "pval" in results

    # Validate w/p against SciPy on the equivalent group vectors
    w_ref, p_ref = st.wilcoxon([1, 2, 3], [5, 4, 3], zero_method="pratt", correction=False)
    assert round(results["w"], 4) == round(float(w_ref), 4)
    assert round(results["pval"], 4) == round(float(p_ref), 4)

