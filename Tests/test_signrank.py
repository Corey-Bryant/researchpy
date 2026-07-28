import pytest
import pandas as pd
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

    # Conduct the test
    results = test.conduct()

    # Assertions
    assert "z-statistic" in results
    assert "p-value" in results
