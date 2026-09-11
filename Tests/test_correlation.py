import pytest
import pandas as pd
from researchpy.correlation import corr_case, corr_pair

def test_corr_case():
    # Create a sample dataframe
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)

    # Call the function
    info, r_vals, p_vals = corr_case(df)

    # Assertions
    assert info.iloc[0, 0] == "Total observations used = 5"
    assert round(r_vals.loc['A', 'B'], 4) == -1.0
    assert round(r_vals.loc['A', 'C'], 4) > 0.9
    assert float(p_vals.loc['A', 'B']) < 0.05

def test_corr_pair():
    # Create a sample dataframe
    data = {
        'X': [1, 2, 3, 4, 5],
        'Y': [5, 4, 3, 2, 1],
        'Z': [2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)

    # Call the function
    results = corr_pair(df)

    # Assertions
    assert results.loc['X & Y', 'r value'] == "-1.0000"
    assert results.loc['X & Z', 'r value'] > "0.9000"
    assert results.loc['X & Y', 'p-value'] < "0.0500"
