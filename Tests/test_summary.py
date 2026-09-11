import pytest
import pandas as pd
import numpy as np
from researchpy.summary import summary_cont

def test_summary_cont_series():
    # Create sample data
    data = pd.Series([1, 2, 3, 4, 5], name="Sample")

    # Call the function
    result = summary_cont(data)

    # Assertions
    assert result.iloc[0, 0] == "Sample"  # Variable name
    assert result.iloc[0, 1] == 5  # Count
    assert result.iloc[0, 2] == 3  # Mean
    assert round(result.iloc[0, 3], 2) == 1.58  # Standard deviation

def test_summary_cont_dataframe():
    # Create sample data
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [6, 7, 8, 9, 10]
    })

    # Call the function
    result = summary_cont(data)

    # Assertions
    assert result.iloc[0, 0] == "A"  # First variable name
    assert result.iloc[0, 1] == 5  # Count for first variable
    assert result.iloc[0, 2] == 3  # Mean for first variable
    assert result.iloc[1, 0] == "B"  # Second variable name
    assert result.iloc[1, 1] == 5  # Count for second variable
    assert result.iloc[1, 2] == 8  # Mean for second variable
