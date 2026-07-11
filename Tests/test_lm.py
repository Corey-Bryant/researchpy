import pytest
import pandas as pd
#from researchpy.MultivariableRegression.lm import lm
from researchpy.models.multivariable import LM as lm

def test_lm_initialization():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the lm object
    model = lm("Y ~ X", data)

    # Assertions
    assert model.__name__ == "researchpy.lm"
    assert model.formula == "Y ~ X"

def test_lm_model_data():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the lm object
    model = lm("Y ~ X", data)

    # Assertions
    assert "sum_of_square_total" in model.model_data
    assert "sum_of_square_model" in model.model_data
    assert "sum_of_square_residual" in model.model_data
