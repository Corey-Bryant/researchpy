import pytest
import pandas as pd
from researchpy.models.multivariable import logistic

def test_logistic_initialization():
    # Create sample data
    data = pd.DataFrame({
        "X": [0, 1, 0, 1],
        "Y": [0, 1, 0, 1]
    })

    # Initialize the logistic object
    model = logistic("Y ~ X", data)

    # Assertions
    assert model.__name__ == "researchpy.Logistic"
    assert model.formula == "Y ~ X"

def test_logistic_model_data():
    # Create sample data
    data = pd.DataFrame({
        "X": [0, 1, 0, 1],
        "Y": [0, 1, 0, 1]
    })

    # Initialize the logistic object
    model = logistic("Y ~ X", data)

    # Assertions
    assert "betas" in model.model_data
    assert len(model.model_data["betas"]) == 2  # Intercept and slope
