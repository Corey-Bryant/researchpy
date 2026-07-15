import pandas as pd
from researchpy.model import model as core_model

def test_core_model_initialization():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the core_model
    model = core_model("Y ~ X", data)

    # Assertions
    assert model.nobs == 5  # Number of observations
    assert model.n == 5  # Number of rows in IV
    assert model.k == 2  # Number of columns in IV (including intercept)
    assert model.DV_name == "Y"  # Dependent variable name
    assert model.formula == "Y ~ X"  # Formula used for the model

def test_core_model_design_matrix():
    # Create sample data
    data = pd.DataFrame({
        'X': [1, 2, 3, 4, 5],
        'Y': [2, 4, 6, 8, 10]
    })

    # Initialize the core_model object without intercept
    model = core_model('Y ~ X', data, matrix_type=0)

    # Assertions
    assert model.IV.shape[1] == 1  # Design matrix should have 1 column (no intercept)
    assert model.DV_name == 'Y'  # Dependent variable name
