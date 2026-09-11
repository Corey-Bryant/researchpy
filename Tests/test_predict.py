import pytest
import numpy as np
from researchpy.predict import predict_y, residuals, standardized_residuals, studentized_residuals, leverage, predict

class MockModelData:
    def __init__(self):
        self.IV = np.array([[1, 2], [1, 3], [1, 4]])
        self.DV = np.array([[5], [7], [9]])
        self.model_data = {
            "betas": np.array([[1], [2]]),
            "mse": 1,
            "H": np.array([[0.5, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        }
        self.nobs = 3
        self._IV_design_info = type("DesignInfo", (), {"column_names": ["Intercept", "X"]})

mock_data = MockModelData()

def test_predict_y():
    result = predict_y(mock_data)
    expected = np.array([[5], [7], [9]])
    assert np.allclose(result, expected)

def test_residuals():
    result = residuals(mock_data)
    expected = np.array([[0], [0], [0]])
    assert np.allclose(result, expected)

def test_standardized_residuals():
    result = standardized_residuals(mock_data)
    assert result.shape == (3, 1)

def test_studentized_residuals():
    result = studentized_residuals(mock_data)
    assert result.shape == (3, 1)

def test_leverage():
    result = leverage(mock_data)
    expected = np.array([[0.5], [0.6], [0.7]])
    assert np.allclose(result, expected)

def test_predict():
    result = predict(mock_data, estimate="y")
    expected = np.array([[5], [7], [9]])
    assert np.allclose(result, expected)
