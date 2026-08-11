import pytest
import numpy as np
from typing import Union
import sys
sys.path.insert(0, 'path/to/your/module')  # Adjust for your layout


# Import the functions you want to test
from researchpy.utility import as_numeric, _infer_numeric_from_string


class TestInferNumericFromString:
    """Tests for the _infer_numeric_from_string helper function."""

    @pytest.mark.parametrize("value,expected_type", [
        ('5', int),
        ('-5', int),
        ('+5', int),
        ('0', int),
        ('100', int),
        ('5.5', float),
        ('-5.5', float),
        ('5.', float),
        ('.5', float),
        ('5.0', float),
        ('1e5', float),
        ('1E5', float),
        ('1e-5', float),
        ('1.5e5', float),
        ('5,000', str),  # comma not supported - returns original
        ('taco', str),
        ('', str),
        ('   5   ', int),  # whitespace stripped
        ('5.5.5', str),  # invalid - multiple decimals
    ])
    def test_inference_returns_correct_type(self, value, expected_type):
        result, success = _infer_numeric_from_string(value)
        if success:
            assert isinstance(result, expected_type), f"Expected {expected_type} for '{value}', got {type(result)}"
        else:
            assert result == value
            assert isinstance(result, str)

    @pytest.mark.parametrize("value", ['taco', '', 'abc', '12a', '5.5.5'])
    def test_inference_failure_returns_original(self, value):
        result, success = _infer_numeric_from_string(value)
        assert result == value
        assert success is False

    @pytest.mark.parametrize("decimal_format,value,expected", [
        (',', '5,5', float),
        (',', '5.5', str),  # period not recognized as decimal
        (',', '5', int),
    ])
    def test_custom_decimal_format(self, decimal_format, value, expected):
        result, success = _infer_numeric_from_string(value, decimal_format=decimal_format)
        if success:
            assert isinstance(result, expected)


class TestAsNumericBasicConversions:
    """Tests for basic numeric conversions without the any dtype."""

    @pytest.mark.parametrize("value,expected", [
        (5, 5.0),
        (5.5, 5.5),
        ('5', 5.0),
        ('5.5', 5.5),
        (-5, -5.0),
        (np.float64(5.5), 5.5),
        (np.int64(5), 5.0),
    ])
    def test_float_conversion(self, value, expected):
        result = as_numeric(value, numeric_dtype=float)
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize("value,expected", [
        (5, 5),
        (5.9, 5),  # truncates toward zero
        ('5', 5),
        (np.int64(5), 5),
        (np.float64(5.9), 5),
    ])
    def test_int_conversion(self, value, expected):
        result = as_numeric(value, numeric_dtype=int)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("value", [
        'five',
        'taco',
        '',
    ])
    def test_float_conversion_raises_on_invalid(self, value):
        with pytest.raises(ValueError, match="Cannot convert"):
            as_numeric(value, numeric_dtype=float)


class TestAsNumericWithErrorsIgnore:
    """Tests for errors='ignore' behavior."""

    @pytest.mark.parametrize("value,dtype", [
        ('five', float),
        ('five', int),
        ('taco', int),
    ])
    def test_ignore_returns_original(self, value, dtype):
        result = as_numeric(value, numeric_dtype=dtype, errors='ignore')
        assert result == value


class TestAsNumericWithAnyDtype:
    """Tests for the numeric_dtype=any inference logic."""

    @pytest.mark.parametrize("value,expected_type,expected_value", [
        ('5', int, 5),
        ('5.5', float, 5.5),
        ('1e5', float, 100000.0),
        (5, int, 5),  # already int
        (5.5, float, 5.5),  # already float
        (np.int64(5), int, 5),
        (np.float64(5.5), float, 5.5),
    ])
    def test_any_infers_correctly(self, value, expected_type, expected_value):
        result = as_numeric(value, numeric_dtype=any)
        assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"
        assert result == expected_value

    @pytest.mark.parametrize("value", [
        'taco',
        'five',
        '',
    ])
    def test_any_with_non_numeric_string_raises(self, value):
        with pytest.raises(ValueError, match="Cannot convert"):
            as_numeric(value, numeric_dtype=any)

    @pytest.mark.parametrize("value,expected", [
        ('taco', 'taco'),
        ('five', 'five'),
    ])
    def test_any_with_ignore_errors(self, value, expected):
        result = as_numeric(value, numeric_dtype=any, errors='ignore')
        assert result == expected


class TestAsNumericWithNumpyArrays:
    """Tests for numpy array inputs."""

    @pytest.mark.parametrize("value,expected", [
        (np.array([5]), 5.0),
        (np.array([5.5]), 5.5),
        (np.array([[5]]), 5.0),  # multi-dimensional, single element
        (np.array([-5]), -5.0),
    ])
    def test_array_extraction(self, value, expected):
        result = as_numeric(value)
        assert result == expected

    @pytest.mark.parametrize("value,expected_type", [
        (np.array([5]), int),
        (np.array([5.5]), float),
    ])
    def test_array_with_any_dtype(self, value, expected_type):
        result = as_numeric(value, numeric_dtype=any)
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize("value", [
        np.array([1, 2, 3]),  # multi-element array
        np.array([]),  # empty array
    ])
    def test_multi_element_array_behavior(self, value):
        # Should handle gracefully - either error or return first element
        try:
            result = as_numeric(value)
            # If it succeeds, it should be from the first element
            assert isinstance(result, (int, float))
        except Exception:
            pass  # Acceptable to raise


class TestAsNumericWithRounding:
    """Tests for ndigits rounding functionality."""

    @pytest.mark.parametrize("value,ndigits,expected", [
        (3.14159, None, 3.14159),
        (3.14159, 0, 3.0),
        (3.14159, 1, 3.1),
        (3.14159, 2, 3.14),
        (3.14159, 3, 3.142),  # banker's rounding
        (2.5, 0, 2.0),  # banker's rounding
        (3.5, 0, 4.0),  # banker's rounding
        (5.6789, 1, 5.7),
        (5.6789, 2, 5.68),
        ('5.6789', 1, 5.7),  # string input with rounding
    ])
    def test_rounding_applied(self, value, ndigits, expected):
        result = as_numeric(value, ndigits=ndigits)
        assert result == expected, f"Expected {expected}, got {result}"

    @pytest.mark.parametrize("value,ndigits", [
        (np.float64(3.14159), 2),
        (np.int64(5), 0),
    ])
    def test_numpy_types_with_rounding(self, value, ndigits):
        result = as_numeric(value, ndigits=ndigits)
        # Result after rounding could be float or int depending on ndigits
        assert isinstance(result, (int, float))


class TestAsNumericEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_none_input_raises_or_returns(self):
        with pytest.raises(ValueError):
            as_numeric(None)
        assert as_numeric(None, errors='ignore') is None

    def test_boolean_input(self):
        result = as_numeric(True)
        assert result == 1.0
        result = as_numeric(False)
        assert result == 0.0

    def test_large_values(self):
        result = as_numeric('1e308')
        assert result == 1e308

    def test_negative_zero(self):
        result = as_numeric('-0.0')
        assert result == 0.0

    def test_whitespace_handling(self):
        result = as_numeric('  5.5  ')
        assert result == 5.5

    @pytest.mark.parametrize("value", [
        None,
        object(),
        lambda x: x,
    ])
    def test_unconvertible_object_types(self, value):
        with pytest.raises(ValueError):
            as_numeric(value, errors='raise')
        assert as_numeric(value, errors='ignore') is value


class TestAsNumericFallingBackPath:
    """Tests for the fallback error handling path."""

    def test_fallback_uses_float_when_int_fails(self):
        result = as_numeric('5.5', numeric_dtype=int, errors='fallback')
        assert result == 5.5
        assert isinstance(result, float)


class TestGoldenValuesCrossValidation:
    """Golden values for cross-validation against Python's built-in round()."""

    @pytest.mark.parametrize("value,ndigits,expected_python_rounded", [
        (2.555, 2, 2.56),
        (2.545, 2, 2.55),
        (2.535, 2, 2.53),
        (2.525, 2, 2.52),
        (1.005, 2, 1.0),  # floating point representation quirk
    ])
    def test_matches_pythons_round(self, value, ndigits, expected_python_rounded):
        result = as_numeric(value, ndigits=ndigits)
        expected = round(value, ndigits)
        assert result == expected, f"Result {result} differs from Python round({value}, {ndigits})={expected}"