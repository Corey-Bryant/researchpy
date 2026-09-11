from typing import Any

import numpy
import pandas





def as_array(data: Any, dtype: object = None, na_value: object = numpy.nan, copy=True) -> numpy.ndarray:
    """
    Convert input data to a numpy array of specified dtype.

    Parameters
    ----------
    data : array_like
        Input data (Series, array, list, etc.)
    dtype : data-type, optional
        Desired data type for the array. If None, infers from input.
    na_value : scalar, optional
        Value to use for missing values when converting from pandas Series. Default is numpy.nan.
    copy: bool, optional
        Whether to return a copy of the data. Default is True.

    Returns
    -------
    numpy.ndarray
        A 1-D array suitable for statistical computation.

    Raises
    ------
    TypeError
        If data cannot be converted to an array of specified type.
    ValueError
        If data is empty after removing non-numeric values.
    """
    if isinstance(data, numpy.ndarray):
        if data.ndim != 1:
            raise ValueError(
                f"Expected 1-D data, got {data.ndim}-D array with shape {data.shape}."
            )

        if dtype is not None and data.dtype != dtype:
            try:
                return data.astype(dtype, copy=False)

            except Exception as e:
                raise TypeError(f"Cannot convert array to dtype {dtype}: {e}")

        return data.copy(copy=copy)

    elif isinstance(data, pandas.Series):
        try:
            arr = data.to_numpy(dtype=dtype, na_value=na_value)

        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert Series '{data.name}' with dtype '{data.dtype}' to {dtype}. "
                f"Ensure data contains only numeric values."
            ) from e

    elif isinstance(data, (list, tuple)):
        try:
            arr = numpy.array(data, dtype=dtype)

        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert input to numeric array. "
                f"Ensure all elements are numeric."
            ) from e
    else:
        try:
            arr = numpy.asarray(data, dtype=dtype)

        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Unsupported data type '{type(data).__name__}'. "
                f"Expected array-like numeric data (Series, ndarray, list)."
            ) from e


    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1-D data, got {arr.ndim}-D array with shape {arr.shape}."
        )

    # Check if there are any valid (non-NaN) observations
    #if numpy.all(numpy.isnan(arr)):
    #    raise ValueError(
    #        "Data contains no valid (non-NaN) observations. "
    #        "Cannot compute summary statistics on empty data."
    #    )

    return arr




def validate_array(data: Any, to_numpy: bool = True, dtype: object = None, na_value: object = numpy.nan, copy = False) -> numpy.ndarray:
    """
    Validate and optionally convert input data to a numpy array of specified dtype.

    Parameters
    ----------
    data : array_like
        Input data (Series, array, list, etc.)
    to_numpy : bool
        Whether to convert the input to a numpy array. If False, returns the original data. Default is True.
    dtype : data-type, optional
        Desired data type for the array. If None, infers from input.
    na_value : scalar, optional
        Value to use for missing values when converting from pandas Series. Default is numpy.nan.

    Returns
    -------
    numpy.ndarray or original data.
        If to_numpy is True, returns a 1-D numpy array of specified dtype. Otherwise, returns the original data.

    Raises
    ------
    TypeError
        If data cannot be converted to a numpy array of specified type.
    ValueError
        If data is empty after removing non-numeric values, or if data is not 1-D.
    """
    if isinstance(data, numpy.ndarray):
        if data.ndim != 1:
            raise ValueError(
                f"Expected 1-D data, got {data.ndim}-D array with shape {data.shape}."
            )

        if dtype is not None and data.dtype != dtype:
            try:
                return data.astype(dtype)

            except Exception as e:
                raise TypeError(f"Cannot convert array to dtype {dtype}: {e}")

        return data

    else:
        if to_numpy:
            return as_array(data, dtype=dtype, na_value=na_value)

        else:
            return data
