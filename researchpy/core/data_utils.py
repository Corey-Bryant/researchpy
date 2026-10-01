"""
This module provides utility functions for validating and converting data to appropriate formats for statistical analysis.
Functions include:
- `validate_array`: Validates and optionally converts input data to a 1-D numpy array of specified dtype.
- `as_array`: Converts input data to a 1-D numpy array of specified dtype.
- `as_frame`: Converts input data to a pandas DataFrame.
- `as_series`: Converts input data to a 1-D pandas Series.
- `infer_variable_kind`: Infers the analytic kind of variable (boolean, datetime, numeric, categorical, or unknown).
"""

from typing import Any, Optional, Sequence

import numpy
import pandas




def validate_array(data: Any, to_numpy: bool = True, dtype: Any = None, na_value: Any = numpy.nan, copy = False) -> numpy.ndarray:
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



def as_array(data: Any, dtype: Any = None, na_value: Any = numpy.nan, copy=True) -> numpy.ndarray:
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

        return data.copy() if copy else data

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



def as_frame(data: Any, columns: Optional[Sequence[str]] = None) -> pandas.DataFrame:
    """
    Convert input data to a pandas DataFrame.

    Parameters
    ----------
    data : array_like
        Input data (DataFrame, Series, ndarray, list, tuple, or dict).
    columns : list of str, optional
        Column names to assign. If None, names are inferred from the input
        or generated.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    TypeError
        If data cannot be converted to a DataFrame.
    ValueError
        If data has more than 2 dimensions.
    """
    if isinstance(data, pandas.DataFrame):
        return data if columns is None else data[list(columns)]

    if isinstance(data, pandas.Series):
        name = data.name if data.name is not None else "value"
        frame = data.to_frame(name=name)

        return frame if columns is None else frame.set_axis(list(columns), axis=1)

    if isinstance(data, dict):
        return pandas.DataFrame(data)

    if isinstance(data, (numpy.ndarray, list, tuple)):
        arr = numpy.asarray(data)

        if arr.ndim == 1:
            return pandas.DataFrame({(columns[0] if columns else "value"): arr})

        elif arr.ndim == 2:
            names = list(columns) if columns else [f"col_{i}" for i in range(arr.shape[-1])]

            return pandas.DataFrame(arr, columns=names)

        else:
            raise ValueError(
                f"Expected 1-D or 2-D data, got {arr.ndim}-D array with shape {arr.shape}."
            )

    raise TypeError(
        f"Unsupported data type '{type(data).__name__}'. "
        f"Expected DataFrame, Series, ndarray, list, tuple, or dict."
    )




def as_series(data: Any, name: Optional[str] = None) -> pandas.Series:
    """
    Convert input data to a 1-D pandas Series.

    Parameters
    ----------
    data : array_like
        Input data (Series, single-column DataFrame, ndarray, list, or tuple).
    name : str, optional
        Name to assign to the Series. If None, an existing name is preserved.

    Returns
    -------
    pandas.Series

    Raises
    ------
    ValueError
        If data is not 1-D, or is a DataFrame with more than one column.
    TypeError
        If data cannot be converted to a Series.
    """
    if isinstance(data, pandas.Series):
        return data.rename(name) if name is not None else data

    if isinstance(data, pandas.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                f"Expected a single-column DataFrame, got {data.shape[1]} columns."
            )

        series = data.iloc[:, 0]

        return series.rename(name) if name is not None else series

    if isinstance(data, (numpy.ndarray, list, tuple)):
        arr = numpy.asarray(data)

        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1-D data, got {arr.ndim}-D array with shape {arr.shape}."
            )

        return pandas.Series(arr, name=name)

    raise TypeError(
        f"Unsupported data type '{type(data).__name__}'. "
        f"Expected Series, single-column DataFrame, ndarray, list, or tuple."
    )




def infer_variable_kind(values: Any) -> str:
    """
    Infer the analytic kind of variable.

    Parameters
    ----------
    values : array_like
        Input data (Series, ndarray, list, or tuple).

    Returns
    -------
    str
        One of "boolean", "datetime", "numeric", "categorical", or "unknown".

    Notes
    -----
    Boolean is checked before numeric because numpy treats bool as a numeric
    dtype, and a boolean variable should be summarized as a proportion rather
    than as a continuous measure.
    """
    if isinstance(values, pandas.Series):
        series = values
    else:
        series = pandas.Series(numpy.asarray(values).ravel())

    dtype = series.dtype

    if pandas.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pandas.api.types.is_datetime64_any_dtype(dtype) or \
       pandas.api.types.is_timedelta64_dtype(dtype):
        return "datetime"

    if pandas.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if isinstance(dtype, pandas.CategoricalDtype) or \
       pandas.api.types.is_object_dtype(dtype) or \
       pandas.api.types.is_string_dtype(dtype):
        return "categorical"

    return "unknown"




