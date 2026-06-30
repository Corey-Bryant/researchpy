# -*- coding: utf-8 -*-
"""
Categorical variable summaries: value counts, proportions, tabulate.

Supports Series, DataFrame, numpy arrays, lists, and tuples.
"""

from typing import Any, Dict, List, Optional, Union

import numpy
import pandas

from ..core.data_utils import as_array, validate_array


def tabulate(
    data: Union[pandas.Series, pandas.DataFrame, numpy.ndarray, list, tuple],
    sort_by: str = "count",
    ascending: bool = False,
    dropna: bool = True,
    decimals: int = 2,
    return_type: str = "Dataframe",
) -> Union[pandas.DataFrame, dict]:
    """Compute frequency table with counts, proportions, and cumulative proportions.

    For each unique level in the data, computes the frequency count, proportion
    (as percentage), and cumulative proportion in a single efficient pass using
    ``numpy.unique(return_counts=True)``.

    Parameters
    ----------
    data : Series, DataFrame, ndarray, list, or tuple
        The categorical data to tabulate. For DataFrames, produces a stacked
        table with a 'Variable' column identifying each column.
    sort_by : str, optional
        How to sort the output. One of:
        - ``"count"`` (default): sort by frequency count
        - ``"value"``: sort by level value (alphabetical/numeric)
    ascending : bool, optional
        Sort order. Default is False (descending for count, A-Z for value).
    dropna : bool, optional
        Whether to exclude NaN/None values. Default is True.
        When False, a row for missing values is included and proportions
        are computed against the total (including missing).
    decimals : int, optional
        Number of decimal places for proportions. Default is 2.
    return_type : str, optional
        Output format: ``"Dataframe"`` (default) or ``"Dictionary"``.

    Returns
    -------
    pandas.DataFrame or dict
        Table with columns: Value, Count, Percent, Cumulative Percent.
        For DataFrames: adds a 'Variable' column.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'], name='letter')
    >>> tabulate(s)
      Value  Count  Percent  Cumulative Percent
    0     a      3    50.00               50.00
    1     b      2    33.33               83.33
    2     c      1    16.67              100.00

    >>> tabulate(s, sort_by="value")
      Value  Count  Percent  Cumulative Percent
    0     a      3    50.00               50.00
    1     b      2    33.33               83.33
    2     c      1    16.67              100.00
    """
    # Validate return_type
    if return_type.upper() not in ("DATAFRAME", "DICTIONARY"):
        raise ValueError(
            f"Unsupported return_type '{return_type}'. Use 'Dataframe' or 'Dictionary'."
        )

    # Handle DataFrame: tabulate each column, stack results
    if isinstance(data, pandas.DataFrame):
        frames = []
        for col in data.columns:
            col_result = _tabulate_single(
                data[col], sort_by=sort_by, ascending=ascending,
                dropna=dropna, decimals=decimals,
            )
            col_result.insert(0, "Variable", "")
            col_result.iloc[0, 0] = col
            frames.append(col_result)
        result = pandas.concat(frames, ignore_index=True)
    else:
        # Convert to Series for uniform handling
        if isinstance(data, pandas.Series):
            series = data
        else:
            # ndarray, list, tuple
            series = pandas.Series(data)
        result = _tabulate_single(series, sort_by, ascending, dropna, decimals)

    if return_type.upper() == "DICTIONARY":
        return result.to_dict(orient="list")
    return result


def _tabulate_single(
    series: pandas.Series,
    sort_by: str,
    ascending: bool,
    dropna: bool,
    decimals: int,
) -> pandas.DataFrame:
    """Tabulate a single Series using numpy.unique for efficiency.

    Single-pass computation:
    1. Separate NaN values
    2. numpy.unique(return_counts=True) → unique levels + counts simultaneously
    3. Vectorized division → proportions
    4. numpy.cumsum → cumulative proportions
    """
    arr = series.to_numpy()

    # Separate missing values
    if arr.dtype.kind == 'f':
        na_mask = numpy.isnan(arr)
    else:
        na_mask = pandas.isna(arr)

    n_missing = int(na_mask.sum())
    clean = arr[~na_mask]

    # Single-pass: unique levels + frequency counts
    unique_values, counts = numpy.unique(clean, return_counts=True)

    # Determine total for proportion base
    if dropna:
        total = counts.sum()
    else:
        total = len(arr)  # includes missing in denominator

    # Vectorized proportions + cumulative (no iteration)
    proportions = (counts / total * 100).round(decimals)

    # Sort before computing cumulative
    if sort_by == "count":
        if ascending:
            sort_idx = numpy.argsort(counts, kind='stable')
        else:
            sort_idx = numpy.argsort(-counts, kind='stable')
        unique_values = unique_values[sort_idx]
        counts = counts[sort_idx]
        proportions = proportions[sort_idx]
    elif sort_by == "value":
        if ascending:
            sort_idx = numpy.argsort(unique_values, kind='stable')
        else:
            sort_idx = numpy.argsort(unique_values, kind='stable')[::-1]
        unique_values = unique_values[sort_idx]
        counts = counts[sort_idx]
        proportions = proportions[sort_idx]

    # Cumulative after sorting
    cumulative = numpy.cumsum(proportions).round(decimals)

    # Build result
    result = pandas.DataFrame({
        "Value": unique_values.tolist(),
        "Count": counts.astype(int).tolist(),
        "Percent": proportions.tolist(),
        "Cumulative Percent": cumulative.tolist(),
    })

    # Optionally append NaN row
    if not dropna and n_missing > 0:
        na_pct = round(n_missing / total * 100, decimals)
        cum_with_na = round(cumulative[-1] + na_pct, decimals) if len(cumulative) > 0 else na_pct
        na_row = pandas.DataFrame({
            "Value": ["<Missing>"],
            "Count": [n_missing],
            "Percent": [na_pct],
            "Cumulative Percent": [cum_with_na],
        })
        result = pandas.concat([result, na_row], ignore_index=True)

    return result

from ..core.data_utils import validate_array


def n_unique(data: pandas.Series) -> int:
    """Count the number of unique non-null values.

    Parameters
    ----------
    data : pandas.Series
        Input categorical or numeric Series.

    Returns
    -------
    int
        Number of unique non-null values.

    Examples
    --------
    >>> import pandas as pd
    >>> n_unique(pd.Series(["a", "b", "a", "c"]))
    3
    """
    return int(data.nunique())



def frequencies(data: Union[pandas.Series, numpy.ndarray, list], dropna=True,
                sort_by="count", ascending=False, decimals=2) -> Dict:
    """Compute the frequency counts of unique values, ignoring NaN.

    Parameters
    ----------
    data : array_like
        Input categorical or numeric data (Series, ndarray, or list).

    Returns
    -------
    dict
        A dictionary mapping unique non-null values to their frequency counts.

    Examples
    --------
    >>> frequencies(["a", "b", "a", "c"])
    {'a': 2, 'b': 1, 'c': 1}
    """
    arr = validate_array(data, copy=False)

    if arr.dtype.kind == "O":
        ...

    else:
        ...






def value_counts(
    data: pandas.Series,
    ascending: bool = False,
    dropna: bool = True,
) -> pandas.DataFrame:
    """Compute frequency counts for each unique value.

    Parameters
    ----------
    data : pandas.Series
        Input Series (categorical, object, or numeric).
    ascending : bool, optional
        Sort by count ascending. Default is False (descending).
    dropna : bool, optional
        Whether to exclude NaN values from the counts. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Value' and 'Count', sorted by count.

    Examples
    --------
    >>> import pandas as pd
    >>> value_counts(pd.Series(["a", "b", "a", "c"], name="letter"))
      Value  Count
    0     a      2
    1     b      1
    2     c      1
    """
    counts = data.value_counts(ascending=ascending, dropna=dropna)
    result = pandas.DataFrame({
        "Value": counts.index.tolist(),
        "Count": counts.values,
    })
    return result


def proportions(
    data: pandas.Series,
    ascending: bool = False,
    dropna: bool = True,
    as_percent: bool = True,
    decimals: int = 2,
) -> pandas.DataFrame:
    """Compute frequency counts and proportions for each unique value.

    Parameters
    ----------
    data : pandas.Series
        Input Series (categorical, object, or numeric).
    ascending : bool, optional
        Sort by count ascending. Default is False (descending).
    dropna : bool, optional
        Whether to exclude NaN values. Default is True.
    as_percent : bool, optional
        If True, proportions are expressed as percentages (0-100).
        If False, expressed as fractions (0-1). Default is True.
    decimals : int, optional
        Number of decimal places for proportion values. Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Value', 'Count', and 'Percent' (or 'Proportion').

    Examples
    --------
    >>> import pandas as pd
    >>> proportions(pd.Series(["a", "b", "a", "c"], name="letter"))
      Value  Count  Percent
    0     a      2    50.00
    1     b      1    25.00
    2     c      1    25.00
    """
    counts = data.value_counts(ascending=ascending, dropna=dropna)
    total = counts.sum()

    if as_percent:
        prop_values = (counts.values / total * 100).round(decimals)
        prop_label = "Percent"
    else:
        prop_values = (counts.values / total).round(decimals)
        prop_label = "Proportion"

    result = pandas.DataFrame({
        "Value": counts.index.tolist(),
        "Count": counts.values,
        prop_label: prop_values,
    })
    return result


def cumulative_proportions(
    data: pandas.Series,
    ascending: bool = False,
    dropna: bool = True,
    decimals: int = 2,
) -> pandas.DataFrame:
    """Compute frequency counts, proportions, and cumulative proportions.

    Parameters
    ----------
    data : pandas.Series
        Input Series (categorical, object, or numeric).
    ascending : bool, optional
        Sort by count ascending. Default is False (descending).
    dropna : bool, optional
        Whether to exclude NaN values. Default is True.
    decimals : int, optional
        Number of decimal places. Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Value', 'Count', 'Percent', 'Cumulative Percent'.

    Examples
    --------
    >>> import pandas as pd
    >>> cumulative_proportions(pd.Series(["a", "b", "a", "c"]))
      Value  Count  Percent  Cumulative Percent
    0     a      2    50.00               50.00
    1     b      1    25.00               75.00
    2     c      1    25.00              100.00
    """
    counts = data.value_counts(ascending=ascending, dropna=dropna)
    total = counts.sum()
    pct = (counts.values / total * 100).round(decimals)
    cum_pct = numpy.cumsum(pct).round(decimals)

    result = pandas.DataFrame({
        "Value": counts.index.tolist(),
        "Count": counts.values,
        "Percent": pct,
        "Cumulative Percent": cum_pct,
    })
    return result

