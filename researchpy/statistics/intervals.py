# -*- coding: utf-8 -*-
"""
Confidence interval computation.

Currently supports t-distribution based intervals for the mean.
Future: bootstrap CIs, proportion CIs, median CIs.

Primary functions support 5 calling conventions:
    1. confidence_interval(series_or_array)
    2. confidence_interval("y ~ C(x)", df)
    3. confidence_interval(["y", "k"], df)
    4. confidence_interval(dv="y", by="x", data=df)
       confidence_interval(dv="y", iv=["x","k"], data=df)
       confidence_interval(dv="y", by="x", over="k", data=df)
    5. confidence_interval("y ~ C(x):C(k)", df)
       confidence_interval("y ~ C(x)*C(k)", df)
"""

from typing import Any, List, Optional, Tuple, Union

import numpy
import pandas
import scipy.stats

from ..core.data_utils import validate_array


def confidence_interval(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    confidence_level: float = 0.95,
    decimals: int = 4,
) -> Union[Tuple[float, float], pandas.DataFrame]:
    """Compute a t-distribution based confidence interval for the mean.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute CI directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
        Mutually exclusive with by/over.
    by : str or list of str, optional
        Row grouping variable(s) for cell computation.
    over : str or list of str, optional
        Column grouping variable(s) for pivot layout. Requires by.
    data : pd.DataFrame, optional
        Data source when using keyword arguments.
    confidence_level : float, optional
        The confidence level (between 0 and 1 exclusive). Default is 0.95.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    tuple of (float, float)
        When single variable, no groups. A tuple of (lower_bound, upper_bound).
    pandas.DataFrame
        When multiple variables or grouped.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1 (exclusive).
        If fewer than 2 non-NaN observations are present.

    Notes
    -----
    Uses the t-distribution with n-1 degrees of freedom:
        CI = mean ± t_{α/2, n-1} * SE

    where SE = SD / sqrt(n) and α = 1 - confidence_level.

    References
    ----------
    .. [1] Devore, J.L. (2011). Probability and Statistics for Engineering
       and the Sciences, 8th ed. Cengage Learning.

    Examples
    --------
    >>> confidence_interval([1, 2, 3, 4, 5], confidence_level=0.95)
    (1.038, 4.962)

    >>> confidence_interval("y ~ C(g)", df, confidence_level=0.95)
             CI Lower  CI Upper
    g
    a          ...       ...
    b          ...       ...
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(
            f"confidence_level must be between 0 and 1 (exclusive), got {confidence_level}. "
            f"For a 95% CI, use confidence_level=0.95."
        )

    def _ci_scalar(arr):
        """Compute CI for a single array, returns tuple."""
        clean = arr[~numpy.isnan(arr)]
        n = len(clean)
        if n < 2:
            raise ValueError(
                f"At least 2 non-NaN observations are required for a confidence interval, "
                f"got {n}. Cannot estimate variability from a single observation."
            )
        df = n - 1
        data_mean = float(numpy.mean(clean))
        se = float(scipy.stats.sem(clean))
        lower, upper = scipy.stats.t.interval(confidence_level, df, loc=data_mean, scale=se)
        return (round(float(lower), decimals), round(float(upper), decimals))

    # For ungrouped, single-DV case: return tuple directly
    # For grouped/multi-DV: use the routing infrastructure with a custom approach
    from ..core.spec import resolve

    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data)

    # === No groups ===
    if spec.iv is None and spec.by is None and spec.over is None:
        if len(spec.dv) == 1:
            arr = validate_array(spec.data[spec.dv[0]].to_numpy(), dtype=float)
            return _ci_scalar(arr)
        else:
            # Multiple DVs → DataFrame with columns per DV
            rows = []
            for col in spec.dv:
                arr = validate_array(spec.data[col].to_numpy(), dtype=float)
                lower, upper = _ci_scalar(arr)
                rows.append({"Variable": col, "CI Lower": lower, "CI Upper": upper})
            return pandas.DataFrame(rows)

    # === Grouped (iv, by, or by+over) ===
    ci_label = f"{int(confidence_level * 100)}% CI"

    def _ci_grouped(data_df, dv_col, groups, dec, **kw):
        """Compute CI for each group via iteration."""
        from ..core.matrix_engine import _build_cell_series
        cell_series = _build_cell_series(data_df, groups)
        unique_cells = sorted(cell_series.unique())

        results = []
        for cell in unique_cells:
            mask = cell_series == cell
            arr = data_df.loc[mask, dv_col].to_numpy(dtype=float)
            clean = arr[~numpy.isnan(arr)]

            if len(clean) >= 2:
                df = len(clean) - 1
                data_mean = float(numpy.mean(clean))
                se = float(scipy.stats.sem(clean))
                lower, upper = scipy.stats.t.interval(confidence_level, df, loc=data_mean, scale=se)
                lower = round(float(lower), dec)
                upper = round(float(upper), dec)
            else:
                lower = numpy.nan
                upper = numpy.nan

            # Build row with individual group variable values
            row = {}
            if len(groups) == 1:
                row[groups[0]] = cell
            else:
                parts = cell.split(":")
                for i, group_name in enumerate(groups):
                    row[group_name] = parts[i]

            row["CI Lower"] = lower
            row["CI Upper"] = upper
            results.append(row)

        return pandas.DataFrame(results)

    # Use _route_computation-like logic but with our custom grouped function
    # Since CI returns two columns (not a scalar), we handle it specially

    if spec.iv is not None:
        # Marginal: compute for each iv independently, stack
        frames = []
        for dv_col in spec.dv:
            for iv_var in spec.iv:
                result_df = _ci_grouped(spec.data, dv_col, [iv_var], decimals)
                normalized = pandas.DataFrame({
                    "Factor": iv_var,
                    "Level": result_df[iv_var].values,
                    "CI Lower": result_df["CI Lower"].values,
                    "CI Upper": result_df["CI Upper"].values,
                })
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                frames.append(normalized)
        return pandas.concat(frames, ignore_index=True)

    elif spec.over is not None:
        # Pivot: compute flat then reshape
        # For CI we produce two pivot tables (lower, upper) or a combined view
        all_groups = spec.by + spec.over
        frames = []
        for dv_col in spec.dv:
            result_df = _ci_grouped(spec.data, dv_col, all_groups, decimals)
            if len(spec.dv) > 1:
                result_df.insert(0, "Variable", dv_col)
            frames.append(result_df)

        flat = pandas.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        # Return flat with MultiIndex for by-variables (pivot of tuple is complex)
        flat = flat.set_index(all_groups)
        return flat

    else:
        # Cell (by only): grouped with MultiIndex
        frames = []
        for dv_col in spec.dv:
            result_df = _ci_grouped(spec.data, dv_col, spec.by, decimals)
            if len(spec.dv) > 1:
                result_df.insert(0, "Variable", dv_col)
            frames.append(result_df)

        if len(frames) == 1:
            result = frames[0]
        else:
            result = pandas.concat(frames, ignore_index=True)

        result = result.set_index(spec.by)
        return result

