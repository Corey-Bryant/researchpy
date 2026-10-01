# -*- coding: utf-8 -*-
"""
Observation counts.

Reports how many values are present, how many are missing, and what share
of a variable is missing.  These are the counting statistics that every
other summary module reports alongside its estimates, so this module
establishes the table shape the rest of the package follows.

Public functions
----------------
observations       - N, N Missing, Percent Missing, and N Total together
n_obs              - Non-missing count only
n_missing          - Missing count only
percent_missing    - Percent missing only

All four accept every calling convention supported by
:class:`~researchpy.statistics._base.DesignSpec`::

    observations(df['systolic'])
    observations(df[['systolic', 'dbp']])
    observations(['systolic', 'dbp'], df)
    observations('systolic ~ C(disease)', df)
    observations(dv='systolic', by='disease', data=df)
    observations(dv='systolic', by='disease', over='drug', data=df)
    observations(dv='systolic', iv=['disease', 'drug'], data=df)

An existing ``DesignSpec`` may also be passed directly, which avoids
re-parsing when several statistics are computed from one specification.

Missing-data behaviour
----------------------
By default no rows are removed, so each variable is counted over the values
available to it.  ``casewise=True`` applies listwise deletion across the
whole design first, after which every variable shares a single N.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from researchpy.engine.table import TableEngine, TableSpec

from ._base import (
    DesignSpec,
    count_missing,
    percent_missing as _percent_missing_of,
)


__all__ = [
    "observations",
    "n_obs",
    "n_missing",
    "percent_missing",
]


# Canonical statistic labels, in display order.
N_OBS = "N"
N_MISSING = "N Missing"
PERCENT_MISSING = "Percent Missing"
N_TOTAL = "N Total"

ALL_STATISTICS: Tuple[str, ...] = (N_OBS, N_MISSING, PERCENT_MISSING, N_TOTAL)

VARIABLE_COLUMN = "Variable"
TERM_COLUMN = "Term"
LEVEL_COLUMN = "Level"


# ======================================================================
# Per-variable computation
# ======================================================================

def _count_values(values: Any, decimals: int) -> Dict[str, Any]:
    """Compute all observation statistics for a single variable.

    Parameters
    ----------
    values : array_like
        Values for one variable within one group.
    decimals : int
        Decimal places for the percentage.

    Returns
    -------
    dict
        Mapping of statistic label to value.
    """
    n_total = len(values)
    n_miss = count_missing(values)

    return {
        N_OBS: n_total - n_miss,
        N_MISSING: n_miss,
        PERCENT_MISSING: _percent_missing_of(values, decimals=decimals),
        N_TOTAL: n_total,
    }


def _rows_for(
    frame: pd.DataFrame,
    dv_names: Sequence[str],
    statistics: Sequence[str],
    decimals: int,
    prefix: Optional[Dict[str, Any]] = None,
    include_variable: bool = True,
) -> List[Dict[str, Any]]:
    """Build one result row per dependent variable for a single group.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows belonging to one group.
    dv_names : sequence of str
        Dependent variable column name(s).
    statistics : sequence of str
        Statistic labels to retain.
    decimals : int
        Decimal places for the percentage.
    prefix : dict or None
        Leading identifier columns (group levels, term labels) to place
        ahead of the statistics on each row.
    include_variable : bool
        Emit the ``Variable`` column.  Suppressed for grouped layouts with
        a single dependent variable, where it is constant and therefore
        redundant.

    Returns
    -------
    list of dict
    """
    rows: List[Dict[str, Any]] = []

    for dv in dv_names:
        row: Dict[str, Any] = dict(prefix or {})

        if include_variable:
            row[VARIABLE_COLUMN] = dv

        computed = _count_values(frame[dv], decimals=decimals)
        for stat in statistics:
            row[stat] = computed[stat]

        rows.append(row)

    return rows


def _level_prefix(group_cols: Sequence[str], key: Any) -> Dict[str, Any]:
    """Map a groupby key onto its grouping column names.

    Parameters
    ----------
    group_cols : sequence of str
        Grouping column name(s).
    key : scalar or tuple
        Key produced by ``DesignSpec.groups()``.

    Returns
    -------
    dict
    """
    values = key if isinstance(key, tuple) else (key,)

    return {col: val for col, val in zip(group_cols, values)}


# ======================================================================
# Layout builders
# ======================================================================

def _build_plain(design: DesignSpec, statistics: Sequence[str], engine: TableEngine) -> pd.DataFrame:
    """Build an ungrouped table: one row per dependent variable."""
    rows = _rows_for(design.frame, design.dv, statistics, design.decimals)

    return engine.build(rows, index_cols=[VARIABLE_COLUMN])


def _build_cell(design: DesignSpec, statistics: Sequence[str], engine: TableEngine) -> pd.DataFrame:
    """Build a grouped table indexed by the grouping variable levels.

    With two or more grouping variables the result carries a true
    ``MultiIndex`` over their combined levels.
    """
    group_cols = design.group_columns
    multi_dv = len(design.dv) > 1
    rows: List[Dict[str, Any]] = []

    for key, subframe in design.groups():
        rows.extend(
            _rows_for(
                subframe,
                design.dv,
                statistics,
                design.decimals,
                prefix=_level_prefix(group_cols, key),
                include_variable=multi_dv,
            )
        )

    index_cols = list(group_cols)
    if multi_dv:
        index_cols.append(VARIABLE_COLUMN)

    return engine.build(rows, index_cols=index_cols)


def _build_pivot(design: DesignSpec, statistics: Sequence[str], engine: TableEngine) -> pd.DataFrame:
    """Build a crossed table: *by* on the rows, *over* on the columns.

    The result carries a column ``MultiIndex`` of ``(statistic, level)``.
    """
    by_cols = list(design.spec.by or [])
    over_cols = list(design.spec.over or [])
    group_cols = by_cols + over_cols
    multi_dv = len(design.dv) > 1

    rows: List[Dict[str, Any]] = []
    for key, subframe in design.groups():
        rows.extend(
            _rows_for(
                subframe,
                design.dv,
                statistics,
                design.decimals,
                prefix=_level_prefix(group_cols, key),
                include_variable=multi_dv,
            )
        )

    long_form = engine.build(rows)

    row_vars = by_cols + ([VARIABLE_COLUMN] if multi_dv else [])

    return engine.pivot(
        long_form,
        row_vars=row_vars,
        col_vars=over_cols,
        value_col=list(statistics),
    )


def _build_stacked(
    design: DesignSpec,
    statistics: Sequence[str],
    engine: TableEngine,
    terms: Sequence[Tuple[str, List[str]]],
) -> pd.DataFrame:
    """Build marginal / mixed tables by stacking one sub-table per term.

    Each term is computed independently over its own grouping variable(s)
    and the results are concatenated, identified by a ``Term`` column.

    Parameters
    ----------
    terms : sequence of (str, list of str)
        ``(term_label, grouping_columns)`` pairs.
    """
    tables: List[pd.DataFrame] = []
    labels: List[str] = []
    multi_dv = len(design.dv) > 1

    for label, term_vars in terms:
        present = [c for c in term_vars if c in design.frame.columns]

        if not present:
            continue

        by: Any = present if len(present) > 1 else present[0]
        grouped = design.frame.groupby(by, dropna=False, observed=False, sort=True)

        rows: List[Dict[str, Any]] = []
        for key, subframe in grouped:
            values = key if isinstance(key, tuple) else (key,)
            level = " : ".join(str(v) for v in values)

            rows.extend(
                _rows_for(
                    subframe,
                    design.dv,
                    statistics,
                    design.decimals,
                    prefix={LEVEL_COLUMN: level},
                    include_variable=multi_dv,
                )
            )

        tables.append(engine.build(rows))
        labels.append(label)

    if not tables:
        return engine.build([])

    stacked = engine.stack(tables, labels=labels, label_column=TERM_COLUMN)

    index_cols = [TERM_COLUMN, LEVEL_COLUMN]
    if multi_dv:
        index_cols.append(VARIABLE_COLUMN)

    return stacked.set_index(index_cols)


# ======================================================================
# Dispatcher
# ======================================================================

def _observation_table(
    design: DesignSpec,
    statistics: Sequence[str],
) -> pd.DataFrame:
    """Route a design to the builder matching its layout.

    Parameters
    ----------
    design : DesignSpec
        Resolved design specification.
    statistics : sequence of str
        Statistic labels to report.

    Returns
    -------
    pd.DataFrame
    """
    spec = TableSpec(
        columns=list(statistics),
        rows=design.group_columns,
        variables=design.dv,
        statistics=list(statistics),
        decimals=design.decimals,
        **design.table_counts(),
    )
    engine = TableEngine(spec)

    layout = design.layout

    if layout == "plain":
        return _build_plain(design, statistics, engine)

    if layout == "cell":
        return _build_cell(design, statistics, engine)

    if layout == "pivot":
        return _build_pivot(design, statistics, engine)

    if layout == "marginal":
        terms = [(var, [var]) for var in (design.spec.IV or [])]

        return _build_stacked(design, statistics, engine, terms)

    if layout == "mixed":
        terms = [
            (sub.name, list(sub.variables))
            for sub in (design.spec.sub_specs or [])
        ]

        return _build_stacked(design, statistics, engine, terms)

    raise ValueError(f"Unrecognized layout '{layout}'.")


def _resolve_design(arg1: Any, arg2: Any, kwargs: Dict[str, Any]) -> DesignSpec:
    """Return a ``DesignSpec``, reusing one if it was supplied directly.

    Passing an existing ``DesignSpec`` avoids re-parsing when several
    statistics are computed from the same specification.
    """
    if isinstance(arg1, DesignSpec):
        return arg1

    return DesignSpec.from_args(arg1, arg2, **kwargs)


def _finalize(table: pd.DataFrame, return_type: str) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Convert the result to the requested return type.

    Parameters
    ----------
    table : pd.DataFrame
        Assembled result table.
    return_type : str
        ``"dataframe"`` or ``"dict"``.

    Returns
    -------
    pd.DataFrame or dict
        When ``"dict"`` is requested the index is reset first so that
        grouping levels survive as ordinary keys.
    """
    if return_type == "dict":
        return table.reset_index().to_dict(orient="list")

    return table


def _observation(
    statistics: Sequence[str],
    arg1: Any = None,
    arg2: Any = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Shared entry point behind every public function in this module."""
    design = _resolve_design(arg1, arg2, kwargs)
    table = _observation_table(design, statistics)

    return _finalize(table, design.return_type)


# ======================================================================
# Public API
# ======================================================================

def observations(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Any = None,
    iv: Any = None,
    by: Any = None,
    over: Any = None,
    data: Any = None,
    weights: Any = None,
    casewise: bool = False,
    decimals: int = 4,
    ci_level: float = 0.95,
    return_type: str = "dataframe",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Report N, N Missing, Percent Missing, and N Total.

    Parameters
    ----------
    arg1, arg2 : various
        Data, formula, column list, or an existing ``DesignSpec``.
    dv, iv, by, over, data, weights
        Keyword specification, as accepted by ``DesignSpec.from_args``.
    casewise : bool, optional
        Apply listwise deletion across the design first.  Default False.
    decimals : int, optional
        Decimal places for the percentage.  Default is 4.
    ci_level : float, optional
        Accepted for signature consistency; unused here.  Default is 0.95.
    return_type : str, optional
        ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

    Returns
    -------
    pd.DataFrame or dict

    Examples
    --------
    >>> observations(df['systolic'])                        # doctest: +SKIP
    >>> observations(dv='systolic', by='disease', data=df)  # doctest: +SKIP
    """
    return _observation(
        ALL_STATISTICS,
        arg1,
        arg2,
        dv=dv, iv=iv, by=by, over=over, data=data, weights=weights,
        casewise=casewise, decimals=decimals, ci_level=ci_level,
        return_type=return_type,
    )


def n_obs(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Any = None,
    iv: Any = None,
    by: Any = None,
    over: Any = None,
    data: Any = None,
    weights: Any = None,
    casewise: bool = False,
    decimals: int = 4,
    ci_level: float = 0.95,
    return_type: str = "dataframe",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Report the number of non-missing observations.

    Parameters
    ----------
    arg1, arg2 : various
        Data, formula, column list, or an existing ``DesignSpec``.
    dv, iv, by, over, data, weights
        Keyword specification, as accepted by ``DesignSpec.from_args``.
    casewise : bool, optional
        Apply listwise deletion across the design first.  Default False.
    decimals : int, optional
        Decimal places for formatted output.  Default is 4.
    ci_level : float, optional
        Accepted for signature consistency; unused here.  Default is 0.95.
    return_type : str, optional
        ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

    Returns
    -------
    pd.DataFrame or dict
    """
    return _observation(
        (N_OBS,),
        arg1,
        arg2,
        dv=dv, iv=iv, by=by, over=over, data=data, weights=weights,
        casewise=casewise, decimals=decimals, ci_level=ci_level,
        return_type=return_type,
    )


def n_missing(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Any = None,
    iv: Any = None,
    by: Any = None,
    over: Any = None,
    data: Any = None,
    weights: Any = None,
    casewise: bool = False,
    decimals: int = 4,
    ci_level: float = 0.95,
    return_type: str = "dataframe",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Report the number of missing observations.

    Parameters
    ----------
    arg1, arg2 : various
        Data, formula, column list, or an existing ``DesignSpec``.
    dv, iv, by, over, data, weights
        Keyword specification, as accepted by ``DesignSpec.from_args``.
    casewise : bool, optional
        Apply listwise deletion across the design first.  Default False.
        Note that this necessarily yields zero missing for every variable.
    decimals : int, optional
        Decimal places for formatted output.  Default is 4.
    ci_level : float, optional
        Accepted for signature consistency; unused here.  Default is 0.95.
    return_type : str, optional
        ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

    Returns
    -------
    pd.DataFrame or dict
    """
    return _observation(
        (N_MISSING,),
        arg1,
        arg2,
        dv=dv, iv=iv, by=by, over=over, data=data, weights=weights,
        casewise=casewise, decimals=decimals, ci_level=ci_level,
        return_type=return_type,
    )


def percent_missing(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Any = None,
    iv: Any = None,
    by: Any = None,
    over: Any = None,
    data: Any = None,
    weights: Any = None,
    casewise: bool = False,
    decimals: int = 4,
    ci_level: float = 0.95,
    return_type: str = "dataframe",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Report the percentage of observations that are missing.

    Parameters
    ----------
    arg1, arg2 : various
        Data, formula, column list, or an existing ``DesignSpec``.
    dv, iv, by, over, data, weights
        Keyword specification, as accepted by ``DesignSpec.from_args``.
    casewise : bool, optional
        Apply listwise deletion across the design first.  Default False.
        Note that this necessarily yields zero percent for every variable.
    decimals : int, optional
        Decimal places for the percentage.  Default is 4.
    ci_level : float, optional
        Accepted for signature consistency; unused here.  Default is 0.95.
    return_type : str, optional
        ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

    Returns
    -------
    pd.DataFrame or dict
        Percentages are of the group total, and are ``numpy.nan`` for an
        empty group.
    """
    return _observation(
        (PERCENT_MISSING,),
        arg1,
        arg2,
        dv=dv, iv=iv, by=by, over=over, data=data, weights=weights,
        casewise=casewise, decimals=decimals, ci_level=ci_level,
        return_type=return_type,
    )


