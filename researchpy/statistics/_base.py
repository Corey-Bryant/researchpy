# -*- coding: utf-8 -*-
"""
Shared foundation for the statistics subpackage.

Provides :class:`DesignSpec`, the formulaic-backed design layer that every
statistics module consumes, together with the missing-data reporting helpers
shared across ``central_tendency``, ``dispersion``, ``observation``,
``shape``, and ``categorical``.

Design contract
---------------
1. :class:`~researchpy.engine.syntax.FormulaSpec` parses any supported
   calling convention.
2. :class:`DesignSpec` wraps that spec, resolves the design columns, and
   records observation counts.
3. Statistics modules consume a ``DesignSpec`` and never touch raw user
   input.

The design layer selects the *raw* design columns rather than a dummy-coded
model matrix, because grouping requires the original level labels (e.g.
``disease == "hypertension"``), which a coded matrix discards.  Formulaic
still governs variable resolution by way of ``FormulaSpec`` / ``ModelTerms``.

Missing-data policy
-------------------
Imputation is the user's responsibility and happens outside the engine.
This layer performs row alignment and reporting only.

- ``casewise=False`` (default): no rows are removed.  Each statistic uses
  the values available for its own variable, via the per-variable *counts*.
- ``casewise=True``: listwise deletion across all design columns, so every
  statistic in the call shares a single N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from researchpy.containers.base import CoreDataclass
from researchpy.core.data_utils import (
    as_array,
    as_frame,
    as_series,
    infer_variable_kind,
    validate_array,
)
from researchpy.engine.syntax import FormulaSpec


__all__ = [
    "DesignSpec",
    "count_missing",
    "count_valid",
    "percent_missing",
    "as_array",
    "as_frame",
    "as_series",
    "infer_variable_kind",
    "validate_array",
]


# ======================================================================
# Missing-data reporting helpers
# ======================================================================

def count_missing(values: Any) -> int:
    """Count missing values in a variable.

    Parameters
    ----------
    values : array_like
        Input data (Series, ndarray, list, or tuple).

    Returns
    -------
    int
        Number of missing (NA/NaN/NaT) values.
    """
    return int(np.count_nonzero(pd.isna(as_series(values))))


def count_valid(values: Any) -> int:
    """Count non-missing values in a variable.

    Parameters
    ----------
    values : array_like
        Input data (Series, ndarray, list, or tuple).

    Returns
    -------
    int
        Number of non-missing values.
    """
    series = as_series(values)

    return int(len(series) - np.count_nonzero(pd.isna(series)))


def percent_missing(values: Any, decimals: int = 4) -> float:
    """Compute the percentage of missing values in a variable.

    Parameters
    ----------
    values : array_like
        Input data (Series, ndarray, list, or tuple).
    decimals : int, optional
        Number of decimal places to round to.  Default is 4.

    Returns
    -------
    float
        Percent missing, or ``numpy.nan`` if the variable is empty.
    """
    series = as_series(values)
    n = len(series)

    if n == 0:
        return float(np.nan)

    return round(float(np.count_nonzero(pd.isna(series))) / n * 100.0, decimals)


# ======================================================================
# DesignSpec
# ======================================================================

@dataclass
class DesignSpec(CoreDataclass):
    """Formulaic-backed design layer consumed by all statistics modules.

    Attributes
    ----------
    spec : FormulaSpec or None
        Parsed input specification produced by ``FormulaSpec.from_args``.
    frame : pd.DataFrame or None
        Resolved design columns.  Retains all rows unless ``casewise=True``.
    n_total : int
        Number of rows handed to the engine, before any deletion.
    n_incomplete : int
        Number of rows with at least one missing value across the design
        columns.  Computed once at construction.
    counts : dict
        Per-variable ``{column: (n_observed, n_missing)}``.  This is the
        computational truth used by the individual statistics.
    casewise : bool
        Opt-in listwise deletion across all design columns.  Default False.
    decimals : int
        Decimal places for formatted output.  Default is 4.
    ci_level : float
        Confidence level for interval estimates.  Default is 0.95.
    return_type : str
        ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

    Notes
    -----
    Under ``casewise=False``, ``n_observed`` reports *complete cases* and is
    not the same as ``len(frame)``: no rows are removed, so the frame still
    holds ``n_total`` rows while individual statistics draw on the larger
    per-variable N recorded in *counts*.
    """

    spec: Optional[FormulaSpec] = None
    frame: Optional[pd.DataFrame] = None
    n_total: int = 0
    n_incomplete: int = 0
    counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    casewise: bool = False
    decimals: int = 4
    ci_level: float = 0.95
    return_type: str = "dataframe"

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = "Researchpy.DesignSpec"

    # ------------------------------------------------------------------
    # Derived counts
    # ------------------------------------------------------------------
    @property
    def n_missing(self) -> int:
        """Rows with at least one missing value across the design columns."""
        return self.n_incomplete

    @property
    def n_observed(self) -> int:
        """Complete cases: ``n_total - n_missing``."""
        return self.n_total - self.n_incomplete

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def dv(self) -> List[str]:
        """Dependent variable column name(s)."""
        return list(self.spec.DV) if self.spec else []

    @property
    def group_columns(self) -> List[str]:
        """All grouping column names (IV, by, over, and sub-spec terms).

        Mixed formulas carry their grouping variables on ``sub_specs``
        rather than on *IV* / *by* / *over*, so those are included here to
        ensure the design frame resolves them.
        """
        if self.spec is None:
            return []

        cols: List[str] = []
        for group in (self.spec.IV, self.spec.by, self.spec.over):
            for col in (group or []):
                if col not in cols:
                    cols.append(col)

        for sub in (self.spec.sub_specs or []):
            for col in (sub.variables or []):
                if col not in cols:
                    cols.append(col)

        return cols

    @property
    def design_columns(self) -> List[str]:
        """Dependent and grouping columns, in order, without duplicates."""
        cols = list(self.dv)

        for col in self.group_columns:
            if col not in cols:
                cols.append(col)

        return cols

    @property
    def layout(self) -> str:
        """Table layout implied by the parsed spec.

        Returns
        -------
        str
            One of ``"plain"``, ``"marginal"``, ``"cell"``, ``"pivot"``,
            or ``"mixed"``.
        """
        if self.spec is None:
            return "plain"

        if self.spec.sub_specs:
            return "mixed"

        if self.spec.IV:
            return "marginal"

        if self.spec.by and self.spec.over:
            return "pivot"

        if self.spec.by:
            return "cell"

        return "plain"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_args(
        cls,
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
    ) -> "DesignSpec":
        """Parse user input and resolve it into a design specification.

        Parsing is delegated in full to ``FormulaSpec.from_args``, so every
        supported calling convention and validation rule is inherited.

        Parameters
        ----------
        arg1, arg2 : various
            Positional arguments forwarded to ``FormulaSpec.from_args``.
        dv, iv, by, over, data, weights
            Keyword arguments forwarded to ``FormulaSpec.from_args``.
        casewise : bool, optional
            Apply listwise deletion across all design columns.  Default False.
        decimals : int, optional
            Decimal places for formatted output.  Default is 4.
        ci_level : float, optional
            Confidence level for interval estimates.  Default is 0.95.
        return_type : str, optional
            ``"dataframe"`` or ``"dict"``.  Default is ``"dataframe"``.

        Returns
        -------
        DesignSpec

        Raises
        ------
        ValueError
            If *ci_level* is not strictly between 0 and 1, if *return_type*
            is unrecognized, or if the input cannot be resolved.
        """
        if not 0.0 < ci_level < 1.0:
            raise ValueError(
                f"'ci_level' must be strictly between 0 and 1, got {ci_level}."
            )

        if return_type not in ("dataframe", "dict"):
            raise ValueError(
                f"'return_type' must be 'dataframe' or 'dict', got '{return_type}'."
            )

        spec = FormulaSpec.from_args(
            arg1,
            arg2,
            dv=dv,
            iv=iv,
            by=by,
            over=over,
            data=data,
            weights=weights,
        )

        design = cls(
            spec=spec,
            casewise=casewise,
            decimals=decimals,
            ci_level=ci_level,
            return_type=return_type,
        )
        design._materialize()

        return design

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------
    def _materialize(self) -> None:
        """Resolve design columns and record observation counts.

        Selects the raw design columns rather than a coded model matrix,
        because grouping requires the original level labels.  Counts are
        computed once here rather than on each property access.
        """
        if self.spec is None or self.spec.data is None:
            self.frame = None
            self.n_total = 0
            self.n_incomplete = 0
            self.counts = {}

            return

        source = as_frame(self.spec.data)
        columns = [c for c in self.design_columns if c in source.columns]
        frame = source[columns] if columns else source

        self.n_total = int(len(frame))

        if len(frame.columns):
            incomplete_mask = pd.isna(frame).any(axis=1)
        else:
            incomplete_mask = pd.Series(False, index=frame.index)

        self.n_incomplete = int(np.count_nonzero(incomplete_mask))

        if self.casewise and self.n_incomplete:
            frame = frame.loc[~incomplete_mask]

        self.frame = frame

        self.counts = {
            col: (count_valid(frame[col]), count_missing(frame[col]))
            for col in frame.columns
        }

    # ------------------------------------------------------------------
    # Iteration contract
    # ------------------------------------------------------------------
    def groups(self) -> Iterator[Tuple[Any, pd.DataFrame]]:
        """Iterate over the analysis groups implied by the layout.

        Yields
        ------
        tuple of (key, pd.DataFrame)
            *key* is ``None`` for ungrouped designs, a scalar for a single
            grouping variable, and a tuple for multiple grouping variables.

        Notes
        -----
        Grouping uses ``dropna=False`` so that missing group levels remain
        visible in the output rather than silently disappearing, and
        ``observed=False`` so unused categorical levels are retained.  This
        is the single iteration contract used by every statistics module.
        """
        if self.frame is None:
            return

        group_cols = [c for c in self.group_columns if c in self.frame.columns]

        if not group_cols:
            yield None, self.frame

            return

        # A single grouping column is passed as a scalar so that group keys
        # stay scalar; a list of length one would yield 1-tuple keys.
        by: Any = group_cols if len(group_cols) > 1 else group_cols[0]

        grouped = self.frame.groupby(
            by,
            dropna=False,
            observed=False,
            sort=True,
        )

        for key, subframe in grouped:
            yield key, subframe

    # ------------------------------------------------------------------
    # Hand-off to the table layer
    # ------------------------------------------------------------------
    def table_counts(self) -> Dict[str, int]:
        """Observation counts formatted for :class:`TableSpec`.

        Returns
        -------
        dict
            Keys ``n_total``, ``n_observed``, and ``n_incomplete``.
        """
        return {
            "n_total": self.n_total,
            "n_observed": self.n_observed,
            "n_incomplete": self.n_incomplete,
        }





