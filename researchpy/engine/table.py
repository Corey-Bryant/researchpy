"""
TableEngine — the result table construction layer.

Takes the output of :class:`~researchpy.engine.matrix.MatrixEngine` (or
raw computed statistics) and assembles the final pandas DataFrames that
researchpy returns to users.

This module provides:

- **TableTermSpec** — per-term layout metadata for mixed formulas.
- **TableSpec** — overall table layout specification (columns, rows,
  variables, statistics).
- **TableEngine** — the builder class that takes computed results and
  assembles formatted DataFrames according to a ``TableSpec``.

Usage
-----
>>> from researchpy.engine.table import TableEngine, TableSpec
>>> spec = TableSpec(
...     columns=["N", "Mean", "SD"],
...     rows=["drug"],
...     variables=["y"],
...     statistics=["N", "Mean", "SD"],
... )
>>> engine = TableEngine(spec)
>>> result_df = engine.build(computed_rows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from researchpy.containers.base import CoreDataclass


# ======================================================================
# Data specifications
# ======================================================================

@dataclass
class TableTermSpec(CoreDataclass):
    """One term extracted from a mixed formula.

    Used when a formula contains both main effects and interactions that
    don't form a pure star expansion (e.g., ``y ~ C(x) + C(k):C(z)``).
    Each term is computed independently and results are stacked with a
    ``Term`` column identifying the source.

    Attributes
    ----------
    term : str
        Raw term string as written in the formula (e.g., ``"C(x)"``
        or ``"C(k):C(z)"``).
    name : str
        Cleaned term name (e.g., ``"x"`` or ``"k:z"``).
    layout : str
        How this term should be computed: ``"iv"`` (marginal) or
        ``"by"`` (cell).
    variables : list of str
        The grouping variable column name(s) for this term.
    """

    term: str = ""
    name: str = ""
    layout: str = ""
    variables: List[str] = field(default_factory=list)


@dataclass
class TableSpec(CoreDataclass):
    """Overall table layout specification.

    Describes how a result table should be structured: which columns to
    include, which variables define the rows, and which statistics to
    compute.

    Attributes
    ----------
    columns : list of str
        Column headers for the result table (e.g., ``["N", "Mean", "SD"]``).
    rows : list of str
        Variable(s) whose levels define the rows (grouping variables).
    variables : list of str
        Dependent variable(s) being summarized.
    statistics : list of str
        Statistics to compute (e.g., ``["N", "Mean", "SD", "SE", "CI"]``).
    title : str or None
        Optional table title for display.
    footnotes : list of str or None
        Optional footnotes to attach to the table.
    decimals : int
        Number of decimal places for formatting.  Default is 4.
    """

    columns: List[str] = field(default_factory=list)
    rows: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)
    title: Optional[str] = None
    footnotes: Optional[List[str]] = None
    decimals: int = 4


# ======================================================================
# TableEngine — the builder
# ======================================================================

@dataclass
class TableEngine(CoreDataclass):
    """Assembles result DataFrames from computed statistics.

    The ``TableEngine`` is the final step in the engine pipeline:

    1. **FormulaSpec** normalizes user input into a spec.
    2. **MatrixEngine** builds design matrices (for models).
    3. **TableEngine** formats computed results into polished DataFrames.

    Attributes
    ----------
    spec : TableSpec
        Layout specification controlling the table structure.
    """

    spec: TableSpec = field(default_factory=TableSpec)

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = "Researchpy.TableEngine"

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------
    def build(
        self,
        rows: List[Dict[str, Any]],
        index_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build a formatted result DataFrame from row data.

        Parameters
        ----------
        rows : list of dict
            Each dict represents one row of the result table.  Keys
            should match ``spec.columns`` (extra keys are kept;
            missing keys become ``NaN``).
        index_cols : list of str or None
            Column(s) to set as the DataFrame index.  Default is ``None``
            (integer index).

        Returns
        -------
        pd.DataFrame
            Formatted result table.
        """
        df = pd.DataFrame(rows)

        # Reorder columns to match spec if provided
        if self.spec.columns:
            ordered = [c for c in self.spec.columns if c in df.columns]
            extra = [c for c in df.columns if c not in self.spec.columns]
            df = df[ordered + extra]

        # Apply rounding
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(self.spec.decimals)

        # Set index
        if index_cols:
            existing = [c for c in index_cols if c in df.columns]
            if existing:
                df = df.set_index(existing)

        return df

    # ------------------------------------------------------------------
    # Stack: combine multiple sub-tables (for marginal / sub_specs)
    # ------------------------------------------------------------------
    def stack(
        self,
        tables: List[pd.DataFrame],
        labels: Optional[List[str]] = None,
        label_column: str = "Term",
    ) -> pd.DataFrame:
        """Vertically stack multiple result DataFrames.

        Used for marginal (``+``) and mixed-formula layouts where each
        term produces its own table and results are concatenated.

        Parameters
        ----------
        tables : list of pd.DataFrame
            Individual result tables to stack.
        labels : list of str or None
            Labels to identify each sub-table.  If provided, a
            *label_column* is prepended to each table.
        label_column : str
            Name of the identifying column.  Default is ``"Term"``.

        Returns
        -------
        pd.DataFrame
            Combined table with all sub-tables stacked vertically.
        """
        if labels and len(labels) == len(tables):
            for label, tbl in zip(labels, tables):
                tbl.insert(0, label_column, label)

        result = pd.concat(tables, ignore_index=True)
        return result

    # ------------------------------------------------------------------
    # Pivot: reshape for pivot layout (by * over)
    # ------------------------------------------------------------------
    def pivot(
        self,
        df: pd.DataFrame,
        row_vars: List[str],
        col_vars: List[str],
        value_col: str,
    ) -> pd.DataFrame:
        """Reshape a long-form result table into a pivot layout.

        Used for star-expansion (``*``) formulas where the first
        factor defines rows and the second defines columns.

        Parameters
        ----------
        df : pd.DataFrame
            Long-form result table containing grouping columns and
            a value column.
        row_vars : list of str
            Column(s) whose unique values form the row index.
        col_vars : list of str
            Column(s) whose unique values form the column headers.
        value_col : str
            Column containing the statistic values to pivot.

        Returns
        -------
        pd.DataFrame
            Pivoted table.
        """
        return df.pivot_table(
            index=row_vars,
            columns=col_vars,
            values=value_col,
            aggfunc="first",
        )

    # ------------------------------------------------------------------
    # Info / repr
    # ------------------------------------------------------------------
    def info(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  columns={self.spec.columns}")
        lines.append(f"  rows={self.spec.rows}")
        lines.append(f"  variables={self.spec.variables}")
        lines.append(f"  statistics={self.spec.statistics}")
        lines.append(f"  decimals={self.spec.decimals}")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.info()
