# -*- coding: utf-8 -*-
"""
Univariate Summary Containers

Standardized containers for univariate summary statistics.
"""

from __future__ import annotations

from . import CoreDataclass
from .base import CoreDataclass
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd





@dataclass
class UnivariateModel(CoreDataclass):
    """Container for univariate summary statistics results.

    Attributes
    ----------
    name : str or None
        The variable name.
    test_name:
    statistics : dict
        Ordered mapping of statistic name -> computed value.

    Examples
    --------
    >>> result = UnivariateModel(name="age", statistics={"N": 100, "Mean": 35.4})
    >>> result.to_dataframe()
       Name    N  Mean
    0   age  100  35.4
    """

    term: str
    test_name: str
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame (single row).

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with statistic names as columns.
        """
        row = {}
        if self.name is not None:
            row["Name"] = self.name

        row.update(self.statistics)

        return pd.DataFrame([row])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with statistic names as keys and computed values as values.
        """
        result = {}
        if self.name is not None:
            result["Name"] = self.name

        result.update(self.statistics)

        return result




@dataclass
class SummaryResult(CoreDataclass):
    """Container for univariate summary statistics results.

    Attributes
    ----------
    name : str or None
        The variable name.
    statistics : dict
        Ordered mapping of statistic name -> computed value.

    Examples
    --------
    >>> result = SummaryResult(name="age", statistics={"N": 100, "Mean": 35.4})
    >>> result.to_dataframe()
       Name    N  Mean
    0   age  100  35.4
    """

    name: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame (single row).

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with statistic names as columns.
        """
        row = {}
        if self.name is not None:
            row["Name"] = self.name

        row.update(self.statistics)

        return pd.DataFrame([row])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with statistic names as keys and computed values as values.
        """
        result = {}
        if self.name is not None:
            result["Name"] = self.name

        result.update(self.statistics)

        return result
