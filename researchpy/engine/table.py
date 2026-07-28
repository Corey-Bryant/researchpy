

from dataclasses import dataclass, field
from typing import List, Optional, Union

from researchpy.containers import CoreDataclass


@dataclass
class TableSpec(CoreDataclass):

    columns: List[str] = field(default_factory=list)
    rows: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)




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
        Raw term string as written in the formula (e.g., ``"C(x)"`` or ``"C(k):C(z)"``).
    name : str
        Cleaned term name (e.g., ``"x"`` or ``"k:z"``).
    layout : str
        How this term should be computed: ``"iv"`` (marginal) or ``"by"`` (cell).
    variables : list of str
        The grouping variable column name(s) for this term.
    """

    term: str
    name: str
    decimals: int
    layout: str
    variables: List[str]