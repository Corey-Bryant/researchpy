"""
researchpy.engine — the three-layer engine pipeline.

Layers
------
1. **FormulaSpec** (``engine.syntax``)
   Universal input parsing — normalizes all calling conventions into a
   single ``FormulaSpec`` dataclass.

2. **DesignMatrix** (``engine.matrix``)
   Design matrix construction — builds numerical matrices via
   ``formulaic`` and wraps them in researchpy containers.

3. **TableEngine** (``engine.table``)
   Result table assembly — formats computed statistics into polished
   pandas DataFrames.
"""

from researchpy.engine.table import TableTermSpec, TableSpec, TableEngine
from researchpy.engine.syntax import FormulaSpec
from researchpy.engine.matrix import DesignMatrix

__all__ = [
    "FormulaSpec",
    "DesignMatrix",
    "TableEngine",
    "TableSpec",
    "TableTermSpec",
]

