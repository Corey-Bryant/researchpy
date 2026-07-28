"""
researchpy.engine — the three-layer engine pipeline.

Layers
------
1. **SyntaxParser** (``engine.syntax``)
   Universal input parsing — normalizes all calling conventions into a
   single ``SyntaxParser`` dataclass.

2. **MatrixEngine** (``engine.matrix``)
   Design matrix construction — builds numerical matrices via
   ``formulaic`` and wraps them in researchpy containers.

3. **TableEngine** (``engine.table``)
   Result table assembly — formats computed statistics into polished
   pandas DataFrames.
"""

from researchpy.engine.table import TableTermSpec, TableSpec, TableEngine
from researchpy.engine.syntax import SyntaxParser, as_continuous, as_categorical
from researchpy.engine.matrix import MatrixEngine

__all__ = [
    "SyntaxParser",
    "MatrixEngine",
    "TableEngine",
    "TableSpec",
    "TableTermSpec",
    "as_continuous",
    "as_categorical",
]

