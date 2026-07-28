# Engine Architecture

> **Module:** `researchpy.engine`
>
> **Location:** `researchpy/engine/`
>
> **Public API:** `from researchpy.engine import SyntaxParser, MatrixEngine, TableEngine, TableSpec, TableTermSpec, as_continuous, as_categorical`

---

## Overview

The `engine/` package implements a **three-layer pipeline** that serves as the universal foundation
for every computation in researchpy. Regression models, descriptive statistics (numerical and
categorical), inferential tests, and everything in between.

```
User Input ──► SyntaxParser ──► MatrixEngine ──► TableEngine ──► pd.DataFrame
               (parse)          (compute)        (format)        (result)
```

| Layer | Module | Class | Responsibility |
|-------|--------|-------|----------------|
| **1. Syntax** | `engine/syntax.py` | `SyntaxParser` | Universal input parsing — normalizes all calling conventions into a single dataclass |
| **2. Matrix** | `engine/matrix.py` | `MatrixEngine` | Design matrix construction via `formulaic` — builds LHS/RHS numerical matrices |
| **3. Table** | `engine/table.py` | `TableEngine` | Result table assembly — formats computed statistics into polished DataFrames |

### Design Principles

- **Single entry point:** Every researchpy function calls `SyntaxParser.from_args()` (or a subclass
  override) first. No function parses user input on its own.
- **Subclass extensibility:** Domain-specific specs (e.g., `AnovaSpec`, `TTestSpec`) subclass
  `SyntaxParser` and override `from_args()` for validation while inheriting the full parsing logic.
- **Separation of concerns:** Parsing, matrix construction, and table formatting are completely
  independent. Each layer consumes the output of the previous layer and knows nothing about the
  layers above it.
- **`formulaic` backbone:** All formula parsing and design matrix construction delegates to the
  `formulaic` package. Researchpy wraps the results in its own container classes (`ModelTerms`,
  `Term`, `TableSpec`, etc.).

### Dependencies on Existing Classes

| Class | Source | Used By |
|-------|--------|---------|
| `CoreDataclass` | `containers/base.py` | All engine classes inherit from it |
| `ModelTerms` / `Term` | `containers/multivariable.py` | `SyntaxParser` (formula parsing), `MatrixEngine` (term metadata) |
| `ModelDesignSpec` | `containers/multivariable.py` | Model-level consumers of `MatrixEngine` output |
| `Family` (+ subclasses) | `models/families.py` | GLM/regression consumers that operate on `MatrixEngine` output |
| `formulaic.Formula` | `formulaic` package | `MatrixEngine.from_formula()` |

---

## Layer 1: SyntaxParser (`engine/syntax.py`)

### Purpose

The **universal input parsing layer**. Takes any supported calling convention and normalizes it
into a single `SyntaxParser` dataclass instance. All downstream computation operates exclusively
on the fields of this dataclass.

### Class: `SyntaxParser`

```python
@dataclass
class SyntaxParser(CoreDataclass):
    DV: List[str]                          # Dependent variable column name(s)
    IV: Optional[List[str]]                # Independent variables (marginal computation)
    by: Optional[List[str]]                # Row grouping variable(s) (cell means / pivot rows)
    over: Optional[List[str]]              # Column grouping variable(s) (pivot columns)
    data: Optional[pd.DataFrame]           # Source DataFrame
    formula: Optional[str]                 # Canonical formula string
    weights: Optional[str]                 # Weight column name
    sub_specs: Optional[List[TableTermSpec]]  # Per-term specs for mixed formulas
```

**Constraints enforced in `__post_init__`:**

- `IV` is mutually exclusive with `by` / `over`. You use one approach or the other.
- `over` requires `by` to also be specified.

### Factory Method: `SyntaxParser.from_args()`

```python
@classmethod
def from_args(
    cls,
    arg1=None,          # positional: DataFrame, Series, str, list[str], ndarray, list, tuple
    arg2=None,          # positional: DataFrame (when arg1 is formula or column list)
    /,
    *,
    dv=None,            # keyword: str or list[str]
    iv=None,            # keyword: str or list[str]
    by=None,            # keyword: str or list[str]
    over=None,          # keyword: str or list[str]
    data=None,          # keyword: pd.DataFrame
    weights=None,       # keyword: str
) -> "SyntaxParser":
```

### Supported Calling Conventions

There are **six calling conventions**, each mapping to a specific resolution path inside
`from_args()`. The method auto-detects which convention is being used based on argument types.

---

#### Convention 1: DataFrame / Array (no groups)

Pass a DataFrame directly. Each column becomes a DV.

```python
# Single column
spec = SyntaxParser.from_args(df[['y']])
# spec.DV = ['y'], spec.by = None, spec.IV = None

# Multiple columns — compute for each
spec = SyntaxParser.from_args(df[['y', 'z']])
# spec.DV = ['y', 'z'], spec.by = None

# Raw ndarray (1-D)
spec = SyntaxParser.from_args(np.array([1, 2, 3, 4]))
# spec.DV = ['value'], spec.data = DataFrame({'value': [1,2,3,4]})

# Raw ndarray (2-D)
spec = SyntaxParser.from_args(np.array([[1, 2], [3, 4]]))
# spec.DV = ['col_0', 'col_1']
```

---

#### Convention 2: Column Name List + DataFrame

Pass a list of column name strings and a DataFrame.

```python
spec = SyntaxParser.from_args(["y", "k", "c"], df)
# spec.DV = ['y', 'k', 'c'], spec.data = df
```

---

#### Convention 3: Series (no groups)

Pass a single pandas Series.

```python
spec = SyntaxParser.from_args(df['y'])
# spec.DV = ['y'], spec.data = DataFrame({'y': ...})
```

---

#### Convention 4: Formula String + DataFrame

Pass a Wilkinson-style formula string and a DataFrame.

```python
# Single grouping variable → by
spec = SyntaxParser.from_args("y ~ C(x)", df)
# spec.DV = ['y'], spec.by = ['x']

# Multiple main effects → marginal (IV)
spec = SyntaxParser.from_args("y ~ C(x) + C(k)", df)
# spec.DV = ['y'], spec.IV = ['x', 'k']
```

---

#### Convention 5: Explicit Keywords

Pass named keyword arguments.

```python
# Marginal (stacked results for each IV independently)
spec = SyntaxParser.from_args(dv="y", iv=["x", "k"], data=df)
# spec.DV = ['y'], spec.IV = ['x', 'k']
# spec.formula = "y ~ C(x) + C(k)"

# Cell grouping
spec = SyntaxParser.from_args(dv="y", by="x", data=df)
# spec.DV = ['y'], spec.by = ['x']
# spec.formula = "y ~ C(x)"

# Pivot layout
spec = SyntaxParser.from_args(dv="y", by="x", over="k", data=df)
# spec.DV = ['y'], spec.by = ['x'], spec.over = ['k']
# spec.formula = "y ~ C(x)*C(k)"
```

---

#### Convention 6: Formula with Operators (`:`, `*`, mixed)

```python
# Cell means (interaction → by)
spec = SyntaxParser.from_args("y ~ C(x):C(k)", df)
# spec.DV = ['y'], spec.by = ['x', 'k']

# Pivot (star expansion → by + over)
spec = SyntaxParser.from_args("y ~ C(x)*C(k)", df)
# spec.DV = ['y'], spec.by = ['x'], spec.over = ['k']

# Mixed formula (main effects + non-star interactions → sub_specs)
spec = SyntaxParser.from_args("y ~ C(x) + C(k):C(z)", data=df)
# spec.DV = ['y'], spec.sub_specs = [
#     TableTermSpec(term='C(x)', name='x', layout='iv', variables=['x']),
#     TableTermSpec(term='C(k):C(z)', name='k:z', layout='by', variables=['k', 'z']),
# ]
```

### Formula Operator Semantics

| Operator | Layout | Description | Example |
|----------|--------|-------------|---------|
| `+` | Marginal | Compute separately for each factor, stack results | `y ~ C(x) + C(k)` |
| `:` | Cell | Compute for each unique combination, MultiIndex rows | `y ~ C(x):C(k)` |
| `*` | Pivot | First factor → rows, second → columns | `y ~ C(x)*C(k)` |

### Reverse Mapping: Keywords → Formula String

The internal function `_args_to_formula()` converts keyword arguments to a canonical formula:

| Keywords | Generated Formula |
|----------|-------------------|
| `dv="y", iv=["x","k"]` | `"y ~ C(x) + C(k)"` |
| `dv="y", by=["x"]` | `"y ~ C(x)"` |
| `dv="y", by=["x","k"]` | `"y ~ C(x):C(k)"` |
| `dv="y", by=["x"], over=["k"]` | `"y ~ C(x)*C(k)"` |

### Formula Parsing Logic (`_parse_formula`)

When a formula string is provided, `_parse_formula()` uses `ModelTerms.from_formula()` to
structurally parse it, then categorizes the RHS terms:

```
RHS terms
├── main_effect_terms (no ":" in term)
└── interaction_terms (contains ":")

Decision tree:
├── Only main effects?
│   ├── 1 term → by (single cell grouping)
│   └── 2+ terms → IV (marginal / stacked)
├── Only interactions?
│   └── → by (cell means, all component vars collected)
└── Both main effects AND interactions?
    ├── Star expansion? (main names == interaction components)
    │   └── → by + over (pivot)
    └── Not star expansion?
        └── → sub_specs (mixed, each term gets TableTermSpec)
```

### Conversion Utilities

```python
# Strip all C() wrappers → treat as continuous
as_continuous("y ~ C(x) + C(k)")     # → "y ~ x + k"

# Wrap bare variable names in C() → treat as categorical
as_categorical("y ~ x + k")          # → "y ~ C(x) + C(k)"

# Smart wrapping (only non-numeric columns wrapped when data provided)
as_categorical("y ~ x + age", data=df)  # → "y ~ C(x) + age"  (if age is numeric)
```

### Validation

- `_validate_columns(columns, data, label)` checks all column names exist in the DataFrame,
  raises `ValueError` with available columns listed if any are missing.
- Called for DV, IV, by, over, and all RHS formula variables.

---

## Layer 2: MatrixEngine (`engine/matrix.py`)

### Purpose

The **design matrix construction layer**. Takes the output of `SyntaxParser` and builds
the numerical LHS/RHS matrices that regression models and computation routines consume.
This is the **single home** for all `formulaic.Formula.get_model_matrix` calls.

### Class: `MatrixEngine`

```python
@dataclass
class MatrixEngine(CoreDataclass):
    DV: Optional[Any]                      # LHS design matrix (formulaic.ModelMatrix)
    IV: Optional[Any]                      # RHS design matrix (formulaic.ModelMatrix)
    model_terms: Optional[Dict[str, ModelTerms]]  # {"dv": ModelTerms, "iv": ModelTerms}
    formula: Optional[str]                 # Formula string used to build matrices
    n: Optional[int]                       # Number of observations (auto-computed)
    k: Optional[int]                       # Number of predictors (auto-computed)
```

**Auto-population in `__post_init__`:** `n` and `k` are automatically computed from array shapes
if not explicitly provided.

### Factory Methods

#### `MatrixEngine.from_formula(formula, data, ...)`

Build directly from a formula string and DataFrame. This is where all `formulaic` interaction
happens.

```python
engine = MatrixEngine.from_formula("y ~ x1 + x2 + C(group)", df)

engine.DV           # formulaic.ModelMatrix (LHS) — shape (n, 1)
engine.IV           # formulaic.ModelMatrix (RHS) — shape (n, k)
engine.model_terms  # {"dv": ModelTerms, "iv": ModelTerms}
engine.n            # 100
engine.k            # 4 (Intercept + x1 + x2 + group[T.b])
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `formula` | *(required)* | Wilkinson-style formula string |
| `data` | *(required)* | pandas DataFrame |
| `output` | `"numpy"` | Output format: `"numpy"`, `"pandas"`, `"sparse"` |
| `include_intercept` | `True` | Include intercept column |
| `ensure_full_rank` | `True` | Drop aliased columns for full rank |

#### `MatrixEngine.from_spec(spec, ...)`

Build from a `SyntaxParser` instance. Delegates to `from_formula()` using the spec's
`formula` and `data` attributes.

```python
spec = SyntaxParser.from_args("y ~ x1 + C(group)", df)
engine = MatrixEngine.from_spec(spec)
```

Raises `ValueError` if the spec has no formula or no data.

### Computational Helpers

These methods provide common linear algebra operations on the design matrices:

```python
engine = MatrixEngine.from_formula("y ~ x1 + x2", df)

# Hat matrix: H = X(X'X)⁻¹X'
H = engine.hat_matrix()          # shape (n, n)

# J matrix: n×n matrix of ones
J = engine.j_matrix()            # shape (n, n)
J = engine.j_matrix(n=50)        # shape (50, 50)

# Identity matrix
I = engine.identity_matrix()     # shape (n, n)

# Eigenvalues of X'X
eig = engine.eigenvalues()       # shape (k,)
```

The hat matrix computation falls back to the Moore-Penrose pseudoinverse (`np.linalg.pinv`) if
`X'X` is singular.

### ModelTerms Integration

The `model_terms` dict provides rich term metadata for both sides of the formula:

```python
engine = MatrixEngine.from_formula("y ~ x1 + C(drug)", df)

iv_terms = engine.model_terms["iv"]  # ModelTerms instance

# Iterate terms
for term in iv_terms.terms:
    print(f"{term.term} → {term.name}")
    print(f"  is_factor={term.is_factor}")
    print(f"  is_interaction={term.is_interaction}")
    print(f"  columns={term.columns}")
    print(f"  columns_cleaned={term.columns_cleaned}")
    print(f"  levels={term.levels}")
    print(f"  reference={term.reference}")

# Output:
# Intercept → Intercept
#   is_factor=False, is_interaction=False
# x1 → x1
#   is_factor=False, is_interaction=False
# C(drug) → drug
#   is_factor=True, is_interaction=False
#   columns=['C(drug)[T.2]', 'C(drug)[T.3]']
#   columns_cleaned=['2', '3']
#   levels=['1', '2', '3']
#   reference='1'
```

---

## Layer 3: TableEngine (`engine/table.py`)

### Purpose

The **result table construction layer**. Takes computed statistics (from the `statistics/`
submodule, model fitting, or direct computation) and assembles polished pandas DataFrames
for the user.

### Data Specifications

#### `TableTermSpec`

Per-term layout metadata for mixed formulas. Created by `_build_sub_specs()` when the syntax
parser encounters a formula with both main effects and non-star interactions.

```python
@dataclass
class TableTermSpec(CoreDataclass):
    term: str            # Raw formula term, e.g., "C(x)" or "C(k):C(z)"
    name: str            # Cleaned name, e.g., "x" or "k:z"
    layout: str          # "iv" (marginal) or "by" (cell)
    variables: List[str] # Grouping variable names, e.g., ["k", "z"]
```

**Example for mixed formula** `"y ~ C(x) + C(k):C(z)"`:

```python
sub_specs = [
    TableTermSpec(term="C(x)",      name="x",   layout="iv", variables=["x"]),
    TableTermSpec(term="C(k):C(z)", name="k:z", layout="by", variables=["k", "z"]),
]
```

#### `TableSpec`

Overall table layout specification. Controls column ordering, rounding, and metadata.

```python
@dataclass
class TableSpec(CoreDataclass):
    columns: List[str]              # Column headers, e.g., ["N", "Mean", "SD"]
    rows: List[str]                 # Row grouping variables
    variables: List[str]            # DVs being summarized
    statistics: List[str]           # Statistics to compute
    title: Optional[str]            # Table title
    footnotes: Optional[List[str]]  # Footnotes
    decimals: int = 4               # Decimal places for rounding
```

### Class: `TableEngine`

```python
@dataclass
class TableEngine(CoreDataclass):
    spec: TableSpec  # Layout specification
```

### Methods

#### `build(rows, index_cols=None) → pd.DataFrame`

The core builder. Takes a list of row dictionaries and produces a formatted DataFrame.

```python
spec = TableSpec(
    columns=["Group", "N", "Mean", "SD"],
    decimals=2,
)
engine = TableEngine(spec=spec)

rows = [
    {"Group": "Control",   "N": 50, "Mean": 3.456789, "SD": 1.234567},
    {"Group": "Treatment", "N": 48, "Mean": 5.678901, "SD": 1.345678},
]

result = engine.build(rows)
#    Group   N  Mean   SD
# 0  Control  50  3.46  1.23
# 1  Treatment 48  5.68  1.35

# With index
result = engine.build(rows, index_cols=["Group"])
#            N  Mean   SD
# Group
# Control   50  3.46  1.23
# Treatment 48  5.68  1.35
```

**Behavior:**
1. Constructs DataFrame from row dicts
2. Reorders columns to match `spec.columns` (extras appended at end)
3. Rounds all numeric columns to `spec.decimals` places
4. Optionally sets index columns

#### `stack(tables, labels=None, label_column="Term") → pd.DataFrame`

Vertically stacks multiple result DataFrames. Used for marginal (`+`) and mixed-formula
layouts where each term produces its own table.

```python
table_x = pd.DataFrame([
    {"Level": "a", "N": 30, "Mean": 1.5},
    {"Level": "b", "N": 35, "Mean": 2.3},
])
table_k = pd.DataFrame([
    {"Level": "m", "N": 32, "Mean": 1.8},
    {"Level": "f", "N": 33, "Mean": 2.0},
])

result = engine.stack(
    [table_x, table_k],
    labels=["x", "k"],
    label_column="Variable",
)
#   Variable Level   N  Mean
# 0        x     a  30   1.5
# 1        x     b  35   2.3
# 2        k     m  32   1.8
# 3        k     f  33   2.0
```

#### `pivot(df, row_vars, col_vars, value_col) → pd.DataFrame`

Reshapes a long-form result table into a pivot layout. Used for star-expansion (`*`)
formulas.

```python
long_df = pd.DataFrame([
    {"x": "a", "k": "m", "Mean": 1.2},
    {"x": "a", "k": "f", "Mean": 1.5},
    {"x": "b", "k": "m", "Mean": 2.1},
    {"x": "b", "k": "f", "Mean": 2.4},
])

result = engine.pivot(long_df, row_vars=["x"], col_vars=["k"], value_col="Mean")
# k      f    m
# x
# a    1.5  1.2
# b    2.4  2.1
```

---

## Full Pipeline Examples

### Example 1: Descriptive Statistics (Marginal)

```python
import pandas as pd
from researchpy.engine import SyntaxParser, TableEngine, TableSpec

df = pd.DataFrame({
    "score": [85, 90, 78, 92, 88, 76, 95, 82],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
})

# Step 1: Parse input
spec = SyntaxParser.from_args(dv="score", iv=["group", "gender"], data=df)
# spec.DV = ['score'], spec.IV = ['group', 'gender']
# spec.formula = "score ~ C(group) + C(gender)"

# Step 2: Compute statistics (done by statistics/ submodule)
# ... computation produces row dicts ...

# Step 3: Format results
table_spec = TableSpec(columns=["Variable", "Level", "N", "Mean", "SD"], decimals=2)
engine = TableEngine(spec=table_spec)
result = engine.build(computed_rows)
```

### Example 2: Regression Model

```python
from researchpy.engine import SyntaxParser, MatrixEngine

df = pd.DataFrame({
    "y": [1.2, 2.3, 3.1, 4.5, 5.2],
    "x1": [1, 2, 3, 4, 5],
    "x2": [2.1, 3.2, 2.8, 4.1, 3.9],
})

# Step 1: Parse input
spec = SyntaxParser.from_args("y ~ x1 + x2", df)

# Step 2: Build design matrices
engine = MatrixEngine.from_spec(spec)
# engine.DV  → (5, 1) array
# engine.IV  → (5, 3) array (Intercept + x1 + x2)
# engine.n = 5, engine.k = 3

# Step 3: Use in model fitting
# beta = np.linalg.lstsq(engine.IV, engine.DV)[0]
# H = engine.hat_matrix()
```

### Example 3: Subclassing SyntaxParser

```python
from researchpy.engine.syntax import SyntaxParser

class AnovaSpec(SyntaxParser):
    """Domain-specific spec for ANOVA that enforces categorical IVs."""

    @classmethod
    def from_args(cls, *args, **kwargs):
        # Delegate parsing to parent
        spec = super().from_args(*args, **kwargs)

        # Domain-specific validation
        if spec.IV is not None:
            for var in spec.IV:
                if pd.api.types.is_numeric_dtype(spec.data[var]):
                    raise ValueError(
                        f"ANOVA requires categorical grouping variables. "
                        f"'{var}' is numeric. Use C({var}) to specify as categorical."
                    )

        return spec
```

### Example 4: Pivot Table (Star Expansion)

```python
spec = SyntaxParser.from_args("score ~ C(group)*C(gender)", df)
# spec.by = ['group'], spec.over = ['gender']

# After computing group×gender cell means:
long_results = pd.DataFrame([
    {"group": "A", "gender": "M", "Mean": 81.5},
    {"group": "A", "gender": "F", "Mean": 91.0},
    {"group": "B", "gender": "M", "Mean": 82.0},
    {"group": "B", "gender": "F", "Mean": 79.0},
])

engine = TableEngine(spec=TableSpec(decimals=1))
pivot_result = engine.pivot(long_results, row_vars=["group"], col_vars=["gender"], value_col="Mean")
# gender     F     M
# group
# A       91.0  81.5
# B       79.0  82.0
```

---

## Module-Level Exports

`researchpy/engine/__init__.py` exports:

```python
__all__ = [
    "SyntaxParser",      # Layer 1: input parsing
    "MatrixEngine",      # Layer 2: design matrix construction
    "TableEngine",       # Layer 3: result table assembly
    "TableSpec",         # Table layout specification
    "TableTermSpec",     # Per-term layout metadata
    "as_continuous",     # Utility: strip C() wrappers
    "as_categorical",    # Utility: add C() wrappers
]
```

---

## File Structure

```
researchpy/engine/
├── __init__.py      # Package exports (SyntaxParser, MatrixEngine, TableEngine, etc.)
├── syntax.py        # Layer 1: SyntaxParser, _parse_formula, _args_to_formula,
│                    #          _is_star_expansion, _build_sub_specs,
│                    #          as_continuous, as_categorical, _validate_columns
├── matrix.py        # Layer 2: MatrixEngine (from_formula, from_spec,
│                    #          hat_matrix, j_matrix, identity_matrix, eigenvalues)
└── table.py         # Layer 3: TableTermSpec, TableSpec, TableEngine
                     #          (build, stack, pivot)
```

---

## Internal Functions Reference

### `engine/syntax.py`

| Function | Visibility | Purpose |
|----------|------------|---------|
| `SyntaxParser.from_args()` | Public | Universal input gate — resolves any calling convention |
| `_args_to_formula()` | Private | Converts keyword args (dv, iv, by, over) → formula string |
| `_parse_formula()` | Private | Parses formula string → `SyntaxParser` via `ModelTerms.from_formula()` |
| `_is_star_expansion()` | Private | Detects if main effects + interactions form a star expansion |
| `_build_sub_specs()` | Private | Builds `TableTermSpec` list for mixed formulas |
| `as_continuous()` | Public | Strips `C()` wrappers from formula |
| `as_categorical()` | Public | Adds `C()` wrappers to bare variable names |
| `_validate_columns()` | Private | Checks column names exist in DataFrame |

### `engine/matrix.py`

| Method | Visibility | Purpose |
|--------|------------|---------|
| `MatrixEngine.from_formula()` | Public | Builds design matrices from formula + data via `formulaic` |
| `MatrixEngine.from_spec()` | Public | Builds from a `SyntaxParser` instance |
| `MatrixEngine.hat_matrix()` | Public | H = X(X'X)⁻¹X' with pseudoinverse fallback |
| `MatrixEngine.j_matrix()` | Public | n×n ones matrix |
| `MatrixEngine.identity_matrix()` | Public | n×n identity matrix |
| `MatrixEngine.eigenvalues()` | Public | Eigenvalues of X'X |

### `engine/table.py`

| Method | Visibility | Purpose |
|--------|------------|---------|
| `TableEngine.build()` | Public | Assembles DataFrame from row dicts, reorders columns, rounds |
| `TableEngine.stack()` | Public | Vertically concatenates sub-tables with optional labels |
| `TableEngine.pivot()` | Public | Reshapes long-form results into pivot layout |

---

## Error Handling

All layers raise descriptive errors with actionable messages:

| Error | When | Message Pattern |
|-------|------|-----------------|
| `ValueError` | `iv` used with `by`/`over` | *"Cannot use 'IV' together with 'by' or 'over'..."* |
| `ValueError` | `over` without `by` | *"'over' requires 'by' to also be specified..."* |
| `ValueError` | No data provided | *"Formula '...' requires a DataFrame..."* |
| `ValueError` | Column not in DataFrame | *"Column(s) [...] specified for ... not found..."* |
| `ValueError` | `from_spec` with no formula | *"SyntaxParser spec has no formula..."* |
| `TypeError` | Unsupported arg1 type | *"Unsupported type for first argument: ..."* |
