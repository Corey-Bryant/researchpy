# ResearchPy Architecture Update: Final Aligned Plan

## Decisions Locked
| Decision | Detail |
|---|---|
| Matrix as universal substrate | `build_indicator_matrix` for ALL group membership. Linear stats: matrix arithmetic. Non-linear stats: matrix boolean selection + `scalar_func`. |
| Return-type contract | float or DataFrame. No dataclasses from stat functions. |
| Four canonical formula patterns | `y ~ C(x)`, `y ~ C(x) + C(k)`, `y ~ C(x):C(k)`, `y ~ C(x)*C(k)`. Everything else (except mixed) gets a helpful ValueError. |
| Mixed formulas | `y ~ C(x) + C(k):C(z)` returns stacked DataFrame with `Term` column (Option B). |
| `*` semantics | Two-factor -> pivot. 3+ factors -> ValueError directing to keyword args. |
| Dropped conventions | Column-list positional (#3) and GroupBy resolution. |
| `pretty_format` parameter | Default `True`. Affects ONLY the `Term` column in mixed formulas. Cleaned names (`x`, `k:z`) when True, raw (`C(x)`, `C(k):C(z)`) when False. Consistent with regression models. |
| Quartiles/percentile grouped output | Wide format (columns per value: `Q1`, `Q2`, `Q3` / `P10`, `P25`, etc.). |
| `describe()` | Comprehensive descriptive stats function. Builds indicator matrix once, runs all stats through it. |
| Formulaic | Replaces Patsy for new API only. Old API stays as-is (deprecated). |

## Phase 1: Simplify resolve() (core/spec.py)
- Convention 3 (column-list positional): `mean(["y", "k"], df)` is gone.
- `_resolve_groupby()` and GroupBy `isinstance` checks: removed entirely.

### Add Validation

```python
group_vars = (iv or []) + (by or []) + (over or [])
overlap = set(dv) & set(group_vars)
if overlap:
    raise ValueError(
        f"Variable(s) {list(overlap)} appear in both dv and grouping variables. "
        f"A variable cannot be both a dependent variable and a grouping variable."
    )
```

### Rewrite _parse_formula() to handle exactly these patterns:

| Formula |	Spec Layout | Detection Logic |
|---|---|---|
| y ~ C(x) |	by=["x"] | 1 term, not interaction |
| y ~ C(x) + C(k) |	iv=["x", "k"] | 2+ terms, all main effects, no interactions |
| y ~ C(x):C(k) |	by=["x", "k"] | 1 term, interaction |
| y ~ C(x)*C(k) |	by=["x"], over=["k"] | star expansion: main effects + 1 interaction, components match |
| y ~ C(x) + C(k):C(z) |	mixed → sub_specs | main effects AND interactions, not pure star |
| Anything else | ValueError with guidance | 3-way *, nested, etc. |

### Mixed formula handling:

The formula decomposes into constituent terms. Each term becomes a TermSpec. Results stack into a single DataFrame with a Term column.

**Conceptual output for: rp.mean("y ~ C(x) + C(k):C(z)", df, pretty_format= True)**
```
#   Term   Level       Mean
# 0    x       a       1.50
# 1    x       b       3.50
# 2    x       c       5.50
# 3  k:z  low:yes       2.00
# 4  k:z   low:no       4.00
# 5  k:z high:yes       6.00
# 6  k:z  high:no       8.00
```

**With pretty_format= False:**
```
#       Term       Level       Mean
# 0     C(x)           a       1.50
# 1     C(x)           b       3.50
# 2     C(x)           c       5.50
# 3  C(k):C(z)   low:yes       2.00
# 4  C(k):C(z)    low:no       4.00
# 5  C(k):C(z)  high:yes       6.00
# 6  C(k):C(z)   high:no       8.00
```

### Star expansion for 3+ factors raises:

```python
raise ValueError(
    "Formula 'y ~ C(x)*C(k)*C(z)' expands to too many terms for a single "
    "descriptive output. Use keyword arguments for explicit control:\n"
    "    rp.mean(dv='y', by=['x','k'], over=['z'], data=df)"
)
```

### New dataclasses:

```python
@dataclass
class TermSpec:
    """One term extracted from a mixed formula."""
    term_name: str          # cleaned name, e.g. "x" or "k:z"
    term_raw: str           # raw name, e.g. "C(x)" or "C(k):C(z)"
    layout: str             # "iv" or "by"
    variables: List[str]    # grouping variables for this term

@dataclass
class ComputeSpec:
    dv: List[str] = field(default_factory=list)
    iv: Optional[List[str]] = None
    by: Optional[List[str]] = None
    over: Optional[List[str]] = None
    data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None
    weights: Optional[str] = None
    sub_specs: Optional[List[TermSpec]] = None  # NEW: for mixed formulas
```
When `sub_specs` is not None, `_route_computation` iterates each sub-spec, computes independently, tags with Term column, and stacks.


## Phase 2: Unify Computation (statistics/_compute.py)

### Remove:

* `fallback_grouped_func` parameter from `_route_computation`, `_compute_marginal`, `_compute_cell`, `_compute_pivot`.
* `_grouped_via_iteration` function entirely.

### Add `_compute_grouped_flat()`:

```python
def _compute_grouped_flat(
    data: pd.DataFrame,
    dv_col: str,
    groups: List[str],
    *,
    scalar_func: Callable[[np.ndarray], float],
    matrix_stat: Optional[str] = None,
    decimals: int = 4,
    stat_label: str = "Value",
) -> pd.DataFrame:
    """Single entry point for grouped computation.

    Uses build_indicator_matrix for group membership.
    - matrix_stat is not None: linear path (matrix arithmetic)
    - matrix_stat is None: non-linear path (matrix boolean selection + scalar_func)
    """
    X, cell_labels, cell_components = build_indicator_matrix(data, groups)
    values = data[dv_col].to_numpy(dtype=np.float64)
    nan_mask = np.isnan(values)

    if matrix_stat is not None:
        result_df = grouped_statistic(data, dv=dv_col, groups=groups,
                                       stat_func=matrix_stat)
        result_df[result_df.columns[-1]] = result_df[result_df.columns[-1]].round(decimals)
        return result_df

    results = []
    for j in range(X.shape[1]):
        mask = X[:, j].astype(bool) & ~nan_mask
        group_values = values[mask]
        value = round(float(scalar_func(group_values)), decimals) if len(group_values) > 0 else np.nan

        row = {}
        for i, group_name in enumerate(groups):
            row[group_name] = cell_components[j][i]
        row[stat_label] = value
        results.append(row)

    return pd.DataFrame(results)
```

### Updated _route_computation signature (no fallback_grouped_func):

```python
def _route_computation(arg1, arg2, *,
    dv=None, iv=None, by=None, over=None, data=None,
    scalar_func: Callable,
    matrix_stat: Optional[str] = None,
    decimals: int = 4,
    weights: Optional[str] = None,
    stat_label: str = "Value",
    pretty_format: bool = True,
    **kwargs,
) -> Union[float, pd.DataFrame]:
```

No more `fallback_grouped_func`. The three layout functions (`_compute_marginal`, `_compute_cell`, `_compute_pivot`) 
each call `_compute_grouped_flat` and reshape. No more internal matrix/fallback branching; that logic lives in one place.

### Mixed-formula routing added to _route_computation:

```python
if spec.sub_specs is not None:
    frames = []
    for ts in spec.sub_specs:
        if ts.layout == "iv":
            sub_spec = ComputeSpec(dv=spec.dv, iv=ts.variables, data=spec.data)
        else:
            sub_spec = ComputeSpec(dv=spec.dv, by=ts.variables, data=spec.data)

        sub_result = _route_computation(
            None, None,
            dv=sub_spec.dv, iv=sub_spec.iv, by=sub_spec.by,
            data=sub_spec.data,
            scalar_func=scalar_func, matrix_stat=matrix_stat,
            decimals=decimals, stat_label=stat_label,
            pretty_format=pretty_format, **kwargs,
        )
        term_label = ts.term_name if pretty_format else ts.term_raw
        sub_result.insert(0, "Term", term_label)
        frames.append(sub_result)

    return pd.concat(frames, ignore_index=True)
```

## Phase 3: Migrate to Formulaic (core/spec.py + containers/multivariable.py)
**Scope**: New API path only. Old published API keeps Patsy as-is.

**Change**: Replace `ModelTerms.from_formula()` internals (currently `patsy.ModelDesc.from_formula()`) with 
`formulaic.Formula.parse()`. The Term and ModelTerms dataclass interfaces stay identical (`term.name`, `term.is_interaction`, 
`term.is_factor`). Downstream code in `_parse_formula` unchanged.


## Phase 4: Extend Routing + describe() (statistics/central_tendency.py)

### Stat Function Updates

| Function | scalar_func | matrix_stat | Grouped? | Notes |
|---------|---------|-----------|------------|--------|
| mean()   | numpy.nanmean | "mean" | Yes | Already routed. Simplify signature (drop fallback_grouped_func). |
| median() | numpy.nanmedian | None | Yes | Already routed. Remove _median_grouped inner function. |
| quartiles() | special (returns 3 values) | None | NEW | Wide output: Q1, Q2, Q3 columns per group. |
| percentile() | special (parametric) | None | NEW | Wide output: P10, P25, etc. columns per group. |
| iqr() | Q3-Q1 | None | NEW | Standard scalar per group. |
| mode() | -- | -- | Deferred | Multimodal output complicates stacking. Lower priority. |

### Quartiles Grouped Output (Wide)
| | disease | Q1 | Q2 | Q3 |
|---|---------|----|----|----|
| 0 | a | 1.5 | 2.5 | 3.5 |
| 1 | b | 4.5 | 5.5 | 6.5 |

### Percentile Grouped Output (Wide)
| | disease | P10 | P25 | P50 | P75 | P90 |
|---|---------|----|----|----|----|----|
| 0 | a | 1.2 | 1.5 | 2.5 | 3.5 | 3.8 |
| 1 | b | 4.2 | 4.5 | 5.5 | 6.5 | 6.8 |

### describe() function:

Comprehensive descriptive statistics. Builds indicator matrix once, runs all stats through it.

```python
# Bare array
rp.describe(df["systolic"])

# Formula
rp.describe("systolic ~ C(disease)", df)
rp.describe("systolic ~ C(disease) + C(drug)", df)
rp.describe("systolic ~ C(disease):C(drug)", df)
rp.describe("systolic ~ C(disease)*C(drug)", df)

# Keywords
rp.describe(dv="systolic", by="disease", data=df)
rp.describe(dv="systolic", iv=["disease", "drug"], data=df)
rp.describe(dv="systolic", by="disease", over="drug", data=df)
rp.describe(dv=["systolic", "diastolic"], by="disease", data=df)
```

**Output for `rp.describe(dv="systolic", by="disease", data=df)`:**

| | disease | N | N Missing | Mean | Median | SD | Min | Q1 | Q3 | Max | IQR |
|---|---------|---|-----------|------|--------|----|-----|----|----|-----|-----|
| 0 | a | 50 | 2 | 120.5 | 121.0 | 8.3 | 98.0 | 115.0 | 127.0 | 142.0 | 12.0 |
| 1 | b | 48 | 4 | 118.2 | 119.0 | 7.9 | 95.0 | 113.0 | 125.0 | 139.0 | 12.0 |
Implementation: builds indicator matrix once via build_indicator_matrix, then runs each stat (mean via matrix arithmetic, 
median/SD/min/max/Q1/Q3/IQR/N via boolean selection) through the same matrix. No redundant matrix construction.


## Phase 5: Public API Surface (statistics/__init__.py)

Three input methods, each unambiguous:

```python
# 1. Bare array -> scalar (or wide DataFrame for multi-stat functions)
rp.mean(df["systolic"])
rp.describe(df["systolic"])

# 2. Formula -> grouped (four canonical patterns + mixed)
rp.mean("systolic ~ C(disease)", df)
rp.mean("systolic ~ C(disease) + C(drug)", df)        # marginal, stacked
rp.mean("systolic ~ C(disease):C(drug)", df)          # cell means
rp.mean("systolic ~ C(disease)*C(drug)", df)          # pivot table
rp.mean("systolic ~ C(disease) + C(drug):C(sex)", df) # mixed, Term column

# 3. Keywords -> explicit control
rp.mean(dv="systolic", by="disease", data=df)
rp.mean(dv="systolic", iv=["disease", "drug"], data=df)
rp.mean(dv="systolic", by="disease", over="drug", data=df)
rp.mean(dv=["systolic", "diastolic"], by="disease", data=df)
```
No column-list positional. No GroupBy objects. Clean.


# Implementation Order

| Step | File(s) | What Changes |
|------|---------|-------------|
| 1 | core/spec.py | Add TermSpec dataclass, add sub_specs to ComputeSpec, add DV overlap validation |
| 2 | core/spec.py | Rewrite _parse_formula with 4 patterns + mixed + errors, drop column-list + GroupBy |
| 3 | statistics/_compute.py | Add _compute_grouped_flat, remove _grouped_via_iteration, remove fallback_grouped_func from all signatures |
| 4 | statistics/_compute.py | Update _compute_marginal, _compute_cell, _compute_pivot to call _compute_grouped_flat |
| 5 | statistics/_compute.py | Add mixed-formula routing in _route_computation with pretty_format |
| 6 | statistics/central_tendency.py | Simplify mean and median signatures, add pretty_format param |
| 7 | statistics/central_tendency.py | Extend routing to quartiles, percentile, iqr (wide grouped output) |
| 8 | statistics/central_tendency.py | Implement describe() with single-matrix-all-stats pattern |
| 9 | containers/multivariable.py | Swap ModelTerms.from_formula internals from Patsy to formulaic (new API only) |

Steps 1-2 are foundational. Steps 3-5 are compute unification. Steps 6-8 extend the public surface. Step 9 is the dependency migration, kept for last.cation. Steps 6-8 extend the public surface. Step 9 is the dependency migration, kept for last.