# ResearchPy Architecture Update: Final Aligned Plan

## Decisions Locked
| Decision | Detail |
|---|---|
| Matrix for linear stats | `build_indicator_matrix` for mean, sum, count, variance, sd, se. Matrix arithmetic path. |
| `_build_cell_series` for non-linear stats | Median, quartiles, percentiles, IQR, etc. use `_build_cell_series` for group membership + scalar function iteration. No unnecessary matrix allocation. |
| Return-type contract | float or DataFrame. No dataclasses from stat functions. |
| Four canonical formula patterns | `y ~ C(x)`, `y ~ C(x) + C(k)`, `y ~ C(x):C(k)`, `y ~ C(x)*C(k)`. |
| Mixed formulas | `y ~ C(x) + C(k):C(z)` returns stacked DataFrame with `Term` column (Option B). |
| `*` semantics | Existing behavior preserved (no arbitrary factor limits). |
| GroupBy support | **Kept.** Existing `_resolve_groupby()` remains operational. |
| Column-list positional | **Kept.** `mean(["y", "k"], df)` continues to work. |
| Malformed formula handling | RHS terms not wrapped in `C()` raise informative `ValueError` with suggested fix. |
| Quartiles/percentile ungrouped | Returns single-row wide DataFrame (consistent with grouped output). |
| `describe()` vs `estable()` | `describe()` implemented now; reconciliation with `estable()` planned separately. |
| Multi-DV in `describe()` | Adds `"Variable"` column when multiple DVs (same pattern as `estable()`). |
| Formulaic | Already in use for new API (`ModelTerms.from_formula()`). No further migration step needed. |
| Patsy deprecation | 2-major-version deprecation cycle. Old API keeps Patsy as-is. |

## Phase 1: Simplify & Extend resolve() (core/spec.py)

### Keep Existing Conventions
- Convention 1 (Series/array) ✅
- Convention 2 (Formula) ✅
- Convention 3 (Column-list positional) ✅
- Convention 4 (Keywords) ✅
- GroupBy resolution ✅

### Add TermSpec and sub_specs for Mixed Formulas

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

When `sub_specs` is not None, `_route_computation` iterates each sub-spec, computes independently, tags with `Term` column, and stacks.

### Rewrite _parse_formula() to handle mixed patterns:

| Formula | Spec Layout | Detection Logic |
|---|---|---|
| y ~ C(x) | by=["x"] | 1 term, not interaction |
| y ~ C(x) + C(k) | iv=["x", "k"] | 2+ terms, all main effects, no interactions |
| y ~ C(x):C(k) | by=["x", "k"] | 1 term, interaction |
| y ~ C(x)*C(k) | by=["x"], over=["k"] | star expansion: main effects + 1 interaction, components match |
| y ~ C(x) + C(k):C(z) | mixed → sub_specs | main effects AND interactions, not pure star |

### Add Malformed Formula Validation

```python
# In _parse_formula(), after parsing RHS terms:
for term in mt.terms:
    for part in term.term.split(":"):
        if "C(" not in part and part != "Intercept":
            raise ValueError(
                f"Term '{part}' on the RHS is not wrapped in C(). "
                f"For descriptive statistics, grouping variables must be categorical. "
                f"Use: 'y ~ C({part})' to specify '{part}' as a grouping factor."
            )
```

### Mixed formula handling:

The formula decomposes into constituent terms. Each term becomes a TermSpec. Results stack into a single DataFrame with a Term column.

**Conceptual output for: rp.mean("y ~ C(x) + C(k):C(z)", df)**
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

### New Tests (interleaved with implementation):
- Mixed formula parsing → correct sub_specs generation
- Malformed formula → ValueError with helpful message
- Existing conventions (GroupBy, column-list, single formula, keywords) remain passing


## Phase 2: Unify Computation (statistics/_compute.py)

### Remove:

* `fallback_grouped_func` parameter from `_route_computation`, `_compute_marginal`, `_compute_cell`, `_compute_pivot`.
* `_grouped_via_iteration` function entirely (its logic absorbed into `_compute_grouped_flat`).

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

    Two paths based on stat type:
    - matrix_stat is not None: linear path (indicator matrix arithmetic via grouped_statistic)
    - matrix_stat is None: non-linear path (_build_cell_series for group membership + scalar_func iteration)
    """
    if matrix_stat is not None:
        # Linear path: matrix arithmetic
        result_df = grouped_statistic(data, dv=dv_col, groups=groups, stat_func=matrix_stat)
        result_df[result_df.columns[-1]] = result_df[result_df.columns[-1]].round(decimals)
        return result_df

    # Non-linear path: iterate over groups using _build_cell_series
    cell_series = _build_cell_series(data, groups)
    unique_cells = sorted(cell_series.unique())
    values = data[dv_col].to_numpy(dtype=np.float64)

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = values[mask]
        clean = arr[~np.isnan(arr)]
        value = round(float(scalar_func(clean)), decimals) if len(clean) > 0 else np.nan

        row = {}
        if len(groups) == 1:
            row[groups[0]] = cell
        else:
            parts = cell.split(":")
            for i, group_name in enumerate(groups):
                row[group_name] = parts[i]
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
    **kwargs,
) -> Union[float, pd.DataFrame]:
```

The three layout functions (`_compute_marginal`, `_compute_cell`, `_compute_pivot`) each call `_compute_grouped_flat` and reshape. No more internal matrix/fallback branching in each layout function.

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
            decimals=decimals, stat_label=stat_label, **kwargs,
        )
        sub_result.insert(0, "Term", ts.term_name)
        frames.append(sub_result)

    return pd.concat(frames, ignore_index=True)
```


## Phase 3: Extend Stat Functions (statistics/central_tendency.py)

### Stat Function Updates

| Function | scalar_func | matrix_stat | Grouped? | Notes |
|---------|---------|-----------|------------|--------|
| mean() | numpy.nanmean | "mean" | Yes | Simplify signature (drop fallback_grouped_func). |
| median() | numpy.nanmedian | None | Yes | Remove _median_grouped inner function, uses _compute_grouped_flat. |
| quartiles() | special (returns 3 values) | None | NEW | Wide output: Q1, Q2, Q3 columns per group. Ungrouped: single-row DataFrame. |
| percentile() | special (parametric) | None | NEW | Wide output: P10, P25, etc. columns per group. Ungrouped: single-row DataFrame. |
| iqr() | Q3-Q1 | None | NEW | Standard scalar per group (returns float ungrouped, DataFrame grouped). |
| mode() | -- | -- | Deferred | Multimodal output complicates stacking. Lower priority. |

### Quartiles Output

**Ungrouped:**
| | Q1 | Q2 | Q3 |
|---|----|----|---|
| 0 | 2.0 | 3.0 | 4.0 |

**Grouped:**
| | disease | Q1 | Q2 | Q3 |
|---|---------|----|----|----|
| 0 | a | 1.5 | 2.5 | 3.5 |
| 1 | b | 4.5 | 5.5 | 6.5 |

### Percentile Output

**Ungrouped (e.g., q=[10, 25, 50, 75, 90]):**
| | P10 | P25 | P50 | P75 | P90 |
|---|----|----|----|----|---|
| 0 | 1.2 | 1.5 | 2.5 | 3.5 | 3.8 |

**Grouped:**
| | disease | P10 | P25 | P50 | P75 | P90 |
|---|---------|----|----|----|----|----|
| 0 | a | 1.2 | 1.5 | 2.5 | 3.5 | 3.8 |
| 1 | b | 4.2 | 4.5 | 5.5 | 6.5 | 6.8 |


## Phase 4: Implement describe() (statistics/describe.py)

### describe() function:

Comprehensive descriptive statistics. Builds indicator matrix once for linear stats, uses `_build_cell_series` once for non-linear stats. Avoids redundant group-membership computation.

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

**Output for `rp.describe(dv=["systolic", "diastolic"], by="disease", data=df)` (multi-DV):**

| | disease | Variable | N | N Missing | Mean | Median | SD | Min | Q1 | Q3 | Max | IQR |
|---|---------|----------|---|-----------|------|--------|----|-----|----|----|-----|-----|
| 0 | a | systolic | 50 | 2 | 120.5 | 121.0 | 8.3 | 98.0 | 115.0 | 127.0 | 142.0 | 12.0 |
| 1 | a | diastolic | 50 | 1 | 78.2 | 79.0 | 5.1 | 62.0 | 75.0 | 82.0 | 96.0 | 7.0 |
| 2 | b | systolic | 48 | 4 | 118.2 | 119.0 | 7.9 | 95.0 | 113.0 | 125.0 | 139.0 | 12.0 |
| 3 | b | diastolic | 48 | 3 | 76.9 | 77.0 | 4.8 | 60.0 | 73.0 | 80.0 | 94.0 | 7.0 |

Implementation: builds indicator matrix once for linear stats (N, Mean, SD), uses `_build_cell_series` once for non-linear stats (Median, Q1, Q3, Min, Max, IQR). No redundant group-membership computation.


## Phase 5: Public API Surface (statistics/__init__.py)

Three input methods, each unambiguous:

```python
# 1. Bare array -> scalar (or single-row DataFrame for multi-value functions)
rp.mean(df["systolic"])              # float
rp.quartiles(df["systolic"])         # single-row DataFrame
rp.describe(df["systolic"])          # single-row DataFrame

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

# 4. Column-list positional (kept)
rp.mean(["systolic", "diastolic"], df)

# 5. GroupBy (kept)
rp.mean(df.groupby("disease")["systolic"])
```


## Implementation Order

| Step | File(s) | What Changes | Tests |
|------|---------|-------------|-------|
| 1 | core/spec.py | Add `TermSpec` dataclass, add `sub_specs` to `ComputeSpec` | Unit tests for TermSpec creation |
| 2 | core/spec.py | Extend `_parse_formula` with mixed-formula detection + malformed formula ValueError | Tests: mixed formula → correct sub_specs; bare variable → ValueError; existing patterns still pass |
| 3 | statistics/_compute.py | Add `_compute_grouped_flat`, remove `_grouped_via_iteration`, remove `fallback_grouped_func` from all signatures | Tests: grouped computation produces same results as before for mean, median |
| 4 | statistics/_compute.py | Update `_compute_marginal`, `_compute_cell`, `_compute_pivot` to call `_compute_grouped_flat` | Integration tests: all calling conventions produce expected output |
| 5 | statistics/_compute.py | Add mixed-formula routing in `_route_computation` (sub_specs iteration) | Tests: mixed formula end-to-end produces Term-tagged stacked DataFrame |
| 6 | statistics/central_tendency.py | Simplify `mean` and `median` signatures (remove fallback_grouped_func usage) | Existing tests still pass |
| 7 | statistics/central_tendency.py | Extend `quartiles`, `percentile`, `iqr` to support grouped calling conventions + wide DataFrame output (ungrouped returns single-row DataFrame) | Tests: scalar, grouped, formula patterns for each; validate against numpy |
| 8 | statistics/describe.py (new) | Implement `describe()` with single-matrix + single-cell-series pattern, multi-DV "Variable" column | Tests: all calling conventions; validate against estable() output; multi-DV |

Steps 1-2 are foundational (spec layer). Steps 3-5 are compute unification. Steps 6-8 extend the public surface.

