# ResearchPy Golden Testing Guide

## What Is a Golden Test?

A **golden test** validates that a function's output matches a trusted
external reference value (the "golden" value). Unlike a unit test that
checks "does it run?" or "does it return the right type?", a golden test
checks "does it produce the *correct* statistical result?"

Golden values are currently transcribed from:

- **Stata output** (the primary source, documented in ResearchPy docs)
- **SciPy computations** (used when SciPy itself is the reference standard)
- **R output** (occasional cross-validation)

Golden tests live in `Tests/` alongside regular unit tests. 
The golden *values* themselves live in `Tests/Golden/golden_values.py`.

---

## How to Create a New Test File

### Step 1: Copy the Template

`bash cp Tests/test_TEMPLATE.py Tests/test_your_module.py`


### Step 2: Add Golden Values

Open `Tests/Golden/golden_values.py` and add a new constant for your module:

```python
# -- YourModule: descriptive statistics --
YOUR_MODULE_GOLDEN = { "N": 58, "Mean": 18.8793, "SD": 12.8009, "95% Conf. Interval": (15.5135, 22.2451), }
```

If introducing a new dataset, also add an entry to `GOLDEN_METADATA`.

### Step 3: Write Fixtures

Fixtures handle model fitting and data loading. Use `scope="module"` or `scope="session"` to avoid refitting:

```python
@pytest.fixture(scope="module")
def model_fixture(systolic_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from researchpy.your_module import YourClass
        return YourClass("systolic ~ C(drug)", data=systolic_df)
```

Dataset fixtures currently available (defined in `conftest.py`):

| Fixture | Description |
|---------|-------------|
| `systolic_df` | Stata systolic dataset (58 obs x 3 vars) |
| `auto_df` | Stata auto dataset (74 obs x 12 vars) |
| `lbw_df` | Stata lbw dataset (189 obs x 11 vars) |
| `auto_price` | `auto_df["price"]` as Series |
| `systolic_series` | `systolic_df["systolic"]` as Series |

### Step 4: Write Golden Validation Tests

Compare each numeric output to its golden value using `pytest.approx()`:

```python
def test_mean(self, model_fixture):
    golden = YOUR_MODULE_GOLDEN["Mean"]
    actual = model_fixture.model_data["mean"]
    assert actual == pytest.approx(golden, rel=APPROX_REL, abs=APPROX_ABS)
```

For multiple similar checks, use `@pytest.mark.parametrize`:

```python
@pytest.mark.parametrize("factor_idx,source_key", [
    (0, "drug"),
    (1, "disease"),
])
def test_factor_ss(self, model_fixture, factor_idx, source_key):
    golden = YOUR_GOLDEN_TABLE[source_key]
    actual = model_fixture.factor_effects["Sum of Squares"][factor_idx]
    assert actual == pytest.approx(golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS)
```


### Step 5: Write Return Type Tests

Verify the API contract (Dataframe vs Dictionary, tuple length, etc.):

```python
def test_returns_dataframe_tuple(self, results_df):
    assert isinstance(results_df, tuple)
    assert len(results_df) == 2
    for item in results_df:
        assert isinstance(item, pd.DataFrame)
```

### Step 6: Write Edge Case Tests

Test mathematical identities that must always hold:

```python
def test_ss_decomposition(self, model_fixture):
    ss_total = model_fixture.model_data["sum_of_square_total"]
    ss_model = model_fixture.model_data["sum_of_square_model"]
    ss_resid = model_fixture.model_data["sum_of_square_residual"]
    assert ss_total == pytest.approx(ss_model + ss_resid, rel=1e-6)
```


### Step 7: Run and Verify

```bash
pytest Tests/test_your_module.py -v
```


All tests should pass. If a golden value fails, investigate whether:
1. The function has a bug (fix the code)
2. The golden value was transcribed incorrectly (fix the golden)
3. The algorithm intentionally changed (update the golden, bump version)

---

## Golden Value Protocol

### When to Add New Goldens

- New functions or modules
- New datasets (add to `GOLDEN_METADATA`)
- Expanding coverage (e.g., adding Type II SS tests to an existing module)

### When to Update Existing Goldens

Golden values should change rarely. Valid reasons:

1. **Algorithm change** (e.g., switching from patsy to formulaic)
2. **Bug fix** that corrects an incorrect golden value
3. **Precision improvement** (e.g., more decimal places in reference output)

### Versioning Protocol

When a golden value changes:

1. Update the value in `golden_values.py`
2. Bump `golden_version` in `GOLDEN_METADATA` for that dataset
3. Add a changelog comment:

```python
# v1.0 -> v1.1: Switched from patsy to formulaic; SS values unchanged
# to 4 decimal places. Intercept SE shifted by 1e-6.
```

4. Update `last_verified` date
5. Update `COVERAGE.md` with the new verification date

### Tolerance Strategy

Currently using global tolerances:

```python
APPROX_REL = 1e-4
APPROX_ABS = 1e-4
```


These are appropriate because golden values are transcribed at 4 decimal
places. If a specific test needs tighter or looser tolerance, override
locally:

```python
assert actual == pytest.approx(golden, rel=1e-6, abs=1e-8)
```


---

## Common Patterns

### Helper: `assert_stats_match()`

For descriptive statistics with many fields, use the helper in `test_summarize_golden.py`:

```python
from Tests.Golden.test_summarize_golden import assert_stats_match

def test_descriptives(self, series):
    golden = YOUR_GOLDEN
    actual = compute_descriptives(series)
    assert_stats_match(actual, golden)
```

### Suppressing Deprecation Warnings in Fixtures

Legacy classes emit `DeprecationWarning`. Wrap fixture model fitting:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    model = YourClass("formula", data=df)
```

### Cross-Validation Against SciPy

When SciPy is the trusted source, compute the reference value directly
in the test:

```python
raw_ct = pd.crosstab(drug, disease)
chi2_scipy, p_scipy, _, _ = scipy.stats.chi2_contingency(raw_ct, correction=False)

assert rp_chi2 == pytest.approx(round(chi2_scipy, 4), rel=APPROX_REL, abs=APPROX_ABS)
```


---

## File Organization

```
Tests/ conftest.py          # Shared fixtures (datasets) 
test_TEMPLATE.py            # Copy this to start a new test file 
test_anova.py               # Golden tests for anova() 
test_ols.py                 # Golden tests for ols() 
test_crosstab.py            # Golden tests for crosstab() 
test_summarize_golden.py    # Golden tests for summarize() 
test_your_module.py         # Your new test file 
Golden/ init.py 
golden_values.py            # All golden values + metadata 
TESTING_GUIDE.md            # This file 
COVERAGE.md                 # Coverage tracking table 
stata_datasets.py           # Local dataset dictionaries
```

---

## Checklist for New Golden Tests

- [ ] Golden values added to `golden_values.py`
- [ ] `GOLDEN_METADATA` updated (if new dataset)
- [ ] `COVERAGE.md` updated
- [ ] Fixtures defined (module-scoped)
- [ ] Golden validation tests written
- [ ] Return type tests written
- [ ] Edge case tests written
- [ ] All tests pass: `pytest Tests/test_your_module.py -v`
- [ ] Deprecation warnings tested (if applicable)