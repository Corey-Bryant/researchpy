# ResearchPy Golden Test Coverage

> Last updated: 2026-08-07
> Golden schema version: 1.0

This document tracks which ResearchPy functions have golden-value tests
and their validation status. It is maintained manually until automated
coverage generation is implemented.

## Status Legend

| Symbol | Meaning |
|-------|---------|
| ✅ | Golden tests exist and pass |
| 🟡 | Partial coverage; gaps remain |
| ❌ | No golden tests; needs migration |
| ❓ | Not yet reviewed |

## Coverage Table

### Inferential Statistics

| Module | Function | Has Golden Test? | Golden Source | Dataset | Last Verified | Notes |
|--------|----------|------------------|---------------|---------|---------------|-------|
| anova | `anova()` | ✅ Yes | Stata docs | systolic | 2026-08-07 | Type I, II, III SS covered; factor effects; regression table |
| ols | `ols()` | ✅ Yes | Stata docs | systolic | 2026-08-07 | Shared goldens with anova; CI levels tested |
| crosstab | `crosstab()` | ✅ Yes | SciPy | systolic | 2026-08-07 | Chi-square, G-test, Fisher's exact; expected freqs; proportions |
| logistic | `logistic()` | 🟡 Tracked | Stata docs | lbw (?) | — | Referenced in notes; confirm test file exists |
| difference_test | `difference_test()` | ❌ No | — | — | — | **Priority: HIGH**; legacy t-test/Wilcoxon |
| signrank | `signrank()` | ❌ No | — | — | — | **Priority: HIGH**; duplicate Wilcoxon logic |
| likelihood_ratio | `LikelihoodRatioTest` | ❌ No | — | — | — | New post-estimation; needs tests |

### Descriptive Statistics

| Module | Function | Has Golden Test? | Golden Source | Dataset | Last Verified | Notes |
|--------|----------|------------------|---------------|---------|---------------|-------|
| summarize | `summarize()` | 🟡 Partial | Stata docs | auto, systolic | 2026-08-07 | Helper-based via `compute_descriptives()`; needs direct `rp.summarize()` calls |
| summary_cat | `summary_cat()` | ✅ Yes | Stata docs | systolic | 2026-08-07 | Via `SYSTOLIC_SUMMARY_CAT` golden dict |
| summary_cont | `summary_cont()` | ❓ Unknown | — | — | — | Need to verify |
| codebook | `codebook()` | ❓ Unknown | — | — | — | Need to verify |
| correlation | `correlation()` | ❓ Unknown | — | — | — | Need to verify |

### Optimization & Utilities

| Module | Function | Has Golden Test? | Golden Source | Dataset | Last Verified | Notes |
|--------|----------|------------------|---------------|---------|---------------|-------|
| methods | `IRLS()` | ❌ No | — | — | — | Optimization solver; needs convergence tests |
| methods | `newton_raphson()` | ❌ No | — | — | — | Optimization solver; needs convergence tests |
| stata_webuse | `fetch_dta()` | ❓ Unknown | — | — | — | Network-dependent; may need mocking |

## Dataset Coverage

| Dataset | Obs | Vars | Goldens Defined | Functions Tested | Golden Version |
|---------|-----|------|----------------|-----------------|----------------|
| auto | 74 | 12 | ✅ Yes | summarize, summary_cat (partial) | 1.0 |
| systolic | 58 | 3 | ✅ Yes | summarize, summary_cat, crosstab, anova, ols, regression | 1.0 |
| lbw | 189 | 11 | ❌ No | — | — |

## Migration Priority

1. 🔴 **HIGH**: `difference_test` - widely used; no tests at all
2. 🔴 **HIGH**: `signrank` - duplicate Wilcoxon logic; untested
3. 🟠 **MEDIUM**: `likelihood_ratio` - new code; needs validation
4. 🟠 **MEDIUM**: `IRLS` / `newton_raphson` - solvers underpin GLMs
5. 🟡 **LOW**: `summarize` - has partial coverage; needs direct call migration
6. 🟡 **LOW**: `logistic` - confirm test file status
7. ❓ **REVIEW**: `summary_cont`, `codebook`, `correlation` - status unknown