# researchpy

ResearchPy is a Python package that is designed to be easy to use and deliver univariate, bivariate, 
and multivariate models and statistical tests with clear and informative outputs ready for interpretation, 
further analysis, and downstream export without additional formatting. Open-source, clear function names, 
intuitive parameters, and robust documentation.

Built for researchers, analysts transitioning from SPSS/Stata/R, and students learning statistics.

[![PyPI version](https://img.shields.io/pypi/v/researchpy.svg)](https://pypi.org/project/researchpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

## Features

- **Descriptive statistics** — `summarize()`, `summary_cont()`, `summary_cat()`, `codebook()`
- **Difference tests** — Independent t-test, paired t-test, Welch's t-test, Wilcoxon signed-rank via `difference_test()`
- **ANOVA** — Type I, II, and III sum of squares with `anova()`
- **Correlation** — Correlation matrices and testing
- **Crosstabs** — Cross-tabulation with chi-square testing
- **OLS regression** — Ordinary least squares modeling
- **Effect sizes** — Cohen's d, Hedge's g, Glass's delta, eta/epsilon/omega squared, and more
- **Confidence intervals** — Included by default in descriptive and inferential output
- **Structured output** — Results returned as pandas DataFrames, ready for reporting

## Installation

```bash
pip install researchpy
```

## Quick Start

```python
import pandas as pd
import researchpy as rp

# Load example data
df = pd.DataFrame({
    "score": [88, 92, 75, 85, 90, 78, 95, 70, 82, 87,
              72, 68, 80, 76, 74, 69, 71, 77, 73, 79],
    "group": ["treatment"]*10 + ["control"]*10
})
```

### ANOVA

```python
model = rp.anova("score ~ C(group)", data=df) # Type III sum of squares by default
descriptives, results = model.results()
print(descriptives, results, sep = "\n"*2)
```

### Codebook

```python
rp.codebook(df)
```

### Difference Test

```python
desc, results = rp.difference_test("score ~ C(group)", data=df).conduct(effect_size="all")
print(desc, results, sep = "\n"*2)
```

### Summarize

```python
# Continuous summary statistics
rp.summarize(df["score"], stats=["N", "Mean", "SD", "SE", "CI"])

# Categorical frequency table
rp.summary_cat(df["group"])
```

## Requirements

- Python ≥ 3.12
- pandas ≥ 3.0.1
- numpy ≥ 2.5.0
- scipy ≥ 1.17.1
- statsmodels ≥ 0.14.0

## Documentation

Full documentation is available at [researchpy.readthedocs.io](https://researchpy.readthedocs.io/).

## License

ResearchPy is released under the [MIT License](LICENSE.txt).

## Citation

If you use ResearchPy in your research, please cite:

> Bryant, C. (2018–2026). *researchpy* (Version X.Y.Z) [Python package]. https://github.com/Corey-Bryant/researchpy

To find your installed version for citation:

```python
import researchpy
print(researchpy.__version__)
```

Current citation with version number:

> Bryant, C. (2018–2026). *researchpy* (Version 0.3.7.1) [Python package]. https://github.com/Corey-Bryant/researchpy