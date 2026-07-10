"""
Golden reference values for ResearchPy cross-validation.

All values are transcribed directly from ResearchPy documentation:

1. Stata 'auto' dataset:
   https://researchpy.readthedocs.io/en/latest/summarize_documentation.html

2. Stata 'systolic' dataset:
   https://researchpy.readthedocs.io/en/latest/anova_documentation.html#examples

These serve as oracles: any function, class, or method producing descriptive
statistics or model outputs should match these values when applied to the same
data slices.

Source datasets loaded from: researchpy.Tests.stata_datasets
  - auto:     74 obs x 12 vars
  - systolic: 58 obs x 3 vars
  - lbw:      189 obs x 11 vars
"""

# Default tolerance for pytest.approx() comparisons.
# Doc values are rounded to 4 decimal places, so this gives a reasonable
# cushion for floating-point differences while catching real bugs.
APPROX_REL = 1e-4
APPROX_ABS = 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# AUTO DATASET GOLDEN VALUES
# Source: summarize() documentation
# ═══════════════════════════════════════════════════════════════════════════

# -- Auto: Single variable auto["price"] -----------------------------------
AUTO_PRICE = {
    "N": 74,
    "Mean": 6165.2568,
    "Median": 5006.5,
    "Variance": 8699525.9743,
    "SD": 2949.4959,
    "SE": 342.8719,
    "95% Conf. Interval": (5481.914, 6848.5995),
}

AUTO_MPG = {
    "N": 74,
    "Mean": 21.2973,
    "Median": 20.0,
    "Variance": 33.472,
    "SD": 5.7855,
    "SE": 0.6726,
    "95% Conf. Interval": (19.9569, 22.6377),
}

# -- Auto: Two variables auto[["price", "mpg"]] ----------------------------
AUTO_PRICE_MPG = {
    "price": AUTO_PRICE,
    "mpg": AUTO_MPG,
}



# -- Auto: Series Groupby auto.groupby("foreign")["price"] ----------------
AUTO_SERIES_GROUPBY = {
    "Domestic": {
        "price": {
            "N": 52,
            "Mean": 6072.4231,
            "Median": 4782.5,
            "Variance": 9592054.9155,
            "SD": 3097.1043,
            "SE": 429.4911,
            "95% Conf. Interval": (5210.1837, 6934.6624),
        },
    },
    "Foreign": {
        "price": {
            "N": 22,
            "Mean": 6384.6818,
            "Median": 5759.0,
            "Variance": 6874438.7035,
            "SD": 2621.9151,
            "SE": 558.9942,
            "95% Conf. Interval": (5222.1898, 7547.1738),
        },
    },
}

# -- Auto: DataFrame Groupby auto.groupby("foreign")[["price", "mpg"]] -----
AUTO_DATAFRAME_GROUPBY = {
    "Domestic": {
        "price": AUTO_SERIES_GROUPBY["Domestic"]["price"],
        "mpg": {
            "N": 52,
            "Mean": 19.8269,
            "Median": 19.0,
            "Variance": 22.4989,
            "SD": 4.7433,
            "SE": 0.6578,
            "95% Conf. Interval": (18.5064, 21.1475),
        },
    },
    "Foreign": {
        "price": AUTO_SERIES_GROUPBY["Foreign"]["price"],
        "mpg": {
            "N": 22,
            "Mean": 24.7727,
            "Median": 24.5,
            "Variance": 43.7078,
            "SD": 6.6112,
            "SE": 1.4095,
            "95% Conf. Interval": (21.8415, 27.704),
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SYSTOLIC DATASET GOLDEN VALUES
# Source: anova() documentation
# ═══════════════════════════════════════════════════════════════════════════

# -- Systolic: rp.summarize(systolic["systolic"]) --------------------------
SYSTOLIC_SUMMARIZE = {
    "Name": "systolic",
    "N": 58,
    "Mean": 18.8793,
    "Median": 21.0,
    "Variance": 163.862,
    "SD": 12.8009,
    "SE": 1.6808,
    "95% Conf. Interval": (15.5135, 22.2451),
}

# -- Systolic: rp.summary_cat(systolic[["disease", "drug"]]) ----------
# Format: {variable_name: {outcome: {"count": int, "percent": float}}}
SYSTOLIC_SUMMARY_CAT = {
    "drug": {
        4: {"count": 16, "percent": 27.59},
        2: {"count": 15, "percent": 25.86},
        1: {"count": 15, "percent": 25.86},
        3: {"count": 12, "percent": 20.69},
    },
    "disease": {
        3: {"count": 20, "percent": 34.48},
        2: {"count": 19, "percent": 32.76},
        1: {"count": 19, "percent": 32.76},
    },
}


# -- Systolic: rp.crosstab(systolic["drug"], systolic["disease"]) ---------------
# Cross-tabulation of drug (rows) x disease (columns)
# Validated against scipy.stats.chi2_contingency
SYSTOLIC_CROSSTAB = {
    "contingency_table": {
        # drug (rows) x disease (columns) - raw counts
        1: {1: 6, 2: 4, 3: 5},
        2: {1: 5, 2: 4, 3: 6},
        3: {1: 3, 2: 5, 3: 4},
        4: {1: 5, 2: 6, 3: 5},
    },
    "n": 58,
    "shape": (4, 3),  # 4 drug levels x 3 disease levels
    "chi_square": {
        "chi2": 1.4048,
        "p_value": 0.9656,
        "dof": 6,
        "cramers_v": 0.1100,
        "phi": 0.1556,
    },
    "g_test": {
        "statistic": 1.3991,
        "p_value": 0.9659,
        "dof": 6,
    },
    "expected_frequencies": [
        [4.9138, 4.9138, 5.1724],
        [4.9138, 4.9138, 5.1724],
        [3.9310, 3.9310, 4.1379],
        [5.2414, 5.2414, 5.5172],
    ],
    "margins_total": 58,
}


# -- Systolic: Series Groupby systolic.groupby('disease')['systolic'] | systolic.groupby('drug')['systolic']_ ----------------
SYSTOLIC_SERIES_GROUPBY = {
    "disease": {
        "1": {
            "N": 19,
            "Mean": 22.7895,
            "Median": 22.0,
            "Variance": 173.1754,
            "SD": 13.1596,
            "SE": 3.019,
            "95% Conf. Interval": [16.4467, 29.1322],
        },
        "2": {
            "N": 19,
            "Mean": 18.2105,
            "Median": 16.0,
            "Variance": 183.731,
            "SD": 13.5547,
            "SE": 3.1097,
            "95% Conf. Interval": [11.6774, 24.7437],
        },
        "3": {
            "N": 20,
            "Mean": 15.8,
            "Median": 18.5,
            "Variance": 127.7474,
            "SD": 11.3025,
            "SE": 2.5273,
            "95% Conf. Interval": [10.5102, 21.0898],
        }
    },
    "drug": {
        "1": {
            "N": 15,
            "Mean": 26.0667,
            "Median": 25.0,
            "Variance": 136.3524,
            "SD": 11.677,
            "SE": 3.015,
            "95% Conf. Interval": [19.6002, 32.5332],
        },
        "2": {
            "N": 15,
            "Mean": 25.5333,
            "Median": 28.0,
            "Variance": 134.981,
            "SD": 11.6181,
            "SE": 2.9998,
            "95% Conf. Interval": [19.0994, 31.9672],
        },
        "3": {
            "N": 12,
            "Mean": 8.75,
            "Median": 8.0,
            "Variance": 100.3864,
            "SD": 10.0193,
            "SE": 2.8923,
            "95% Conf. Interval": [2.384, 15.116],
        },
        "4": {
            "N": 16,
            "Mean": 13.5,
            "Median": 13.5,
            "Variance": 86.9333,
            "SD": 9.3238,
            "SE": 2.331,
            "95% Conf. Interval": [8.5317, 18.4683],
        }
    },
}



# -- Systolic: ANOVA Type III SS table -------------------------------------
# Format: {source_name: {stat_name: value}}
SYSTOLIC_ANOVA_TABLE = {
    "Model": {
        "Source": "Model",
        "Sum of Squares": 4259.3385,
        "Degrees of Freedom": 11,
        "Mean Squares": 387.2126,
        "F value": 3.5057,
        "p-value": 0.0013,
        "Eta squared": 0.4560,
        "Omega squared": 0.3221,
    },
    "drug": {
        "Source": "drug",
        "Sum of Squares": 2997.4719,
        "Degrees of Freedom": 3,
        "Mean Squares": 999.1573,
        "F value": 9.0460,
        "p-value": 0.0001,
        "Eta squared": 0.3711,
        "Omega squared": 0.2939,
    },
    "disease": {
        "Source": "disease",
        "Sum of Squares": 415.8730,
        "Degrees of Freedom": 2,
        "Mean Squares": 207.9365,
        "F value": 1.8826,
        "p-value": 0.1637,
        "Eta squared": 0.0757,
        "Omega squared": 0.0295,
    },
    "drug:disease": {
        "Source": "drug:disease",
        "Sum of Squares": 707.2663,
        "Degrees of Freedom": 6,
        "Mean Squares": 117.8777,
        "F value": 1.0672,
        "p-value": 0.3958,
        "Eta squared": 0.1222,
        "Omega squared": 0.0069,
    },
    "Residual": {
        "Source": "Residual",
        "Sum of Squares": 5080.8167,
        "Degrees of Freedom": 46,
        "Mean Squares": 110.4525,
    },
    "Total": {
        "Source": "Total",
        "Sum of Squares": 9340.1552,
        "Degrees of Freedom": 57,
        "Mean Squares": 163.8624,
    },
}

# -- Systolic: ANOVA fit statistics ----------------------------------------
SYSTOLIC_ANOVA_FIT = {
    "Number of obs": 58,
    "Root MSE": 10.5096,
    "R-squared": 0.4560,
    "Adj R-squared": 0.3259,
}

# -- Systolic: OLS / ANOVA regression table --------------------------------
# Header and reference rows have empty strings for non-applicable fields.
# Format: list of dicts, one per row.
SYSTOLIC_REGRESSION_TABLE = [
    {
        "term": "Intercept",
        "Coef.": 29.3333,
        "Std. Err.": 4.2905,
        "t": 6.8367,
        "p-value": 0.0000,
        "95% Conf. Interval": (20.6969, 37.9697),
    },
    # -- drug (reference = 1) --
    {
        "term": "drug",
        "Coef.": "",
        "Std. Err.": "",
        "t": "",
        "p-value": "",
        "95% Conf. Interval": "",
    },
    {
        "term": "1",
        "Coef.": "(reference)",
        "Std. Err.": "",
        "t": "",
        "p-value": "",
        "95% Conf. Interval": "",
    },
    {
        "term": "2",
        "Coef.": -1.3333,
        "Std. Err.": 6.3639,
        "t": -0.2095,
        "p-value": 0.8350,
        "95% Conf. Interval": (-14.1432, 11.4765),
    },
    {
        "term": "3",
        "Coef.": -13.0000,
        "Std. Err.": 7.4314,
        "t": -1.7493,
        "p-value": 0.0869,
        "95% Conf. Interval": (-27.9587, 1.9587),
    },
    {
        "term": "4",
        "Coef.": -15.7333,
        "Std. Err.": 6.3639,
        "t": -2.4723,
        "p-value": 0.0172,
        "95% Conf. Interval": (-28.5432, -2.9235),
    },
    # -- disease (reference = 1) --
    {
        "term": "disease",
        "Coef.": "",
        "Std. Err.": "",
        "t": "",
        "p-value": "",
        "95% Conf. Interval": "",
    },
    {
        "term": "1",
        "Coef.": "(reference)",
        "Std. Err.": "",
        "t": "",
        "p-value": "",
        "95% Conf. Interval": "",
    },
    {
        "term": "2",
        "Coef.": -1.0833,
        "Std. Err.": 6.7839,
        "t": -0.1597,
        "p-value": 0.8738,
        "95% Conf. Interval": (-14.7387, 12.572),
    },
    {
        "term": "3",
        "Coef.": -8.9333,
        "Std. Err.": 6.3639,
        "t": -1.4038,
        "p-value": 0.1671,
        "95% Conf. Interval": (-21.7432, 3.8765),
    },
    # -- drug:disease interaction --
    {
        "term": "drug:disease",
        "Coef.": "",
        "Std. Err.": "",
        "t": "",
        "p-value": "",
        "95% Conf. Interval": "",
    },
    {
        "term": "2:2",
        "Coef.": 6.5833,
        "Std. Err.": 9.7839,
        "t": 0.6729,
        "p-value": 0.5044,
        "95% Conf. Interval": (-13.1107, 26.2774),
    },
    {
        "term": "2:3",
        "Coef.": -0.9000,
        "Std. Err.": 8.9999,
        "t": -0.1000,
        "p-value": 0.9208,
        "95% Conf. Interval": (-19.0159, 17.2159),
    },
    {
        "term": "3:2",
        "Coef.": -10.8500,
        "Std. Err.": 10.2435,
        "t": -1.0592,
        "p-value": 0.2950,
        "95% Conf. Interval": (-31.4692, 9.7692),
    },
    {
        "term": "3:3",
        "Coef.": 1.1000,
        "Std. Err.": 10.2435,
        "t": 0.1074,
        "p-value": 0.9150,
        "95% Conf. Interval": (-19.5192, 21.7192),
    },
    {
        "term": "4:2",
        "Coef.": 0.3167,
        "Std. Err.": 9.3017,
        "t": 0.0340,
        "p-value": 0.9730,
        "95% Conf. Interval": (-18.4066, 19.04),
    },
    {
        "term": "4:3",
        "Coef.": 9.5333,
        "Std. Err.": 9.2022,
        "t": 1.0360,
        "p-value": 0.3056,
        "95% Conf. Interval": (-8.9897, 28.0564),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_auto_golden(scenario, group=None, variable=None):
    """
    Retrieve golden values for the auto dataset by scenario.

    Parameters
    ----------
    scenario : str
        One of: 'single_price', 'two_variables', 'series_groupby',
        'dataframe_groupby'
    group : str, optional
        Group name for groupby scenarios ('Domestic', 'Foreign')
    variable : str, optional
        Variable name ('price', 'mpg')

    Returns
    -------
    dict
        Dictionary of stat name -> expected value

    Examples
    --------
    >>> get_auto_golden('single_price')
    {'N': 74, 'Mean': 6165.2568, ...}

    >>> get_auto_golden('series_groupby', group='Foreign', variable='price')
    {'N': 22, 'Mean': 6384.6818, ...}
    """
    _LOOKUP = {
        'single_price': AUTO_PRICE,
        'single_mpg': AUTO_MPG,
        'two_variables': AUTO_PRICE_MPG,
        'series_groupby': AUTO_SERIES_GROUPBY,
        'dataframe_groupby': AUTO_DATAFRAME_GROUPBY,
    }

    data = _LOOKUP[scenario]

    if group is not None:
        data = data[group]

    if variable is not None:
        data = data[variable]

    return data


def get_systolic_golden(category, row_key=None):
    """
    Retrieve golden values for the systolic dataset by category.

    Parameters
    ----------
    category : str
        One of: 'summarize', 'crosstab', 'anova_table', 'anova_fit',
        'regression_table'
    row_key : str, optional
        For anova_table: source name ('Model', 'drug', 'disease',
        'drug:disease', 'Residual', 'Total')
        For crosstab: variable name ('drug', 'disease')

    Returns
    -------
    dict or list
        Expected values for the specified category/row

    Examples
    --------
    >>> get_systolic_golden('summarize')
    {'N': 58, 'Mean': 18.8793, ...}

    >>> get_systolic_golden('anova_table', row_key='drug')
    {'Sum of Squares': 2997.4719, 'F value': 9.0460, ...}
    """
    _LOOKUP = {
        'summarize': SYSTOLIC_SUMMARIZE,
        'summary_cat': SYSTOLIC_SUMMARY_CAT,
        'crosstab': SYSTOLIC_CROSSTAB,
        'anova_table': SYSTOLIC_ANOVA_TABLE,
        'anova_fit': SYSTOLIC_ANOVA_FIT,
        'regression_table': SYSTOLIC_REGRESSION_TABLE,
    }

    data = _LOOKUP[category]

    if row_key is not None:
        data = data[row_key]

    return data



