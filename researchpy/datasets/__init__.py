# -*- coding: utf-8 -*-
"""
Datasets and Dataset Importers for ResearchPy

Provides access to datasets for cross-validation and teaching purposes. Designed to complement
the core modeling and summary tools in ResearchPy by offering a convenient way to load example datasets.

Example usage:
    >>> from researchpy.datasets import fetch_dta, auto
    >>> df = fetch_dta('auto')
    >>> print(df.head())
"""
from .stata_webuse import (
    fetch_dta, list_available_datasets, auto, census, citytemp, cancer, lifeexp,
    nlsw88, sp500, systolic, lbw, uslifeexp, voter
)

__all__ = [
    'fetch_dta',
    'list_available_datasets',
    'auto',
    'census',
    'citytemp',
    'cancer',
    'lifeexp',
    'nlsw88',
    'sp500',
    'systolic',
    'lbw',
    'uslifeexp',
    'voter'
]