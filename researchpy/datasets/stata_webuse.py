"""
Stata Web Dataset Importer for ResearchPy

Provides access to Stata's web-hosted datasets for cross-validation and
teaching purposes. Designed to complement existing Stata workflow, not replace it.

Example usage:
    >>> import researchpy as rp
    >>> df = rp.datasets.stata_webuse.systolic()
    >>> df.head()

    >>> # Or more explicitly:
    >>> df = rp.datasets.stata_webuse.fetch_dta('nlsw88',version='r19')
"""

import pandas as pd
from typing import Optional
import requests
from io import BytesIO

# Base URLs for different Stata versions
_STATA_BASE_URLS = {
    'r19': 'https://www.stata-press.com/data/r19/',
    'r18': 'https://www.stata-press.com/data/r18/',
    'r17': 'https://www.stata-press.com/data/r17/',
    'r16': 'https://www.stata-press.com/data/r16/',
    'r15': 'https://www.stata-press.com/data/r15/',
    'r14': 'https://www.stata-press.com/data/r14/',
    'r13': 'https://www.stata-press.com/data/r13/',
    'r12': 'https://www.stata-press.com/data/r12/',
}

# Default version matching current Stata release
_DEFAULT_VERSION = 'r19'

# Known popular datasets for convenience functions
_COMMON_DATASETS = {
    'auto': ('auto.dta', 'Sample automobile price and features data'),
    'census': ('census.dta', '1980 US Census data by state'),
    'citytemp': ('citytemp.dta', 'Temperature and energy consumption data'),
    'cancer': ('cancer.dta', 'Cancer study data'),
    'lifeexp': ('lifeexp.dta', 'Life expectancy data'),
    'nlsw88': ('nlsw88.dta', '1988 National Longitudinal Survey young women'),
    'sp500': ('sp500.dta', 'S&P 500 index data'),
    'uslifeexp': ('uslifeexp.dta', 'US life expectancy time series'),
    'voter': ('voter.dta', 'Voter behavior data'),
    'sysdsn1': ('sysdsn1.dta', 'System supply data for multinomial models'),
    'mroz87': ('mroz87.dta', 'Mroz 1987 labor participation data'),
    'airline': ('airline.dta', 'Airline cost data'),
}


def fetch_dta(name: str, version: Optional[str] = None, url: Optional[str] = None, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch a dataset directly from Stata Press web repository.

    Parameters
    ----------
    name : str
        Dataset filename without extension (e.g., 'auto', 'nlsw88')
    version : str, optional
        Stata version directory (e.g., 'r19', 'r18'). Defaults to 'r19'.
    url : str, optional
        Custom base URL for dataset retrieval. If provided, overrides version. Defaults to None.
    force_download : bool, default False
        If True, re-download even if available locally
    timeout : int, default 30
        Request timeout in seconds

    Returns
    -------
    pd.DataFrame
        Dataset loaded into pandas DataFrame with appropriate dtypes preserved

    Raises
    ------
    ValueError
        If version not supported or dataset not found
    ConnectionError
        If unable to reach Stata Press server

    Examples
    --------
    >>> import researchpy as rp
    >>>
    >>> # Get auto dataset (default latest version)
    >>> df = rp.datasets.stata_webuse.fetch_dta('auto')
    >>> print(df.shape)
    (74, 12)
    >>>
    >>> # Specify older version for reproducibility
    >>> df_old = rp.datasets.stata_webuse.fetch_dta('auto',version='r15')
    >>>
    >>> # Force refresh from server
    >>> df_new = rp.datasets.stata_webuse.fetch_dta('auto',force_download=True)
    >>>
    >>> # Get Stata dataset from custom URL
    >>> df_cust = rp.datasets.stata_webuse.fetch_dta('glm-reg', url='https://academicweb.nd.edu/~rwilliam/statafiles/')
    >>> print(df_cust.shape)
    (500, 5)
    """
    if version is None:
        version = _DEFAULT_VERSION

    if version not in _STATA_BASE_URLS:
        raise ValueError(
                f"Unsupported version '{version}'. Supported versions: "
                f"{', '.join(_STATA_BASE_URLS.keys())}"
        )

    if url is not None:
        base_url = url
    else:
        base_url = _STATA_BASE_URLS[version]
    url = f"{base_url}{name}.dta"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Load directly from bytes buffer
        return pd.read_stata(BytesIO(response.content))

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                    f"Dataset '{name}' not found at version {version}. "
                    f"Check spelling or try different version."
            ) from e
        raise ConnectionError(f"Failed to download dataset: {e}") from e

    except Exception as e:
        raise ConnectionError(f"Error loading dataset: {e}") from e


def list_available_datasets(version: Optional[str] = None):
    """
    List all available datasets for a given Stata version.

    Parameters
    ----------
    version : str, optional
        Stata version directory. Defaults to 'r19'.

    Returns
    -------
    dict
        Dictionary mapping dataset names to descriptions from manual pages

    Notes
    -----
    This scrapes the index page metadata. For complete accuracy, refer to
    https://www.stata-press.com/data/r{version}/r.html for Base Reference Manual.
    """
    # Note: This would need HTML scraping logic - placeholder for now
    raise NotImplementedError(
        "list_available_datasets requires HTML parsing. "
        "For now, consult https://www.stata-press.com/data/r19/r.html"
    )


# Convenience wrapper functions for common datasets
def systolic(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('systolic', version=version)

def lbw(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('lbw', version=version)

def auto(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('auto', version=version)

def census(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('census', version=version)

def citytemp(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('citytemp', version=version)

def cancer(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('cancer', version=version)

def lifeexp(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('lifeexp', version=version)

def nlsw88(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('nlsw88', version=version)

def sp500(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('sp500', version=version)

def uslifeexp(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('uslifeexp', version=version)

def voter(version: Optional[str] = None) -> pd.DataFrame:
    return fetch_dta('voter', version=version)