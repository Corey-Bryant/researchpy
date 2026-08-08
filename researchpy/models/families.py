# -*- coding: utf-8 -*-
"""
Researchpy GLM Family / Link Classes

This module defines the distribution-family abstractions used by
Generalized Linear Models (GLMs) in ResearchPy.  Each ``Family``
subclass encapsulates the three pieces of information that make a GLM
family-specific:

1. **Inverse link function** — maps the linear predictor η to the conditional mean μ = g⁻¹(η).
2. **Variance function** — V(μ), the variance of Y as a function of its mean under the assumed distribution.
3. **Derivative dμ/dη** — needed to construct the working weights W = (dμ/dη)² / V(μ) used by IRLS.

Additional helpers (``log_likelihood``, ``deviance_residuals``) are
provided so that convergence monitoring and goodness-of-fit statistics
can be computed generically.

Usage
-----
>>> from researchpy.models.families import get_family
>>> fam = get_family("binomial")       # resolve from string
>>> mu  = fam.link_inverse(eta)        # η → μ
>>> w   = fam.working_weights(eta)     # IRLS diagonal weights

The ``get_family`` registry allows the rest of the codebase to keep
passing plain strings (``"binomial"``, ``"poisson"``, …) while the
optimizer receives a fully-typed ``Family`` object.

References
----------
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman & Hall.
- Nelder, J. A. & Wedderburn, R. W. M. (1972). Generalized Linear
  Models. *Journal of the Royal Statistical Society A*, 135(3),
  370–384.

"""

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Family(ABC):
    """Abstract base class for GLM distribution families.

    Every concrete subclass must implement the five abstract methods
    below.  Together they supply everything the generic IRLS solver
    (and any future solver) needs to fit a GLM without hard-coding
    distribution-specific logic.

    Attributes
    ----------
    distribution : str
        Human-readable distribution name used in summary tables and
        diagnostics (e.g. ``"Binomial"``).
    name : str
        Lowercase canonical key used for internal identification,
        registry look-ups, and string comparisons throughout the
        codebase (e.g. ``"binomial"``).
    link : str
        Name of the canonical link function (e.g. ``"logit"``).

    Notes
    -----
    The ``working_weights`` convenience method is *not* abstract — it
    derives W from ``dmu_deta`` and ``variance`` using the standard
    GLM formula:

        W = (dμ/dη)² / V(μ)

    Subclasses may override it for numerically superior alternatives.
    """

    distribution: str = "Family"
    name: str = distribution.lower()
    link: str = "link"


    # ----- abstract interface -----

    @abstractmethod
    def link_forward(self, mu: np.ndarray) -> np.ndarray:
        """Link function  g(μ) → η.

        Maps the conditional mean *mu* to the linear predictor scale.

        Parameters
        ----------
        mu : np.ndarray
            Conditional mean, shape ``(n, 1)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            Linear predictor η, same shape as *mu*.
        """
        ...

    @abstractmethod
    def link_inverse(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link function  g⁻¹(η) → μ.

        Maps the linear predictor *eta* (= X @ β) to the conditional
        mean of the response.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor, shape ``(n, 1)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            Conditional mean μ, same shape as *eta*.
        """
        ...

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function  V(μ).

        Returns the variance of Y as a function of the mean under the
        assumed distribution.  Used in the IRLS working-weight
        calculation.

        Parameters
        ----------
        mu : np.ndarray
            Conditional mean, shape ``(n, 1)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            V(μ), same shape as *mu*.
        """
        ...

    @abstractmethod
    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative of μ with respect to η, i.e.  dg⁻¹/dη.

        For the canonical link this equals V(μ), but the method is
        kept separate so that non-canonical links can be supported
        without changing the solver.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor, shape ``(n, 1)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            dμ/dη, same shape as *eta*.
        """
        ...

    @abstractmethod
    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Per-observation unit deviance  d(y, μ).

        The total deviance D = Σ d(yᵢ, μᵢ) is used for convergence
        checks and goodness-of-fit reporting.

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray
            Fitted mean values.

        Returns
        -------
        np.ndarray
            Unit deviance contributions, same shape as *y*.
        """
        ...

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Log-likelihood  ℓ(μ; y).

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray
            Fitted mean values.

        Returns
        -------
        float
            Scalar log-likelihood value.
        """
        ...

    # ----- concrete convenience methods -----

    def working_weights(self, eta: np.ndarray) -> np.ndarray:
        """IRLS diagonal working weights  W = (dμ/dη)² / V(μ).

        These weights appear on the diagonal of the weight matrix in
        the iteratively reweighted least-squares update:

            β_new = (X'WX)⁻¹ X'Wz

        where z is the adjusted dependent variate (working response).

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor, shape ``(n, 1)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            Diagonal weights, same shape as *eta*.

        Notes
        -----
        For a canonical link, dμ/dη = V(μ), so W simplifies to V(μ).
        Subclasses may override this for better numerical behaviour.
        """
        mu = self.link_inverse(eta)
        d = self.dmu_deta(eta)
        v = self.variance(mu)

        # Guard against division by zero in edge cases
        v = np.clip(v, 1e-15, None)

        return (d ** 2) / v

    def working_response(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Adjusted dependent variate (working response) for IRLS.

        z = η + (y − μ) / (dμ/dη)

        Parameters
        ----------
        eta : np.ndarray
            Current linear predictor.
        y : np.ndarray
            Observed response values.

        Returns
        -------
        np.ndarray
            Working response z, same shape as *eta*.

        References
        ----------
        McCullagh & Nelder (1989), §2.5.
        """
        mu = self.link_inverse(eta)
        d = self.dmu_deta(eta)
        # Guard against division by zero
        d = np.where(np.abs(d) < 1e-15, 1e-15, d)

        return eta + (y - mu) / d

    def estimate_dispersion(self, y: np.ndarray, mu: np.ndarray, n: int, k: int) -> float:
        """Estimate the dispersion parameter φ.

        For families with known dispersion (Binomial, Poisson), returns 1.0.
        Families with estimated dispersion (Gaussian, Gamma) override this
        to compute φ̂ from the Pearson chi-squared statistic.

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray
            Fitted mean values.
        n : int
            Number of observations.
        k : int
            Number of estimated parameters (including intercept).

        Returns
        -------
        float
            Dispersion parameter φ (1.0 for known-dispersion families).

        References
        ----------
        McCullagh & Nelder (1989), §2.4.
        """
        return 1.0

    def initial_intercept(self, y: np.ndarray) -> float:
        """Smart intercept initialization: g(mean(y)).

        Computes the sample mean of the response and applies the forward
        link function to produce a sensible starting value for the
        intercept parameter in iterative fitting algorithms.

        Parameters
        ----------
        y : np.ndarray
            Observed response values.

        Returns
        -------
        float
            Initial intercept estimate on the linear predictor scale.
        """
        y_mean = float(np.mean(y))
        return float(self.link_forward(np.atleast_1d(y_mean)).ravel()[0])

    # ----- representation ----- #
    def __call__(self) -> str:
        """Return the canonical family key when the instance is called."""
        return self.name

    def __repr__(self) -> str:
        return self.name




# ---------------------------------------------------------------------------
# Concrete families
# ---------------------------------------------------------------------------

class BinomialFamily(Family):
    """Binomial distribution with the canonical **logit** link.

    Link function
        g(μ) = log(μ / (1 − μ))   (logit)

    Inverse link
        μ = g⁻¹(η) = 1 / (1 + exp(−η))   (logistic / expit)

    Variance function
        V(μ) = μ(1 − μ)

    Canonical for binary (0/1) outcomes and proportions.

    References
    ----------
    McCullagh & Nelder (1989), Ch. 4.
    """

    distribution: str = "Binomial"
    name: str = distribution.lower()
    link: str = "logit"

    def link_forward(self, mu: np.ndarray) -> np.ndarray:
        """Logit link: g(μ) = log(μ / (1 − μ))."""
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(mu / (1 - mu))

    def link_inverse(self, eta: np.ndarray) -> np.ndarray:
        """Logistic (sigmoid) function — numerically stable via ``scipy.special.expit``."""
        return expit(eta)

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(μ) = μ(1 − μ)."""
        return mu * (1 - mu)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """For the logit link, dμ/dη = μ(1 − μ) = V(μ)."""
        mu = self.link_inverse(eta)
        return mu * (1 - mu)

    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Unit deviance for the Binomial family.

        d(y, μ) = 2[y·log(y/μ) + (1−y)·log((1−y)/(1−μ))]

        Edges (y=0 or y=1) are handled by clipping.
        """
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        # Use safe ratios to avoid log(0) warnings when y is exactly 0 or 1
        safe_y = np.clip(y, 1e-15, 1 - 1e-15)
        term1 = y * np.log(safe_y / mu)
        term2 = (1 - y) * np.log((1 - safe_y) / (1 - mu))
        return 2.0 * (term1 + term2)

    def log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Bernoulli log-likelihood  ℓ = Σ[y·log(μ) + (1−y)·log(1−μ)]."""
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


class PoissonFamily(Family):
    """Poisson distribution with the canonical **log** link.

    Link function
        g(μ) = log(μ)

    Inverse link
        μ = g⁻¹(η) = exp(η)

    Variance function
        V(μ) = μ

    Canonical for count data.

    References
    ----------
    McCullagh & Nelder (1989), Ch. 6.
    """

    distribution: str = "Poisson"
    name: str = distribution.lower()
    link: str = "log"

    def link_forward(self, mu: np.ndarray) -> np.ndarray:
        """Log link: g(μ) = log(μ)."""
        return np.log(np.clip(mu, 1e-15, None))

    def link_inverse(self, eta: np.ndarray) -> np.ndarray:
        """Exponential inverse link.

        Clipped to ``exp(eta) ≤ 1e15`` to prevent overflow in
        early iterations with poor starting values.
        """
        return np.exp(np.clip(eta, None, 35))  # exp(35) ≈ 1.6e15

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(μ) = μ."""
        return mu

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """For the log link, dμ/dη = exp(η) = μ."""
        return self.link_inverse(eta)

    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Unit deviance for the Poisson family.

        d(y, μ) = 2[y·log(y/μ) − (y − μ)]
        """
        mu = np.clip(mu, 1e-15, None)
        term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * (term1 - (y - mu))

    def log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Poisson log-likelihood  ℓ = Σ[y·log(μ) − μ − log(y!)]."""
        from scipy.special import gammaln
        mu = np.clip(mu, 1e-15, None)
        return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


class GaussianFamily(Family):
    """Gaussian (Normal) distribution with the canonical **identity** link.

    Link function
        g(μ) = μ   (identity)

    Inverse link
        μ = g⁻¹(η) = η

    Variance function
        V(μ) = 1   (constant)

    This is the family underlying ordinary least-squares regression.
    Included here for completeness so that OLS can also be expressed
    as a GLM fitted via IRLS (the IRLS solution in one iteration
    equals the normal-equation solution).

    References
    ----------
    McCullagh & Nelder (1989), Ch. 2.
    """

    distribution: str = "Gaussian"
    name: str = distribution.lower()
    link: str = "identity"

    def link_forward(self, mu: np.ndarray) -> np.ndarray:
        """Identity link: g(μ) = μ."""
        return mu

    def link_inverse(self, eta: np.ndarray) -> np.ndarray:
        """Identity — μ = η."""
        return eta

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(μ) = 1 (constant variance)."""
        return np.ones_like(mu)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = 1 for the identity link."""
        return np.ones_like(eta)

    def estimate_dispersion(self, y: np.ndarray, mu: np.ndarray, n: int, k: int) -> float:
        """Estimate the dispersion parameter φ = σ² for the Gaussian family.

        Computed as the Pearson chi-squared statistic divided by the
        residual degrees of freedom:

            φ̂ = Σ[(yᵢ − μᵢ)² / V(μᵢ)] / (n − k)

        For the Gaussian family V(μ) = 1, so this simplifies to:

            φ̂ = RSS / (n − k)

        which is the familiar mean squared error (MSE).

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray
            Fitted mean values.
        n : int
            Number of observations.
        k : int
            Number of estimated parameters (including intercept).

        Returns
        -------
        float
            Estimated dispersion (scale) parameter.

        References
        ----------
        McCullagh & Nelder (1989), §2.4.
        """
        residuals = y - mu
        variance_mu = self.variance(mu)
        pearson_chi2 = float(np.sum((residuals ** 2) / variance_mu))
        return pearson_chi2 / (n - k)

    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Unit deviance for the Gaussian family: d(y, μ) = (y − μ)²."""
        return (y - mu) ** 2

    def log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Gaussian log-likelihood (up to a constant depending on σ²).

        ℓ = −n/2 · log(2π·RSS/n) − n/2

        where RSS = Σ(y − μ)².  This is the *profile* log-likelihood
        with σ² = RSS/n plugged in.
        """
        n = len(y.ravel())
        rss = float(np.sum((y - mu) ** 2))
        if rss <= 0:
            rss = 1e-15
        return -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0


# ---------------------------------------------------------------------------
# Family registry — maps canonical string names to Family classes
# ---------------------------------------------------------------------------

_FAMILY_REGISTRY: Dict[str, Type[Family]] = {
    "binomial": BinomialFamily,
    "poisson": PoissonFamily,
    "gaussian": GaussianFamily,
    "normal": GaussianFamily,      # common alias
}


def get_family(name: str) -> Family:
    """Resolve a family name string to a ``Family`` instance.

    Parameters
    ----------
    name : str
        Case-insensitive family name (e.g. ``"binomial"``,
        ``"poisson"``, ``"gaussian"``).

    Returns
    -------
    Family
        A new instance of the corresponding ``Family`` subclass.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.

    Examples
    --------
    >>> fam = get_family("binomial")
    >>> fam.distribution
    'Binomial'
    >>> fam.name
    'binomial'
    >>> fam.link_inverse(np.array([0.0]))
    array([0.5])
    """

    key = name.strip().lower()
    if key not in _FAMILY_REGISTRY:
        available = ", ".join(sorted(_FAMILY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown family '{name}'. Available families: {available}"
        )

    return _FAMILY_REGISTRY[key]()

