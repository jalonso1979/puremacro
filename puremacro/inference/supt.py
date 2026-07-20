"""Sup-t simultaneous confidence bands for impulse-response paths.

Reference
---------
Montiel Olea, J.L. and Plagborg-Møller, M. (2019). "Simultaneous confidence
bands: Theory, implementation, and an application to SVARs". Journal of
Applied Econometrics 34(1), 1-17.

A pointwise (1-α) band covers each coordinate separately; the probability
that the *whole path* lies inside is well below 1-α. The sup-t band widens
every coordinate by a single critical value ``c`` — the (1-α) quantile of the
maximum absolute studentized deviation across coordinates — so the band covers
the entire path jointly with probability 1-α. It is the narrowest band of the
"constant multiple of the pointwise standard errors" family, always at least
as wide as the pointwise band and never wider than Bonferroni.

Three constructions from the paper are implemented:

* ``method='plugin'``   — Algorithm 1: Monte Carlo quantile of ``max_h |Z_h|``
  with ``Z ~ N(0, corr(Sigma))``, applied to a point estimate + covariance.
* ``method='bootstrap'`` — Algorithm 2: empirical quantile of the max absolute
  studentized deviation of bootstrap draws around the point estimate.
* ``method='bayes'``    — Algorithm 3 (simple calibrated version): smallest
  ``c`` such that at least a fraction 1-α of posterior draws lies entirely
  inside ``mean ± c·sd``.

Simplifications relative to the paper: the generalized-Pareto tail refinement
of Algorithm 3 is omitted (the plain order-statistic calibration is used), and
the plugin quantile is Monte Carlo rather than exact — seedable via ``rng``.

Pyodide-pure: numpy only.
"""
from __future__ import annotations

import numpy as np

from ._results import SupTBandResult

__all__ = ["supt_band", "SupTBandResult"]

_METHODS = ("plugin", "bootstrap", "bayes")


def _normalize_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _mvn_max_abs_quantile(R: np.ndarray, alpha: float, n_mc: int,
                          rng: np.random.Generator) -> float:
    """(1-alpha) MC quantile of max_h |Z_h|, Z ~ N(0, R), R a correlation matrix."""
    H = R.shape[0]
    eigval, eigvec = np.linalg.eigh(R)
    if eigval.min() < -1e-8 * max(1.0, float(eigval.max())):
        raise ValueError(
            f"corr(Sigma) is not positive semi-definite "
            f"(min eigenvalue {eigval.min():.3e}); check Sigma."
        )
    factor = eigvec * np.sqrt(np.clip(eigval, 0.0, None))  # F F' = R
    maxima = np.empty(n_mc)
    chunk = max(1, min(n_mc, int(4e6 / max(1, H))))
    pos = 0
    while pos < n_mc:
        k = min(chunk, n_mc - pos)
        Z = rng.standard_normal((k, H)) @ factor.T
        maxima[pos:pos + k] = np.abs(Z).max(axis=1)
        pos += k
    return float(np.quantile(maxima, 1.0 - alpha))


def _max_studentized_crit(draws: np.ndarray, center: np.ndarray,
                          scale: np.ndarray, alpha: float) -> float:
    """Smallest c with >= (1-alpha) of draws inside center ± c*scale (all h)."""
    t = np.max(np.abs(draws - center[None, :]) / scale[None, :], axis=1)
    k = int(np.ceil((1.0 - alpha) * t.size))
    k = min(max(k, 1), t.size)
    return float(np.sort(t)[k - 1])


def supt_band(theta=None, Sigma=None, *, draws=None, method: str = "plugin",
              alpha: float = 0.10, n_mc: int = 100_000, rng=None) -> SupTBandResult:
    """Sup-t simultaneous (1-alpha) confidence band across H coordinates.

    Montiel Olea & Plagborg-Møller (2019, Journal of Applied Econometrics
    34(1), 1-17). See the module docstring for the three constructions and
    the stated simplifications.

    Parameters
    ----------
    theta : array-like, shape (H,), optional
        Point estimates (band center). Required for ``method='plugin'``;
        optional for ``'bootstrap'`` (defaults to the mean of ``draws`` — a
        documented simplification of Algorithm 2, which centers on the
        original-sample estimate); must be omitted for ``'bayes'`` (the
        calibrated credible band is centered on the posterior mean).
    Sigma : array-like, shape (H, H), optional
        Joint covariance of ``theta`` (``'plugin'`` only). The critical value
        uses only ``corr(Sigma)``; band half-width is ``c * sqrt(diag(Sigma))``.
    draws : array-like, shape (n_draws, H), optional
        Bootstrap replications (``'bootstrap'``) or posterior draws
        (``'bayes'``). Rows with non-finite entries are rejected — drop
        failed replications before calling.
    method : {'plugin', 'bootstrap', 'bayes'}
        Which construction to use (Algorithms 1-3 of the paper).
    alpha : float
        Two-sided simultaneous level; ``alpha=0.10`` gives a 90% band.
    n_mc : int
        Monte Carlo draws for the plugin quantile (default 100,000).
    rng : numpy.random.Generator | int | None
        Seed or generator for the plugin Monte Carlo (default: seed 0).

    Returns
    -------
    SupTBandResult
        Frozen dataclass with ``lower``, ``upper``, ``center``, ``scale``,
        ``crit_value``, ``method``, ``alpha``, ``n_draws``.

    Raises
    ------
    ValueError
        On NaN/degenerate inputs (zero-variance coordinates, non-PSD
        correlation, shape mismatches, too few draws) with a diagnostic
        message, rather than returning silent garbage.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {_METHODS}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    if method == "plugin":
        if theta is None or Sigma is None:
            raise ValueError("method='plugin' requires both theta and Sigma")
        if draws is not None:
            raise ValueError("method='plugin' takes theta+Sigma, not draws; "
                             "use method='bootstrap' or 'bayes' for draws")
        if int(n_mc) < 100:
            raise ValueError(f"n_mc={n_mc} is too small for a meaningful "
                             f"quantile; use at least 100 (default 100,000)")
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1-D (H,); got shape {theta.shape}")
        H = theta.shape[0]
        if Sigma.shape != (H, H):
            raise ValueError(f"Sigma must be ({H}, {H}) to match theta; "
                             f"got {Sigma.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains NaN/inf entries")
        if not np.all(np.isfinite(Sigma)):
            raise ValueError("Sigma contains NaN/inf entries")
        scale_ref = max(1.0, float(np.abs(Sigma).max()))
        if not np.allclose(Sigma, Sigma.T, rtol=1e-8, atol=1e-10 * scale_ref):
            raise ValueError(
                f"Sigma must be symmetric; max asymmetry "
                f"{np.abs(Sigma - Sigma.T).max():.3e}"
            )
        Sigma = 0.5 * (Sigma + Sigma.T)
        var = np.diag(Sigma).copy()
        bad = np.flatnonzero(var <= 0)
        if bad.size:
            raise ValueError(
                f"Sigma has non-positive variances at coordinates "
                f"{bad.tolist()}; drop degenerate coordinates before calling"
            )
        sd = np.sqrt(var)
        R = Sigma / np.outer(sd, sd)
        np.fill_diagonal(R, 1.0)
        c = _mvn_max_abs_quantile(R, alpha, int(n_mc), _normalize_rng(rng))
        center, scale, n_used = theta, sd, int(n_mc)

    else:  # 'bootstrap' or 'bayes'
        if draws is None:
            raise ValueError(f"method={method!r} requires draws (n_draws, H)")
        if Sigma is not None:
            raise ValueError(f"method={method!r} takes draws, not Sigma")
        draws = np.asarray(draws, dtype=float)
        if draws.ndim != 2:
            raise ValueError(f"draws must be 2-D (n_draws, H); got shape "
                             f"{draws.shape}")
        n_used, H = draws.shape
        if n_used < 20:
            raise ValueError(
                f"only {n_used} draws; need at least 20 for a usable "
                f"quantile-of-max calibration"
            )
        if not np.all(np.isfinite(draws)):
            n_bad = int(np.sum(~np.isfinite(draws).all(axis=1)))
            raise ValueError(
                f"draws contain non-finite entries in {n_bad} row(s); drop "
                f"failed replications before calling supt_band"
            )
        if method == "bayes":
            if theta is not None:
                raise ValueError(
                    "method='bayes' centers the calibrated credible band on "
                    "the posterior mean; do not pass theta"
                )
            center = draws.mean(axis=0)
        else:  # bootstrap
            if theta is None:
                center = draws.mean(axis=0)
            else:
                center = np.atleast_1d(np.asarray(theta, dtype=float))
                if center.shape != (H,):
                    raise ValueError(f"theta must have shape ({H},) to match "
                                     f"draws; got {center.shape}")
                if not np.all(np.isfinite(center)):
                    raise ValueError("theta contains NaN/inf entries")
        scale = draws.std(axis=0, ddof=1)
        # Relative degeneracy check: an exactly-constant column has std ~1e-16
        # in floating point, which would explode the studentization.
        tiny = 1e-12 * np.maximum(1.0, np.abs(draws).max(axis=0))
        bad = np.flatnonzero(~(scale > tiny))
        if bad.size:
            raise ValueError(
                f"draws are degenerate (zero variance) at coordinates "
                f"{bad.tolist()}; drop constant coordinates before calling"
            )
        c = _max_studentized_crit(draws, center, scale, alpha)

    return SupTBandResult(
        lower=center - c * scale,
        upper=center + c * scale,
        center=center,
        scale=scale,
        crit_value=c,
        method=method,
        alpha=float(alpha),
        n_draws=n_used,
    )
