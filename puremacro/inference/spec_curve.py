"""Specification curve (Simonsohn-Simmons-Nelson 2020).

Enumerates the Cartesian product of a spec grid, applies a user-supplied
estimator to each cell, returns a DataFrame of (sigma_hat, se, ci_lo,
ci_hi) ordered by sigma_hat. Includes a two-sided non-parametric
bootstrap p-value for H0: median(sigma_hat) = h0.

The estimator callable must return a dict with at least ``sigma_hat``
and ``se`` keys; any other keys flow through to the output DataFrame
columns. Use this when you have many candidate specifications of the
same estimand and want to summarise their distribution rather than
privilege one as the headline.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd


def enumerate_specs(grid: Mapping[str, Iterable[Any]]) -> list[dict]:
    """Return the Cartesian product of `grid` as a list of dicts.

    Order of keys matches insertion order of `grid`. The returned list
    has length ``prod(len(v) for v in grid.values())`` and each dict
    has the same keys as `grid`.
    """
    keys = list(grid.keys())
    out: list[dict] = []
    for combo in product(*[list(grid[k]) for k in keys]):
        out.append(dict(zip(keys, combo)))
    return out


def run_spec_curve(
    *,
    data: Any,
    specs: list[dict],
    estimator: Callable[[Any, dict], dict],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Apply ``estimator(data, spec)`` to each spec; return ordered curve.

    Parameters
    ----------
    data : passed through to ``estimator`` unchanged.
    specs : list of spec-dicts (e.g., from :func:`enumerate_specs`).
    estimator : callable returning ``{"sigma_hat": float, "se": float, ...}``.
    ci_level : two-sided coverage for ``ci_lo`` / ``ci_hi`` columns.

    Returns
    -------
    DataFrame with one row per spec, columns include all spec keys plus
    ``sigma_hat``, ``se``, ``ci_lo``, ``ci_hi`` and any additional keys
    returned by ``estimator``. Sorted ascending by ``sigma_hat``.
    """
    if abs(ci_level - 0.95) < 1e-9:
        z = 1.959963984540054
    elif abs(ci_level - 0.90) < 1e-9:
        z = 1.6448536269514722
    elif abs(ci_level - 0.99) < 1e-9:
        z = 2.5758293035489004
    else:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    rows = []
    for spec in specs:
        result = estimator(data, spec)
        row = {**spec, **result}
        sh = float(result["sigma_hat"])
        se = float(result.get("se", float("nan")))
        row["ci_lo"] = sh - z * se
        row["ci_hi"] = sh + z * se
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("sigma_hat", kind="stable").reset_index(drop=True)


def bootstrap_pvalue_median(
    curve_df: pd.DataFrame,
    *,
    h0: float,
    B: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided bootstrap p-value for H0: median(sigma_hat) = h0.

    Resamples rows of ``curve_df`` with replacement; the bootstrap
    distribution of the centered statistic ``median(sigma_hat) - obs_median``
    serves as the empirical null. Returns the share of bootstrap draws
    where the absolute centered statistic equals or exceeds the observed
    centered statistic ``obs_median - h0``.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma = curve_df["sigma_hat"].dropna().to_numpy()
    n = len(sigma)
    if n == 0:
        return float("nan")
    obs_median = float(np.median(sigma))
    obs_stat = obs_median - h0
    boot = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.median(sigma[idx])) - obs_median
    return float(np.mean(np.abs(boot) >= abs(obs_stat)))


__all__ = ["enumerate_specs", "run_spec_curve", "bootstrap_pvalue_median"]
