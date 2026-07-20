"""Jarociński-Karadi (2020) monetary-vs-information decomposition.

The high-frequency surprise around a central-bank announcement reflects two
shocks: a "monetary policy" shock (Taylor-rule innovation) and a "central bank
information" shock (the central bank revealing private information about the
economy). JK 2020 separate them via sign restrictions on the joint reaction of
the policy rate and a broad asset price (typically S&P 500).

Two variants are shipped in 0.4.0:
    - :func:`jk_poor_man`        — sign-of-comovement attribution.
    - :func:`jk_median_target`   — median admissible rotation under
                                    sign restrictions.

The full Bayesian sign-restriction variant is deferred to 0.5.0+.
"""
from __future__ import annotations

import numpy as np

from ._results import JKResult


def jk_poor_man(
    rate_surprise: np.ndarray,
    asset_surprise: np.ndarray,
) -> JKResult:
    """Jarociński-Karadi (2020) "poor-man's" decomposition.

    Attributes each announcement to either a monetary-policy shock or a
    central-bank-information shock based on the sign comovement of the rate
    and asset surprises:

    - Opposite-sign (rate up + asset down, or rate down + asset up) → MP shock.
    - Same-sign (both up or both down) → information shock.

    Within each category, the rate surprise carries through; the other category
    is zero at that announcement.

    Parameters
    ----------
    rate_surprise : ndarray, shape (T,)
        High-frequency interest-rate surprise.
    asset_surprise : ndarray, shape (T,)
        High-frequency broad-asset (e.g., S&P 500) surprise in the same window.

    Returns
    -------
    JKResult with ``method="poor_man"`` and ``rotation=None``, ``n_admissible=None``.
    """
    rate = np.asarray(rate_surprise, dtype=float)
    asset = np.asarray(asset_surprise, dtype=float)
    if rate.shape != asset.shape:
        raise ValueError(
            f"jk_poor_man: rate {rate.shape} and asset {asset.shape} must match"
        )
    same_sign = (rate * asset) > 0
    opp_sign = (rate * asset) < 0
    mp = np.where(opp_sign, rate, 0.0)
    info = np.where(same_sign, rate, 0.0)
    return JKResult(
        mp_shock=mp,
        info_shock=info,
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )


def _2d_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def jk_median_target(
    rate_surprise: np.ndarray,
    asset_surprise: np.ndarray,
    n_rotations: int = 10_000,
    seed: int | None = None,
) -> JKResult:
    """Jarociński-Karadi (2020) median-target sign-restriction decomposition.

    Searches the 2x2 orthogonal rotation group for rotations U such that the
    decomposed shocks satisfy:

    - column 0 (MP shock) : impact on rate > 0, impact on asset < 0
    - column 1 (info shock): impact on rate > 0, impact on asset > 0

    The *median* admissible rotation (Fry-Pagan 2011) is selected, and the
    surprise vector is rotated to produce the two shock series.

    Parameters
    ----------
    rate_surprise : ndarray, shape (T,)
        High-frequency interest-rate surprise.
    asset_surprise : ndarray, shape (T,)
        High-frequency asset-price (e.g., S&P 500) surprise in the same window.
    n_rotations : int, default 10_000
        Number of rotations to draw uniformly on [0, 2π) when searching the
        admissible set.
    seed : int or None
        RNG seed.

    Returns
    -------
    JKResult with ``method="median_target"``, ``rotation`` filled, and
    ``n_admissible`` reporting how many of ``n_rotations`` satisfied the
    sign restrictions.

    Notes
    -----
    For a 2x2 problem the admissible set, if non-empty, is a contiguous arc
    in θ. The median rotation is the one at the median θ within that arc.
    """
    rate = np.asarray(rate_surprise, dtype=float)
    asset = np.asarray(asset_surprise, dtype=float)
    if rate.shape != asset.shape:
        raise ValueError(
            f"jk_median_target: rate {rate.shape} and asset {asset.shape} must match"
        )
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2 * np.pi, size=n_rotations)
    # The "impact" of the shocks on (rate, asset) under rotation U is just U
    # itself, since the surprise vector IS the shock pair (no further VAR step
    # at this stage). The sign restrictions on U:
    #   col 0 (MP)  : U[0,0] > 0,  U[1,0] < 0
    #   col 1 (info): U[0,1] > 0,  U[1,1] > 0
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # U = [[cos, -sin], [sin, cos]]
    admissible = (cos_t > 0) & (sin_t < 0) & (-sin_t > 0) & (cos_t > 0)
    n_adm = int(admissible.sum())
    if n_adm == 0:
        raise ValueError(
            "jk_median_target: no admissible rotations found. Check sign "
            "conventions on rate_surprise and asset_surprise."
        )
    theta_med = float(np.median(thetas[admissible]))
    U = _2d_rotation(theta_med)
    # Rotate the (rate, asset) pair into shock space: shocks = (rate, asset) @ U
    shocks = np.column_stack([rate, asset]) @ U
    return JKResult(
        mp_shock=shocks[:, 0],
        info_shock=shocks[:, 1],
        rotation=U,
        n_admissible=n_adm,
        method="median_target",
    )
