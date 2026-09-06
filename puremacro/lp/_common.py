"""Shared keyword handling for the ``puremacro.lp`` estimators.

Every estimator in :mod:`puremacro.lp` accepts the canonical arguments
``horizons`` / ``n_lags`` / ``alpha`` **and** the 2.0 aliases
``horizon`` / ``lags`` / ``ci``.  :func:`resolve_lp_kwargs` applies the
aliases and validates the result so that a wrong *scale* (``ci=90``
instead of ``ci=0.90``, ``alpha=1.5``) raises immediately instead of
silently producing all-NaN confidence bands.
"""
from __future__ import annotations

from typing import Iterable


def _is_int_like(v: object) -> bool:
    if isinstance(v, bool):
        return False
    try:
        return float(v).is_integer()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def resolve_lp_kwargs(
    horizons: Iterable[int],
    n_lags: int,
    alpha: float,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    name: str = "lp",
) -> tuple[list[int], int, float]:
    """Apply the ``lags`` / ``horizon`` / ``ci`` aliases and validate.

    Parameters
    ----------
    horizons, n_lags, alpha
        Canonical arguments as received by the estimator.
    lags
        Alias for ``n_lags`` (wins when both are given).
    horizon
        Maximum horizon ``H``; sets ``horizons = range(0, H + 1)``.
    ci
        Coverage of the confidence bands in ``(0, 1)``; sets
        ``alpha = 1 - ci``.
    name
        Estimator name used in error messages.

    Returns
    -------
    (horizons, n_lags, alpha)
        ``horizons`` as a list of ints, ``n_lags`` as an int and
        ``alpha`` as a float in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``ci`` or ``alpha`` is not a probability in ``(0, 1)`` — the
        common mistake is passing a percentage (``ci=90``) — or if
        ``horizon`` / ``lags`` is not a non-negative integer.
    """
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        if not _is_int_like(horizon) or int(horizon) < 0:
            raise ValueError(
                f"{name}: horizon must be a non-negative integer (the maximum "
                f"horizon H, estimating h = 0..H); got {horizon!r}")
        horizons = range(0, int(horizon) + 1)
    if ci is not None:
        if not (0.0 < float(ci) < 1.0):
            raise ValueError(
                f"{name}: ci must be a coverage probability in (0, 1), e.g. "
                f"ci=0.90 for 90% bands; got {ci!r}")
        alpha = 1.0 - float(ci)
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            f"{name}: alpha must be a two-sided significance level in (0, 1), "
            f"e.g. alpha=0.10 for 90% bands; got {alpha!r}")
    if not _is_int_like(n_lags) or int(n_lags) < 0:
        raise ValueError(
            f"{name}: lags must be a non-negative integer; got {n_lags!r}")
    hs = [int(h) for h in horizons]
    return hs, int(n_lags), float(alpha)


__all__ = ["resolve_lp_kwargs"]
