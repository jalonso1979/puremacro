"""Canova-Ciccarelli (2013) mean-group panel SVAR.

For each country i in the panel, estimate a SVAR(p) and compute
structural IRFs Phi_i^h for h = 0 ... H. Aggregate via simple
cross-sectional average (mean-group estimator). Uncertainty bands
come from the cross-country distribution of country-level IRFs.

This canonical port supports the two non-bootstrap identification
schemes: ``cholesky`` and ``bq``. For ``proxy``, ``maxshare``, or
``rigobon`` (which need per-country bootstrap kwargs), call the
canonical ``var/identify/<scheme>`` module per-country directly.

References
----------
Canova, F. and Ciccarelli, M. (2013). Panel vector autoregressive
    models: a survey. Advances in Econometrics 32, 205-246.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import PanelSVARResult
from .cholesky import cholesky_factor
from .bq import _bq_impact


def _identify_country(
    A_list, Sigma, *, identification: str, horizon: int, **id_kwargs
) -> np.ndarray:
    """Run one country's identification. Returns IRF shape (H+1, n, n)."""
    if identification == "cholesky":
        B = cholesky_factor(Sigma)
        return compute_irf(A_list, B, horizon)
    if identification == "bq":
        B = _bq_impact(
            A_list, Sigma,
            permanent_var_idx=id_kwargs.get("permanent_var_idx", 0),
        )
        # compute_irf returns (H+1, n, n); cumsum along horizon axis (0)
        return np.cumsum(compute_irf(A_list, B, horizon), axis=0)
    raise ValueError(
        f"Unsupported identification scheme: {identification!r}. "
        "Supported: 'cholesky', 'bq'. For 'proxy', 'maxshare', or "
        "'rigobon', call the canonical var.identify.<scheme> module "
        "per-country directly."
    )


def mean_group_svar(
    panel_data: dict[str, np.ndarray],
    *,
    p: int,
    horizon: int,
    identification: str = "cholesky",
    ci: float = 0.9,
    seed: int = 0,
    **id_kwargs: Any,
) -> PanelSVARResult:
    """Mean-group panel SVAR estimator.

    Parameters
    ----------
    panel_data : dict mapping country_id -> ndarray (T_i, n)
        Per-country time series. All countries must share the same n
        but may have different T_i.
    p : int
        VAR lag order (common across countries).
    horizon : int
        IRF horizon H.
    identification : str, default ``"cholesky"``
        Identification scheme; one of ``'cholesky'`` or ``'bq'``.
    ci : float, default 0.9
        Coverage for cross-country percentile bands.
    seed : int, default 0
        RNG seed; currently unused (reserved for bootstrap extension).
    **id_kwargs
        Extra kwargs for the identification scheme (e.g.
        ``permanent_var_idx`` for ``bq``).

    Returns
    -------
    PanelSVARResult

    Raises
    ------
    ValueError
        If ``identification`` is not one of the supported schemes.
    """
    country_ids = tuple(sorted(panel_data.keys()))
    country_irfs = []

    for cid in country_ids:
        Y = panel_data[cid]
        A_list, _, Sigma, _, _ = estimate_var(Y, p)
        irf = _identify_country(
            A_list, Sigma,
            identification=identification, horizon=horizon, **id_kwargs,
        )
        country_irfs.append(irf)

    country_irfs_arr = np.stack(country_irfs, axis=0)  # (N, H+1, n, n)
    irf_mean = country_irfs_arr.mean(axis=0)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    irf_lower = np.quantile(country_irfs_arr, lo_q, axis=0)
    irf_upper = np.quantile(country_irfs_arr, hi_q, axis=0)

    return PanelSVARResult(
        irf_mean=irf_mean,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        country_irfs=country_irfs_arr,
        country_ids=country_ids,
        identification=identification,
        p=p,
        horizon=horizon,
        ci=ci,
    )


__all__ = ["mean_group_svar"]
