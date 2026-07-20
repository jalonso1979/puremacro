"""Canova-Ciccarelli (2013) mean-group panel SVAR.

For each country i in the panel:
  1. Estimate a SVAR(p) using the chosen identification scheme.
  2. Compute structural IRFs  Phi_i^h  for h = 0 ... H.

Mean-group aggregation (Pesaran-Smith 1995):
  Phi_bar^h = (1/N) * sum_i  Phi_i^h

Uncertainty bands are derived from the cross-country distribution of
country-level IRFs (percentiles across i), following Canova-Ciccarelli (2013).

Supported identification schemes
---------------------------------
'cholesky'   : Cholesky (recursive)
'bq'         : Blanchard-Quah long-run
'proxy'      : Proxy-SVAR / external instrument
'maxshare'   : Faust-Uhlig max-FEV-share
'rigobon'    : Rigobon heteroskedasticity

API
---
``mean_group_svar(panel_data, *, p, horizon, identification='cholesky', **id_kwargs)``

References
----------
Canova, F. and Ciccarelli, M. (2013). Panel vector autoregressive models: a survey.
    Advances in Econometrics, 32, 205-246.
Pesaran, M.H. and Smith, R. (1995). Estimating long-run relationships from
    dynamic heterogeneous panels. J. Econometrics 68(1), 79-113.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._linalg import safe_cholesky
from .estimate import estimate_var
from .irf import irf as compute_irf


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #

@dataclass
class PanelSVARResult:
    """Output of mean-group panel SVAR estimation.

    Attributes
    ----------
    irf_mean : ndarray (n, n, H+1)
        Mean-group IRF: simple average across country-level IRFs.
    irf_lo : ndarray (n, n, H+1)
        Lower band from cross-country percentile distribution.
    irf_hi : ndarray (n, n, H+1)
        Upper band from cross-country percentile distribution.
    country_irfs : ndarray (N, n, n, H+1)
        Country-level IRFs stacked along the first axis.
    country_ids : list of str
        Country identifiers in the order they appear in ``country_irfs``.
    identification : str
        Identification scheme used.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon H.
    ci : float
        Confidence level for the cross-country distribution bands.
    """
    irf_mean: np.ndarray        # (N, H+1, n, n) -> averaged (H+1, n, n)
    irf_lo: np.ndarray          # (H+1, n, n)
    irf_hi: np.ndarray          # (H+1, n, n)
    country_irfs: np.ndarray    # (N, H+1, n, n)
    country_ids: list
    identification: str
    p: int
    horizon: int
    ci: float


# --------------------------------------------------------------------------- #
# Internal: country-level IRF extraction
# --------------------------------------------------------------------------- #

def _cholesky_irf(Y: np.ndarray, p: int, horizon: int, **kwargs) -> np.ndarray:
    """Cholesky-identified IRF for a single country. Returns (H+1, n, n)."""
    ordering = kwargs.get("ordering", None)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    if ordering is not None:
        perm = np.array(ordering)
        Sigma_p = Sigma[np.ix_(perm, perm)]
        A_list_p = [A[np.ix_(perm, perm)] for A in A_list]
        P_p = safe_cholesky(Sigma_p, name="panel_var permuted Σ")
        irf_p = compute_irf(A_list_p, P_p, horizon)   # (H+1, n, n)
        inv = np.argsort(perm)
        irf_out = irf_p[np.ix_(np.arange(horizon + 1), inv, inv)]
    else:
        P = safe_cholesky(Sigma, name="panel_var Σ")
        irf_out = compute_irf(A_list, P, horizon)
    return irf_out


def _bq_irf(Y: np.ndarray, p: int, horizon: int, **kwargs) -> np.ndarray:
    """BQ long-run-identified IRF. Returns (H+1, n, n)."""
    from .identify.bq import _bq_impact
    permanent_var_idx = kwargs.get("permanent_var_idx", 0)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    B = _bq_impact(A_list, Sigma, permanent_var_idx)
    return np.cumsum(compute_irf(A_list, B, horizon), axis=0)


def _proxy_irf(Y: np.ndarray, p: int, horizon: int, **kwargs) -> np.ndarray:
    """Proxy-SVAR IRF. Returns (H+1, n, n).

    Required kwarg: ``instrument_series`` (array-like, length >= T_eff).
    """
    from .identify.proxy import _proxy_impact_factory
    instrument_series = kwargs["instrument_series"]
    shock_target_idx = kwargs.get("shock_target_idx", 0)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    impact_fn = _proxy_impact_factory(instrument_series, shock_target_idx)
    B = impact_fn(A_list, Sigma, resid)
    return compute_irf(A_list, B, horizon)


def _rigobon_irf(Y: np.ndarray, p: int, horizon: int, **kwargs) -> np.ndarray:
    """Rigobon heteroskedasticity IRF. Returns (H+1, n, n).

    Required kwarg: ``regime_indicator`` (array-like, length T or T-p).
    """
    from .identify.hetero import (
        _regime_covariances,
        _rigobon_impact,
    )
    regime_indicator = kwargs["regime_indicator"]
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]
    ri = np.asarray(regime_indicator, dtype=int)
    if ri.shape[0] == Y.shape[0]:
        ri = ri[p:]
    if ri.shape[0] != T_eff:
        ri = ri[:T_eff] if ri.shape[0] > T_eff else np.resize(ri, T_eff)
    Sigma_0, Sigma_1 = _regime_covariances(resid, ri)
    B, _ = _rigobon_impact(Sigma_0, Sigma_1)
    return compute_irf(A_list, B, horizon)


_IDENTIFICATION_MAP = {
    "cholesky": _cholesky_irf,
    "bq":       _bq_irf,
    "proxy":    _proxy_irf,
    "rigobon":  _rigobon_irf,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

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
    """Canova-Ciccarelli mean-group panel SVAR estimator.

    Parameters
    ----------
    panel_data : dict mapping country_id -> ndarray (T_i, n)
    p : int
        VAR lag order (common across countries).
    horizon : int
        IRF horizon H.
    identification : str
        One of 'cholesky', 'bq', 'proxy', 'rigobon'.
    ci : float
        Coverage for cross-country distribution bands.
    seed : int
        Random seed.
    **id_kwargs
        Forwarded to the country-level identification function.

    Returns
    -------
    PanelSVARResult
    """
    if identification not in _IDENTIFICATION_MAP:
        raise KeyError(
            f"Unknown identification '{identification}'. "
            f"Choose from {sorted(_IDENTIFICATION_MAP.keys())}."
        )

    irf_fn = _IDENTIFICATION_MAP[identification]

    country_ids = list(panel_data.keys())
    N = len(country_ids)
    if N < 2:
        raise ValueError("Panel must contain at least 2 countries.")

    n_vars = {v.shape[1] for v in panel_data.values()}
    if len(n_vars) > 1:
        raise ValueError(
            f"All countries must have the same number of variables; found {n_vars}."
        )
    n = n_vars.pop()

    country_irfs_list: list[np.ndarray] = []
    for cid in country_ids:
        Y_i = panel_data[cid]
        try:
            irf_i = irf_fn(Y_i, p, horizon, **id_kwargs)  # (H+1, n, n)
        except (np.linalg.LinAlgError, ValueError, KeyError) as exc:
            import warnings
            warnings.warn(
                f"Country '{cid}' identification failed ({exc!r}); "
                "substituting zeros.",
                stacklevel=2,
            )
            irf_i = np.zeros((horizon + 1, n, n))
        country_irfs_list.append(irf_i)

    country_irfs = np.stack(country_irfs_list, axis=0)  # (N, H+1, n, n)

    irf_mean = country_irfs.mean(axis=0)  # (H+1, n, n)

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 - (1.0 - ci) / 2.0) * 100.0
    irf_lo = np.percentile(country_irfs, lo_q, axis=0)
    irf_hi = np.percentile(country_irfs, hi_q, axis=0)

    return PanelSVARResult(
        irf_mean=irf_mean,
        irf_lo=irf_lo,
        irf_hi=irf_hi,
        country_irfs=country_irfs,
        country_ids=country_ids,
        identification=identification,
        p=p,
        horizon=horizon,
        ci=ci,
    )
