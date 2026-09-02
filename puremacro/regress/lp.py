"""Panel local projection estimator with Driscoll-Kraay standard errors.

NOTE (2026-05-18, 0.43.0): this module is an INDEPENDENT pure-numpy
implementation of panel LP; it is NOT a thin re-export of
``puremacro.lp.panel.panel_lp``. The two share neither signature
(``shock`` / ``unit`` / ``date`` here vs canonical ``x`` / ``entity_level`` /
``time_level``) nor return shape (``horizon`` / ``ci_lo`` / ``ci_hi`` here vs
canonical ``h`` / ``lo`` / ``hi``). Active callers
(``tools/run_logurate_revision.py``, ``tools/run_paper_extensions.py``) use
this module in a manual "split-then-compare" regime pattern that
``panel_lp`` does not support. Retirement deferred to a future release
once a canonical equivalent with the same signature ships.

Jordà (2005) local projections in a panel setting. For each horizon h:

    y_{i, t+h} = alpha_i + beta_h * shock_t + gamma' x_{t-1} + u_{i,t+h}

beta_h is the impulse response of y to the shock at horizon h.
Driscoll-Kraay (1998) SEs are robust to cross-sectional dependence and
serial correlation; the truncation lag defaults to h+1.

Pure-numpy implementation, pyodide-clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._linalg import inv_xtx


def _dk_default_lag(horizon: int) -> int:
    """Default Driscoll-Kraay truncation: h + 1 (covers LP overlap)."""
    return horizon + 1


def _newey_west_weight(j: int, lag: int) -> float:
    """Bartlett kernel weight: 1 - j/(lag+1)."""
    return 1.0 - j / (lag + 1)


def _within_demean(y: np.ndarray, unit_idx: np.ndarray) -> np.ndarray:
    """Subtract per-unit mean from y. unit_idx is integer-coded.

    `np.bincount` is the same sum-by-group reduction the Python loop was doing,
    in C. It is called `K + 1` times per horizon, so a 13-horizon run with four
    regressors made 65 full Python passes over the panel and accounted for 58%
    of `lp_panel`'s runtime on a 15,300-row frame. Bit-identical output.
    """
    n_units = int(unit_idx.max()) + 1
    sums = np.bincount(unit_idx, weights=y, minlength=n_units)
    counts = np.bincount(unit_idx, minlength=n_units).astype(float)
    means = sums / np.maximum(counts, 1.0)
    return y - means[unit_idx]


def _ols_with_dk_se(
    Y: np.ndarray,           # (N,)
    X: np.ndarray,           # (N, K)
    time_idx: np.ndarray,    # (N,) integer time-period codes
    dk_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled OLS with Driscoll-Kraay covariance.

    Returns (beta_hat, var_beta_hat) as length-K arrays.
    """
    N, K = X.shape
    # Route the normal-equations inverse through the package helper, per
    # CONTRIBUTING: a rank-deficient X'X gets a named diagnostic here instead
    # of a silently wrong `inv`.
    XtX_inv = inv_xtx(X, name="lp_panel")
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    # h_t: time-aggregated moment conditions (1xK per period). Summing the
    # score by period is a bincount per regressor -- the row loop it replaces
    # was 36% of `lp_panel`'s runtime. Bit-identical output.
    T = int(time_idx.max() + 1)
    scores = X * resid[:, None]
    h_t = np.empty((T, K))
    for k in range(K):
        h_t[:, k] = np.bincount(time_idx, weights=scores[:, k], minlength=T)
    # Newey-West on h_t: S = Sum_j w_j * (Gamma_j + Gamma_j')
    Gamma_0 = h_t.T @ h_t
    S = Gamma_0.copy()
    for j in range(1, dk_lag + 1):
        if j >= T:
            break
        Gamma_j = h_t[j:].T @ h_t[:-j]
        w = _newey_west_weight(j, dk_lag)
        S = S + w * (Gamma_j + Gamma_j.T)
    var_beta = XtX_inv @ S @ XtX_inv
    return beta, var_beta


def lp_panel(
    panel: pd.DataFrame,
    *,
    y: str,
    shock: str,
    horizons=range(0, 13),
    unit: str = "unit",
    date: str = "date",
    unit_fe: bool = True,
    controls: list[str] | None = None,
    se: str = "driscoll_kraay",
    dk_lag: int | None = None,
    dummies: list[str] | None = None,
) -> pd.DataFrame:
    """Run panel LP for each horizon h in horizons.

    Returns long-format DataFrame with columns:
        horizon, beta, se, t, p, ci_lo, ci_hi, n_obs
    """
    if se != "driscoll_kraay":
        raise NotImplementedError(f"se={se!r} not implemented")

    df = panel.copy()
    df = df.sort_values([unit, date]).reset_index(drop=True)

    unit_codes, unit_uniq = pd.factorize(df[unit])
    date_codes, date_uniq = pd.factorize(df[date])
    df["_unit_idx"] = unit_codes
    df["_date_idx"] = date_codes

    out_rows = []
    controls = controls or []
    dummies = dummies or []
    regressor_names = [shock] + controls + dummies

    for h in horizons:
        # Build (i, t) -> (i, t+h) mapping. y_lead at row (i,t) = y at (i, t+h).
        df_h = df.copy()
        df_h["y_lead"] = df_h.groupby("_unit_idx")[y].shift(-h)
        df_h = df_h.dropna(subset=["y_lead", shock] + controls + dummies)
        if df_h.empty:
            continue

        Y = df_h["y_lead"].to_numpy(dtype=float)
        X_cols = [df_h[c].to_numpy(dtype=float).reshape(-1, 1) for c in regressor_names]
        X = np.hstack(X_cols)

        if unit_fe:
            Y = _within_demean(Y, df_h["_unit_idx"].to_numpy())
            X = np.column_stack([
                _within_demean(X[:, k], df_h["_unit_idx"].to_numpy())
                for k in range(X.shape[1])
            ])

        lag = dk_lag if dk_lag is not None else _dk_default_lag(h)
        beta, var_beta = _ols_with_dk_se(
            Y, X, df_h["_date_idx"].to_numpy(), lag,
        )
        b = float(beta[0])
        se_b = float(np.sqrt(max(var_beta[0, 0], 0.0)))
        t_stat = b / se_b if se_b > 0 else np.nan
        # 90% CI
        ci_z = 1.6448536269514722
        out_rows.append({
            "horizon": h,
            "beta": b,
            "se": se_b,
            "t": t_stat,
            "p": 2.0 * (1.0 - _norm_cdf(abs(t_stat))) if se_b > 0 else np.nan,
            "ci_lo": b - ci_z * se_b,
            "ci_hi": b + ci_z * se_b,
            "n_obs": int(len(Y)),
        })

    return pd.DataFrame(out_rows)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (math.erf is in stdlib)."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


__all__ = ["lp_panel", "_dk_default_lag"]
