"""Lag-augmented local projections (Montiel Olea & Plagborg-Møller, 2021).

Reference: Montiel Olea, J. L. and M. Plagborg-Møller (2021), "Local
Projection Inference Is Simpler and More Robust Than You Think",
*Econometrica* 89(4), 1789-1823, doi:10.3982/ECTA18756 (arXiv:2007.13888).
Section, equation and page numbers below refer to arXiv v4.

Estimator
---------
For every horizon ``h`` in ``horizons`` :func:`la_lp` runs one OLS
regression on the rows where all terms are observed:

    y_{t+h} = mu_h + beta_h x_t
              + sum_{l=1}^{p_aug} (a_{h,l} x_{t-l} + b_{h,l} y_{t-l})
              + sum_c (g_{h,c} c_t + sum_{l=1}^{q} d_{h,c,l} c_{t-l})
              + xi_{t,h},

with ``p_aug = n_lags + extra_lags`` and ``q = control_lags`` (default
``p_aug``). ``beta_h`` is the response of ``y`` at ``t+h`` (in the units of
``y``) to a one-unit change in ``x_t``: the response to an innovation in
``x`` ordered after the controls ``c`` and before ``y`` in a recursive VAR.
With ``y == x`` it is the response of a series to its own one-step
innovation, MOPM's AR(1) design (§2.1, eq. 3). When ``p_aug >= 1`` the
regression is run on ``y_{t+h} - y_{t-1}``, which gives the same ``beta_h``
and residuals because ``y_{t-1}`` is a regressor.

Inference
---------
The standard error is the Eicker-Huber-White (HC0) standard error of
``beta_h`` (MOPM eq. 5 for the AR(1); the formula below eq. 12 for a VAR(p)),
and the interval is ``beta_h ± z_{1-alpha/2} se``. No HAC/HAR correction is
applied. The multi-step residual ``xi_{t,h}`` is serially correlated, but once
the regression controls for enough lags that the regressor of interest,
residualized on the controls, is the one-step innovation, the regression
scores are serially uncorrelated (MOPM eq. 7), so the EHW standard error is
valid - under MOPM's assumptions, uniformly over the persistence of the
data, unit roots included, and over horizons ``h <= h_T`` with
``h_T / T -> 0`` (MOPM §2.1 and Proposition 1). In MOPM's Monte Carlo
(Table 1: AR(1), T = 240, 90% intervals) the coverage of these intervals is
0.878/0.838/0.806/0.814/0.833 at h = 1/6/12/36/60 for rho = 0.95; the
percentile-t bootstrap of their §5 is closer to nominal and is not
implemented here.
MOPM's replication code (github.com/jm4474/Lag-augmented_LocalProjections,
``functions/linreg.m``) additionally multiplies the EHW variance by
``T/(T-k)`` (Stata's ``regress, robust``); this module reports eq. (5) without
that factor, so its standard errors are smaller by ``sqrt((T-k)/T)``.

How many lags
-------------
Lag augmentation adds *one* lag beyond what the LP representation needs, the
same at every horizon. In the AR(1) the LA-LP regresses ``y_{t+h}`` on ``y_t``
and "uses y_{t-1} as an additional control variable" (§2.1, p. 7). In a
VAR(p) it regresses on ``y_t`` and controls for ``y_{t-1}, ..., y_{t-p}``:
"the population regression coefficients on the last n control variables
y_{t-p} equal zero. Thus, we are including one additional lag" (§4.1,
p. 20). The augmentation applies to *all* the series of the VAR ("controls
for p lags of all the time series that enter into the VAR model", §4, p. 19),
which is why the controls get ``p_aug`` lags by default. ``extra_lags=1``
(the default) is that single extra lag on top of the ``n_lags`` lags of the
non-augmented LP; the reference code does the same
(``functions/ir_estim.m``: ``lp(Y, p-1+lag_aug, ...)``). MOPM advise choosing
the lag length conservatively - "there is no asymptotic efficiency cost of
controlling for more than p0 lags" (§6, p. 28) - so a larger ``n_lags`` is
the robustness knob. No source prescribes ``p + h`` or ``p + max(h)`` lags.

.. versionchanged:: after 4.3.0
   The default was ``extra_lags = max(horizons)`` lags at every horizon (so
   the estimate at a given ``h`` depended on the largest horizon requested,
   and ``horizons=[0]`` had no augmentation), controls entered with
   ``n_lags`` lags only, and the method was credited to Plagborg-Møller and
   Wolf (2021), a paper on the population equivalence of LPs and VARs. The
   old specification is ``extra_lags=max(horizons), control_lags=n_lags``.

LP-IV
-----
When an instrument ``z`` is supplied, the same regressors form the exogenous
block of a lag-augmented LP-IV: HC0 two-stage least squares, the
heteroskedasticity-robust Montiel Olea & Pflueger (2013) effective F
statistic, and White-robust Anderson-Rubin confidence sets. MOPM's theory
covers the OLS case; the IV variant is a puremacro extension.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .._linalg import inv_xtx
from ..inference._ols_helpers import tsls_hac
from ._common import resolve_lp_kwargs
from ._results import LPResult
from .iv import _compute_anderson_rubin_multi, mop_critical_values


def _ols_eicker_huber(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with Eicker-Huber-White (HC0) heteroskedasticity-robust SE."""
    XtX_inv = inv_xtx(X, name="la_lp")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = (X * u[:, None]).T @ (X * u[:, None])
    V = XtX_inv @ meat @ XtX_inv
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "vcov": V, "resid": u}


def _compute_white_ar_ci(
    dy: np.ndarray,
    x: np.ndarray,
    Z_mat: np.ndarray,
    W_ctl_mat: np.ndarray,
    alpha: float,
    beta_hat: float = 0.0,
    se_hat: float = 1.0,
) -> tuple[float, float, str]:
    """Eicker-Huber-White robust Anderson-Rubin confidence set.

    ``k_z = 1``: exact quadratic inversion (the White special case of
    :func:`puremacro.lp.iv._compute_anderson_rubin_ci`). ``k_z >= 2``: the
    exact inversion of :func:`puremacro.lp.iv._compute_anderson_rubin_multi`
    with ``lags = 0`` (White ``Omega = S'S`` is the zero-lag Bartlett case).
    """
    k_z = Z_mat.shape[1]
    if k_z != 1:
        return _compute_anderson_rubin_multi(
            dy, x, W_ctl_mat, Z_mat, lags=0, alpha=alpha, beta_hat=beta_hat, se_hat=se_hat
        )

    W_inv = np.linalg.pinv(W_ctl_mat.T @ W_ctl_mat)
    y_tilde = dy - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ dy))
    x_tilde = x - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ x))
    Z_tilde = Z_mat - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ Z_mat))

    z_t = Z_tilde[:, 0]
    ztz = float(z_t @ z_t)
    if ztz <= 1e-14:
        return np.nan, np.nan, "empty"
    gamma_y = float(z_t @ y_tilde / ztz)
    gamma_x = float(z_t @ x_tilde / ztz)
    u_y = y_tilde - z_t * gamma_y
    u_x = x_tilde - z_t * gamma_x
    h_vec = z_t / ztz
    s_y = h_vec * u_y
    s_x = h_vec * u_x
    V_yy = float(np.dot(s_y, s_y))
    V_xx = float(np.dot(s_x, s_x))
    V_yx = float(np.dot(s_y, s_x))
    crit = float(chi2.ppf(1.0 - alpha, df=1))

    A = gamma_x ** 2 - crit * V_xx
    B = -2.0 * (gamma_y * gamma_x - crit * V_yx)
    C = gamma_y ** 2 - crit * V_yy

    disc = B ** 2 - 4.0 * A * C
    if A > 0:
        if disc >= 0:
            r1 = (-B - np.sqrt(disc)) / (2.0 * A)
            r2 = (-B + np.sqrt(disc)) / (2.0 * A)
            return float(min(r1, r2)), float(max(r1, r2)), "bounded"
        return np.nan, np.nan, "empty"
    if disc >= 0:
        r1 = (-B - np.sqrt(disc)) / (2.0 * A)
        r2 = (-B + np.sqrt(disc)) / (2.0 * A)
        return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"
    return -np.inf, np.inf, "all_real"


#: MOPM (2021) lag augmentation: one lag beyond the non-augmented LP.
_MOPM_EXTRA_LAGS = 1


def _nonneg_int(value: Any, name: str) -> int:
    """Return ``value`` as an ``int >= 0`` or raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            value = int(value)
        else:
            raise ValueError(
                f"la_lp: {name} must be a non-negative integer; got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(
            f"la_lp: {name} must be a non-negative integer; got {value!r}")
    return value


def la_lp(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 4,
    extra_lags: int | None = _MOPM_EXTRA_LAGS,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    z: str | Sequence[str] | None = None,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    control_lags: int | None = None,
) -> LPResult:
    """Lag-augmented local projection with Eicker-Huber-White standard errors.

    Montiel Olea and Plagborg-Møller (2021, *Econometrica* 89(4), 1789-1823;
    "MOPM"). For each ``h`` the OLS regression is

        y_{t+h} = mu_h + beta_h x_t
                  + sum_{l=1}^{p_aug} (a_{h,l} x_{t-l} + b_{h,l} y_{t-l})
                  + sum_c (g_{h,c} c_t + sum_{l=1}^{q} d_{h,c,l} c_{t-l})
                  + xi_{t,h},

    ``p_aug = n_lags + extra_lags`` and ``q = control_lags`` (default
    ``p_aug``); the same ``p_aug`` is used at every horizon. ``beta_h`` is
    reported with its HC0 standard error (MOPM eq. 5) and the normal interval
    ``beta_h ± z_{1-alpha/2} se``; no HAC correction. See the module
    docstring for the derivation and the page references.

    Parameters
    ----------
    df : pd.DataFrame
        Time series in rows (regular frequency, sorted), one column per
        variable. Lags and leads are row shifts.
    y : str
        Outcome variable. ``beta_h`` is in the units of ``y`` per unit of
        ``x``.
    x : str
        Shock / policy variable, entering at ``t``. ``y == x`` gives the
        response of a series to its own one-step innovation (the AR(p) case;
        the common lags then enter once): with ``n_lags=0`` and the default
        ``extra_lags=1`` this is MOPM's AR(1) lag-augmented LP, the
        regression of ``y_{t+h}`` on ``(1, y_t, y_{t-1})`` in their eq. (3)
        and Table 1.
    horizons : iterable of int, default range(0, 21)
        Horizons ``h >= 0`` in periods of the data frequency.
    n_lags : int, default 4
        Lags of ``x`` and ``y`` in the non-augmented LP. For a VAR(p) the
        non-augmented LP needs ``p - 1`` lags besides the current values
        (MOPM eq. 10), so ``n_lags = p - 1`` with the default
        ``extra_lags=1`` is MOPM's specification (``p`` lags of every
        series). A larger ``n_lags`` is the conservative choice MOPM
        recommend (§6, p. 28: no asymptotic efficiency cost).
    extra_lags : int, default 1
        Augmentation lags added to ``n_lags``, the same at every horizon.
        ``1`` is MOPM's lag augmentation ("we are including one additional
        lag", §4.1, p. 20). ``None`` means the default. ``0`` switches the
        augmentation off; the EHW standard error is then justified only if
        ``n_lags`` already includes one lag more than the LP needs (``p``
        for a VAR(p)), and a ``UserWarning`` is issued. Larger values are allowed (e.g.
        ``extra_lags=max(horizons)`` reproduces the pre-fix default) but are
        not MOPM's specification.
    controls : sequence of str, optional
        Additional series ``c``, entering at ``t`` (ordered before ``x`` in a
        recursive scheme) and with ``control_lags`` lags.
    alpha : float, default 0.10
        Two-sided significance level (0.10 -> 90% intervals, as in MOPM
        Table 1).
    z : str or sequence of str, optional
        External instrument(s). If provided, runs lag-augmented LP-IV with the
        same regressors as exogenous controls (a puremacro extension; MOPM
        treat the OLS case).
    anderson_rubin : bool, default False
        If True (when z is provided), computes White-robust Anderson-Rubin
        confidence sets.
    weak_iv_robust : bool, default False
        Alias for anderson_rubin.
    lags : int, optional
        Alias for n_lags.
    horizon : int, optional
        Sets horizons = range(0, horizon + 1).
    ci : float, optional
        Confidence level in (0, 1), e.g. 0.90 for 90% CI.
    control_lags : int, optional
        Lags of each control. Default ``p_aug``: MOPM's LA-LP "controls for p
        lags of all the time series that enter into the VAR model" (§4,
        p. 19), so controls are augmented like ``x`` and ``y``.
        ``control_lags=n_lags`` reproduces the pre-fix treatment.

    Returns
    -------
    LPResult
        One row per horizon: ``h, beta, se, lo, hi, p_aug`` (plus the IV
        columns when ``z`` is given). The lag specification is recorded as
        attributes ``res.n_lags``, ``res.extra_lags``, ``res.control_lags``
        and in ``res.attrs`` (keys ``n_lags``, ``extra_lags``, ``p_aug``,
        ``control_lags``), which survive pandas operations.

    Raises
    ------
    ValueError
        If ``n_lags``, ``extra_lags`` or ``control_lags`` is not a
        non-negative integer.

    References
    ----------
    Montiel Olea, J. L. and M. Plagborg-Møller (2021). Local Projection
    Inference Is Simpler and More Robust Than You Think. *Econometrica*
    89(4), 1789-1823. doi:10.3982/ECTA18756.
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="la_lp"
    )
    horizons = list(horizons)
    n_lags = _nonneg_int(n_lags, "n_lags")
    if extra_lags is None:
        extra_lags = _MOPM_EXTRA_LAGS
    extra_lags = _nonneg_int(extra_lags, "extra_lags")
    p_aug = n_lags + extra_lags
    control_lags = p_aug if control_lags is None else _nonneg_int(control_lags, "control_lags")
    if extra_lags == 0:
        warnings.warn(
            "la_lp: extra_lags=0 switches off the Montiel Olea-Plagborg-Møller "
            "(2021) lag augmentation; the Eicker-Huber-White standard errors are "
            "then valid only if n_lags already includes one lag more than the "
            "local projection needs (p lags for a VAR(p), MOPM Sec. 4.1). Use "
            "extra_lags=1 (the default) for the lag-augmented LP.",
            UserWarning,
            stacklevel=2,
        )

    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)
    do_ar = anderson_rubin or weak_iv_robust
    own = y == x  # response of a series to its own innovation: lags enter once

    has_iv = z is not None
    z_names = ([z] if isinstance(z, str) else list(z)) if has_iv else []
    k_z = len(z_names)
    cv_10, cv_20 = mop_critical_values(k_z) if has_iv else (np.nan, np.nan)

    # Design built once: current values and the lags, which do not depend on h.
    base_names = list(dict.fromkeys([y, x] + z_names + ctl))
    # numpy values: row shifts are positional and nothing is index-aligned.
    design = {name: df[name].to_numpy() for name in base_names}
    lag_cols: list[str] = []  # column order of the lag block in W_ctl
    for lag in range(1, p_aug + 1):
        design[f"__{x}_L{lag}__"] = df[x].shift(lag).to_numpy()
        lag_cols.append(f"__{x}_L{lag}__")
        if not own:
            design[f"__{y}_L{lag}__"] = df[y].shift(lag).to_numpy()
            lag_cols.append(f"__{y}_L{lag}__")
    for c in ctl:
        for lag in range(1, control_lags + 1):
            design[f"__{c}_L{lag}__"] = df[c].shift(lag).to_numpy()
            lag_cols.append(f"__{c}_L{lag}__")
        lag_cols.append(c)
    design_df = pd.DataFrame(design, index=df.index)
    y_lag1 = df[y].shift(1).to_numpy()

    rows = []
    for h in horizons:
        # Outcome y_{t+h} (MOPM eqs. 3 and 12). When y_{t-1} is a regressor
        # (p_aug >= 1) the regression is run on y_{t+h} - y_{t-1}: the same
        # coefficient on x_t and the same residuals, less rounding on data in
        # levels, and bit-identical to the pre-fix code for a given lag set.
        lead = df[y].shift(-h).to_numpy()
        if p_aug >= 1:
            lead = lead - y_lag1
        sub = design_df.assign(__y_lead__=lead).dropna()
        if sub.empty:
            empty_row: dict[str, Any] = {
                "h": h,
                "beta": np.nan,
                "se": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "p_aug": p_aug,
            }
            if has_iv:
                empty_row.update(
                    {
                        "first_stage_f": np.nan,
                        "mop_f": np.nan,
                        "mop_cv_10": cv_10,
                        "mop_cv_20": cv_20,
                    }
                )
                if do_ar:
                    empty_row.update({"ar_lo": np.nan, "ar_hi": np.nan, "ar_set_type": "empty"})
            rows.append(empty_row)
            continue

        n = len(sub)
        # Exogenous block: constant + p_aug lags of x and y + controls (lags, current)
        W_ctl = np.column_stack([np.ones(n)] + [sub[col].to_numpy(float) for col in lag_cols])
        dep = sub["__y_lead__"].to_numpy(float)
        x_now = sub[x].to_numpy(float)

        if not has_iv:
            # Standard OLS lag-augmented LP
            X = np.column_stack([W_ctl, x_now])
            out = _ols_eicker_huber(dep, X)
            beta_h = float(out["beta"][-1])
            se_h = float(out["se"][-1])
            rows.append(
                {
                    "h": h,
                    "beta": beta_h,
                    "se": se_h,
                    "lo": beta_h - z_crit * se_h,
                    "hi": beta_h + z_crit * se_h,
                    "p_aug": p_aug,
                }
            )
        else:
            # IV lag-augmented LP
            Z_mat = sub[z_names].to_numpy(float)
            X_fs = np.column_stack([W_ctl, Z_mat])
            out_fs = _ols_eicker_huber(x_now, X_fs)
            x_hat = X_fs @ out_fs["beta"]

            # Compute White-robust Montiel Olea & Pflueger effective F
            W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
            x_tilde = x_now - W_ctl @ (W_inv @ (W_ctl.T @ x_now))
            Z_tilde = Z_mat - W_ctl @ (W_inv @ (W_ctl.T @ Z_mat))
            ZtZ = Z_tilde.T @ Z_tilde
            ZtZ_inv = np.linalg.pinv(ZtZ)
            pi_hat = ZtZ_inv @ (Z_tilde.T @ x_tilde)
            v_hat = x_tilde - Z_tilde @ pi_hat
            scores = Z_tilde * v_hat[:, None]
            Omega_white = scores.T @ scores

            num = float(pi_hat.T @ ZtZ @ pi_hat)
            denom = float(np.trace(ZtZ_inv @ Omega_white))
            mop_f = float(num / denom) if denom > 0 else float("inf")
            first_stage_f = mop_f

            # Second stage
            X_ss = np.column_stack([W_ctl, x_hat])
            X_actual = np.column_stack([W_ctl, x_now])
            out = tsls_hac(dep, X_actual, X_ss, lags=0)  # lags=0: HC0
            beta_h = float(out["beta"][-1])
            se_h = float(out["se"][-1])

            row = {
                "h": h,
                "beta": beta_h,
                "se": se_h,
                "lo": beta_h - z_crit * se_h,
                "hi": beta_h + z_crit * se_h,
                "p_aug": p_aug,
                "first_stage_f": first_stage_f,
                "mop_f": mop_f,
                "mop_cv_10": cv_10,
                "mop_cv_20": cv_20,
            }

            if do_ar:
                ar_lo, ar_hi, ar_kind = _compute_white_ar_ci(
                    dep,
                    x_now,
                    Z_mat,
                    W_ctl,
                    alpha=alpha,
                    beta_hat=beta_h,
                    se_hat=se_h,
                )
                row.update({"ar_lo": ar_lo, "ar_hi": ar_hi, "ar_set_type": ar_kind})

            rows.append(row)

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "la_lp_iv" if has_iv else "la_lp"
    res.ci_level = 1.0 - alpha
    # Lag specification: plain attributes on this object, and ``attrs``
    # (propagated by pandas through copies, slicing and set_index).
    res.n_lags = n_lags
    res.extra_lags = extra_lags
    res.control_lags = control_lags
    res.attrs.update(
        {"n_lags": n_lags, "extra_lags": extra_lags, "p_aug": p_aug,
         "control_lags": control_lags}
    )
    return res


def la_lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str | Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 4,
    extra_lags: int | None = _MOPM_EXTRA_LAGS,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    control_lags: int | None = None,
) -> LPResult:
    """Lag-augmented local projection with instrumental variables.

    The instrument-first spelling of :func:`la_lp` with ``z`` given: the
    exogenous block is the lag-augmented control set of Montiel Olea and
    Plagborg-Møller (2021) - ``n_lags + extra_lags`` lags of ``x`` and ``y``
    (default ``extra_lags=1``, one extra lag at every horizon) and
    ``control_lags`` (default the same) lags of each control - and inference
    is heteroskedasticity-robust throughout (HC0 2SLS standard errors, the
    White-robust Montiel Olea & Pflueger (2013) effective F, optional
    White-robust Anderson-Rubin sets). MOPM prove the OLS case; the IV
    variant is a puremacro extension.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, endogenous regressor, instrument(s), and controls.
    y : str
        Outcome variable name.
    x : str
        Endogenous shock / policy variable name.
    z : str or sequence of str
        Instrument variable name(s).
    horizons : iterable of int, default range(0, 21)
        Impulse response horizons.
    n_lags : int, default 4
        Lags of x and y in the non-augmented LP (see :func:`la_lp`).
    extra_lags : int, default 1
        Augmentation lags, the same at every horizon (MOPM: one). ``None``
        means the default.
    controls : sequence of str, optional
        Additional control variables.
    alpha : float, default 0.10
        Significance level (default 90% CI).
    anderson_rubin : bool, default False
        Compute White-robust Anderson-Rubin confidence sets.
    weak_iv_robust : bool, default False
        Alias for anderson_rubin.
    lags : int, optional
        Alias for n_lags.
    horizon : int, optional
        Sets horizons = range(0, horizon + 1).
    ci : float, optional
        Confidence level, e.g. 0.90.
    control_lags : int, optional
        Lags of each control; default ``n_lags + extra_lags``.

    Returns
    -------
    LPResult
        DataFrame subclass containing impulse responses, White-robust MOP effective F,
        and optional AR confidence sets; the lag specification is recorded as
        in :func:`la_lp`.
    """
    return la_lp(
        df=df,
        y=y,
        x=x,
        horizons=horizons,
        n_lags=n_lags,
        extra_lags=extra_lags,
        controls=controls,
        alpha=alpha,
        z=z,
        anderson_rubin=anderson_rubin,
        weak_iv_robust=weak_iv_robust,
        lags=lags,
        horizon=horizon,
        ci=ci,
        control_lags=control_lags,
    )


__all__ = ["la_lp", "la_lp_iv"]
