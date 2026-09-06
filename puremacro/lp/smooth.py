"""Barnichon-Brownlees (2019) Smooth Local Projections.

Estimates impulse response functions jointly across all horizons h = 0, ..., H
via penalized least squares (PLS) or penalized generalized least squares
(PGLS) on a B-spline basis with a difference (roughness) penalty.

After partialling out the controls z_t (constant, lags of y and x, controls)
horizon by horizon via Frisch-Waugh-Lovell, the estimator solves

    min_θ  Σ_h Σ_{t ∈ S_h} ( ỹ_{h,t} - w̃_{h,t} B_h θ )²  +  λ θ' P θ          (PLS)

where

- S_h = { t : t + h ≤ T } is the sample available at horizon h (each horizon
  uses its own overlapping sample, exactly as horizon-by-horizon local
  projections do; ``gls=True`` uses the common balanced sample instead so that
  the cross-horizon covariance Ω can be estimated),
- ỹ_{h,t} and w̃_{h,t} are the residualized lead y_{t+h} and shock x_t,
- B is the (H+1) x K clamped B-spline basis evaluated at the horizons and
  β(h) = B_h θ is the smoothed impulse response at horizon h,
- P = D_d' D_d is the d-th order difference penalty on the spline coefficients.

Equivalently, with X = B ⊗ w̃ the stacked design, the objective is
||Y - X θ||² + λ θ' P θ (PGLS replaces the squared norm by the Ω^{-1}-weighted
quadratic form).  ``lam`` and ``optimal_lambda`` are the λ of *this* objective,
i.e. on the scale of the stacked sum of squared residuals.  Because the
sum of squares grows with the sample size, the automatic grid is
``logspace(-5, 5, 50) * mean_h(w̃_h' w̃_h)`` so that it always spans from
(essentially) unpenalized to (essentially) polynomial impulse responses.

Provides automated data-driven smoothing parameter selection (AIC, BIC, GCV,
K-fold CV), analytical sandwich HAC standard errors, and moving block bootstrap
inference.

Reference:
    Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth
    Local Projections. The Review of Economics and Statistics, 101(3), 522-530.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm

from ._results import LPResult

_SELECTION_CRITERIA = ("aic", "bic", "gcv", "cv")
_CI_TYPES = {"analytic": "analytic", "hac": "analytic", "bootstrap": "bootstrap", "boot": "bootstrap"}
# Grid of λ / mean_h(w̃_h' w̃_h): from (essentially) unpenalized to (essentially)
# polynomial impulse responses for any sample size.
_LAMBDA_GRID_NORMALISED = np.logspace(-5, 5, 50)


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------
class SmoothLPResult(LPResult):
    """:class:`LPResult` returned by :func:`smooth_lp`.

    Same DataFrame layout as every other ``puremacro.lp`` estimator
    (columns ``h, beta, se, lo, hi, lambda, t`` indexed by ``h``) plus the
    smoothing metadata as attributes (``optimal_lambda``, ``df_lambda``,
    ``theta``, ``vcov``, ``vcov_theta``, ``B``, ``P``, ``lambda_grid``,
    ``selection_criterion``, ``ci_type``, ``n_knots``, ``degree``,
    ``penalty_order``, ``gls``, ``n_obs``).  ``summary()`` reports the
    selected λ and the effective degrees of freedom.
    """

    _metadata = LPResult._metadata + [
        "vcov_theta", "n_obs", "hac_lags", "alpha", "sample",
        "selection_scores", "n_basis",
    ]

    @property
    def _constructor(self):
        return SmoothLPResult

    @property
    def metadata(self) -> dict[str, Any]:
        """Estimation metadata dictionary (including the smoothing settings)."""
        out = super().metadata
        for key in (
            "optimal_lambda", "df_lambda", "selection_criterion", "ci_type",
            "n_knots", "degree", "penalty_order", "gls", "sample",
        ):
            out[key] = getattr(self, key, None)
        return out

    def summary(self) -> str:
        """Text summary: the LP table plus the smoothing diagnostics."""
        base = super().summary()
        lam = getattr(self, "optimal_lambda", None)
        if lam is None:
            return base
        crit = getattr(self, "selection_criterion", None)
        if crit in (None, "fixed"):
            how = "fixed by the user"
        else:
            how = f"selected by {str(crit).upper()}"
        lines = [base, "", "Smoothing (Barnichon-Brownlees 2019):",
                 f"  lambda = {float(lam):.4g}  [{how}]"]
        df_lam = getattr(self, "df_lambda", None)
        if df_lam is not None:
            lines.append(f"  effective degrees of freedom = {float(df_lam):.2f}")
        n_knots = getattr(self, "n_knots", None)
        degree = getattr(self, "degree", None)
        order = getattr(self, "penalty_order", None)
        n_basis = getattr(self, "n_basis", None)
        if n_knots is not None and degree is not None:
            lines.append(
                f"  basis: {n_knots} interior knots, degree {degree}"
                + (f", {n_basis} basis functions" if n_basis is not None else "")
                + (f", penalty order {order}" if order is not None else "")
            )
        ci_type = getattr(self, "ci_type", None)
        alpha = getattr(self, "alpha", None)
        if ci_type is not None:
            if ci_type == "analytic":
                L = getattr(self, "hac_lags", None)
                desc = "analytic sandwich HAC" + (f" (Bartlett, L={L})" if L is not None else "")
            else:
                desc = "moving block bootstrap"
            if alpha is not None:
                desc += f", {100 * (1 - float(alpha)):.0f}% bands"
            lines.append(f"  inference: {desc}")
        n_obs = getattr(self, "n_obs", None)
        sample = getattr(self, "sample", None)
        if n_obs is not None:
            n_arr = np.asarray(n_obs)
            if n_arr.size:
                hs = np.asarray(self["h"]) if "h" in self.columns else np.arange(n_arr.size)
                if sample == "balanced" or n_arr.min() == n_arr.max():
                    lines.append(f"  sample: balanced, T = {int(n_arr[0])} at every horizon")
                else:
                    lines.append(
                        f"  sample: per-horizon, T_h = {int(n_arr[0])} at h={int(hs[0])} "
                        f"to {int(n_arr[-1])} at h={int(hs[-1])}"
                    )
        gls = getattr(self, "gls", None)
        if gls is not None:
            lines.append("  weighting: " + ("PGLS (cross-horizon Omega^-1)" if gls else "PLS (identity)"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Basis and penalty
# ---------------------------------------------------------------------------
def _build_bspline_basis(
    horizons: np.ndarray,
    n_knots: int | None = None,
    degree: int = 3,
) -> tuple[np.ndarray, int]:
    """Construct a clamped B-spline basis matrix over the horizon grid.

    Parameters
    ----------
    horizons : np.ndarray
        1D array of evaluation horizons (e.g. 0, 1, ..., H); at least two.
    n_knots : int or None
        Number of *interior* knots (equally spaced strictly between the first
        and the last horizon).  The basis then has ``n_knots + degree + 1``
        functions.  If None, chosen adaptively (about one knot per three
        horizons).  The number of basis functions is capped at the number of
        horizons so that λ → 0 recovers the unpenalized local projection; an
        explicit ``n_knots`` that violates the cap is reduced with a warning.
    degree : int
        Spline polynomial degree (default 3 for cubic B-splines).  Reduced to
        ``len(horizons) - 1`` on very short horizon grids.

    Returns
    -------
    B : np.ndarray
        (H_num, n_basis) basis matrix evaluated at each horizon.
    n_basis : int
        Number of spline basis functions (= effective interior knots + degree + 1).
    """
    horizons = np.asarray(horizons, dtype=float).ravel()
    H_num = len(horizons)
    if H_num < 2:
        raise ValueError(
            f"smooth_lp needs at least two horizons to build a spline basis (got {H_num}); "
            "use horizons=1 or larger, or a longer iterable of horizons."
        )
    h_min = float(np.min(horizons))
    h_max = float(np.max(horizons))

    if int(degree) < 1:
        raise ValueError(f"degree must be a positive integer, got {degree!r}.")
    # Guard degree against very short horizon grids
    deg = min(int(degree), H_num - 1)
    max_interior = max(0, H_num - deg - 1)

    if n_knots is None:
        n_interior = min(max_interior, max(0, max(4, int(np.ceil(H_num / 3))) - 2))
    else:
        n_interior = int(n_knots)
        if n_interior < 0:
            raise ValueError(f"n_knots must be a non-negative integer (interior knots), got {n_knots!r}.")
        if n_interior > max_interior:
            warnings.warn(
                f"n_knots={n_interior} interior knots with degree {deg} gives "
                f"{n_interior + deg + 1} basis functions, more than the {H_num} horizons; "
                f"reduced to n_knots={max_interior} ({max_interior + deg + 1} basis functions, "
                "a saturated basis). Use res.n_knots to read the effective value.",
                UserWarning,
                stacklevel=3,
            )
            n_interior = max_interior

    knots_inner = np.linspace(h_min, h_max, n_interior + 2)[1:-1]
    t_knots = np.r_[[h_min] * (deg + 1), knots_inner, [h_max] * (deg + 1)]
    n_basis = len(t_knots) - deg - 1  # = n_interior + deg + 1

    B = np.zeros((H_num, n_basis), dtype=float)
    for j in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[j] = 1.0
        B[:, j] = BSpline(t_knots, c, deg, extrapolate=True)(horizons)

    # Clean up numerical fuzz and enforce exact partition of unity
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = B.sum(axis=1, keepdims=True)
    mask = row_sums.ravel() > 1e-12
    if np.any(mask):
        B[mask, :] /= row_sums[mask]

    return B, n_basis


def _difference_penalty_matrix(n_basis: int, order: int = 2) -> np.ndarray:
    """Construct roughness difference penalty matrix P = D_d' D_d."""
    d = min(max(1, int(order)), max(1, n_basis - 1))
    D = np.diff(np.eye(n_basis, dtype=float), n=d, axis=0)
    return D.T @ D


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------
def _as_1d(arr: Any, what: str) -> tuple[np.ndarray, str | None]:
    """Coerce an array-like to a float 1-D array; return it with its name if any."""
    name = getattr(arr, "name", None)
    try:
        vals = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be numeric array-like, got {type(arr).__name__}: {exc}") from exc
    if vals.ndim != 1:
        vals = vals.squeeze()
        if vals.ndim != 1:
            raise ValueError(f"{what} must be a 1-D array, got shape {np.shape(arr)}.")
    return vals, (str(name) if name is not None else None)


def _resolve_controls(
    work: pd.DataFrame,
    controls: Any,
    *,
    names_allowed: bool,
) -> list[str]:
    """Return the list of control column names in ``work``, adding array controls as columns."""
    if controls is None:
        return []
    n = len(work)
    if isinstance(controls, str):
        controls = [controls]
    if isinstance(controls, pd.DataFrame):
        cols: list[str] = []
        for j, col in enumerate(controls.columns):
            vals, _ = _as_1d(controls[col], f"controls column {col!r}")
            if len(vals) != n:
                raise ValueError(f"controls column {col!r} has {len(vals)} rows but the data has {n}.")
            cname = f"__ctl_{j}__"
            work[cname] = vals
            cols.append(cname)
        return cols
    if isinstance(controls, pd.Series):
        controls = controls.to_numpy()
    if isinstance(controls, np.ndarray) or not all(isinstance(c, str) for c in list(controls)):
        C = np.asarray(controls, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.ndim != 2:
            raise ValueError(f"controls array must be 1-D or 2-D (T x k), got shape {C.shape}.")
        if C.shape[0] != n:
            raise ValueError(f"controls array has {C.shape[0]} rows but the data has {n}.")
        cols = []
        for j in range(C.shape[1]):
            cname = f"__ctl_{j}__"
            work[cname] = C[:, j]
            cols.append(cname)
        return cols
    names = [str(c) for c in controls]
    if not names_allowed:
        raise ValueError(
            "controls given as column names require df to be a DataFrame; with array inputs "
            "pass controls as a (T,) or (T, k) array."
        )
    missing = [c for c in names if c not in work.columns]
    if missing:
        raise ValueError(f"controls {missing} not found in df columns {list(work.columns)}.")
    return names


def _coerce_inputs(
    df: pd.DataFrame | np.ndarray | Sequence[float],
    y: str | np.ndarray | Sequence[float] | None,
    x: str | np.ndarray | Sequence[float] | None,
    controls: Any,
) -> tuple[pd.DataFrame, str, str, list[str], str, str]:
    """Normalise the (df, y, x, controls) inputs to a working DataFrame.

    Two calling conventions are supported, mirroring :func:`puremacro.lp.lp_hac`:

    * DataFrame: ``smooth_lp(df, y='y', x='shock', controls=[...])``.  ``y``,
      ``x`` may also be arrays of length ``len(df)``; ``controls`` may be
      column names, a DataFrame, or a (T,) / (T, k) array.
    * Arrays: ``smooth_lp(y_arr, x_arr, controls=C)`` (first positional =
      response, second = shock), or ``smooth_lp(y_arr, x=x_arr)``.

    Returns ``(work, y_col, x_col, ctl_cols, y_name, x_name)``.
    """
    if isinstance(df, pd.DataFrame):
        work = df.reset_index(drop=True).copy()
        n = len(work)

        def _resolve(arg: Any, default: str, role: str) -> tuple[str, str]:
            if arg is None:
                raise ValueError(
                    f"{role} is required when df is a DataFrame: pass a column name "
                    f"(e.g. {default}='{default}') or a 1-D array of length {n}."
                )
            if isinstance(arg, str):
                if arg not in work.columns:
                    raise ValueError(f"{role} column {arg!r} not found in df columns {list(work.columns)}.")
                return arg, arg
            vals, nm = _as_1d(arg, role)
            if len(vals) != n:
                raise ValueError(f"{role} array has {len(vals)} rows but df has {n}.")
            col = f"__{default}_arr__"
            work[col] = vals
            return col, (nm if nm is not None else default)

        y_col, y_name = _resolve(y, "y", "y (response)")
        x_col, x_name = _resolve(x, "x", "x (shock)")
        ctl_cols = _resolve_controls(work, controls, names_allowed=True)
        return work, y_col, x_col, ctl_cols, y_name, x_name

    # Array convention (same as lp_hac): first positional = response, second = shock.
    if isinstance(y, str) or isinstance(x, str):
        raise ValueError(
            "Column names for y/x require df to be a DataFrame. With array inputs call "
            "smooth_lp(y_array, x_array, ...) (response first, shock second)."
        )
    if y is not None and x is not None:
        raise ValueError(
            "Ambiguous array inputs: pass the shock either as the second positional argument, "
            "smooth_lp(y_array, x_array, ...), or as x=..., but not both."
        )
    shock = y if y is not None else x
    if shock is None:
        raise ValueError(
            "Shock array missing: call smooth_lp(y_array, x_array, ...) with the response "
            "first and the shock second (or use a DataFrame with y= and x= column names)."
        )
    y_vals, y_nm = _as_1d(df, "y (response, first positional argument)")
    x_vals, x_nm = _as_1d(shock, "x (shock)")
    if len(y_vals) != len(x_vals):
        raise ValueError(f"Length mismatch: y has {len(y_vals)} rows, x has {len(x_vals)} rows.")
    work = pd.DataFrame({"y": y_vals, "x": x_vals})
    ctl_cols = _resolve_controls(work, controls, names_allowed=False)
    return work, "y", "x", ctl_cols, (y_nm or "y"), (x_nm or "x")


# ---------------------------------------------------------------------------
# Data preparation (FWL, per-horizon samples)
# ---------------------------------------------------------------------------
class _LPData(NamedTuple):
    """Residualized data for the stacked smooth LP system.

    All (T0, H_num) arrays are aligned on the union of the per-horizon samples
    (rows ordered by time) and are zero outside the sample of the given
    horizon, so that every sum over t automatically runs over S_h only.
    """

    w_tilde: np.ndarray  # (T0, H_num) residualized shock, per horizon
    y_tilde: np.ndarray  # (T0, H_num) residualized lead of y, per horizon
    s_ww: np.ndarray     # (H_num,) w̃_h' w̃_h
    b_ols: np.ndarray    # (H_num,) unpenalized horizon-by-horizon LP estimates
    u_ols: np.ndarray    # (T0, H_num) OLS residuals, zero outside S_h
    t_eff: np.ndarray    # (H_num,) per-horizon sample sizes |S_h|


def _prepare_lp_data(
    df: pd.DataFrame | np.ndarray | Sequence[float],
    y: str | np.ndarray | Sequence[float] | None,
    x: str | np.ndarray | Sequence[float] | None,
    horizons: Sequence[int],
    n_lags: int,
    controls: Any,
    *,
    balanced: bool = False,
) -> _LPData:
    """Prepare aligned time-series arrays and project out controls via FWL.

    Each horizon h uses its own sample S_h = {t : y_{t+h} observed} (as in
    horizon-by-horizon local projections); with ``balanced=True`` every
    horizon uses the common sample ∩_h S_h (needed for the GLS weighting).
    """
    work, y_col, x_col, ctl_cols, _, _ = _coerce_inputs(df, y, x, controls)
    horizons = [int(h) for h in horizons]
    H_num = len(horizons)
    n = len(work)
    n_lags = int(n_lags)

    try:
        yv = work[y_col].to_numpy(dtype=float)
        xv = work[x_col].to_numpy(dtype=float)
        C = work[ctl_cols].to_numpy(dtype=float) if ctl_cols else np.empty((n, 0), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"y, x and controls must be numeric columns (y={y_col!r}, x={x_col!r}, "
            f"controls={ctl_cols}): {exc}"
        ) from exc

    def _lag(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if k < n:
            out[k:] = a[: n - k]
        return out

    def _lead(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if k < n:
            out[: n - k] = a[k:]
        return out

    regressors = [np.ones(n, dtype=float)]
    for lag in range(1, n_lags + 1):
        regressors.append(_lag(xv, lag))
        regressors.append(_lag(yv, lag))
        for j in range(C.shape[1]):
            regressors.append(_lag(C[:, j], lag))
    for j in range(C.shape[1]):
        regressors.append(C[:, j])
    Z = np.column_stack(regressors)
    n_z = Z.shape[1]

    Y_lead = np.column_stack([_lead(yv, h) for h in horizons])

    reg_ok = np.isfinite(xv) & np.all(np.isfinite(Z), axis=1)
    lead_ok = np.isfinite(Y_lead)
    if balanced:
        common = reg_ok & np.all(lead_ok, axis=1)
        ok = np.repeat(common[:, None], H_num, axis=1)
    else:
        ok = lead_ok & reg_ok[:, None]

    rows = np.flatnonzero(np.any(ok, axis=1))
    mask = ok[rows, :]
    t_eff = mask.sum(axis=0).astype(int)
    T0 = len(rows)

    min_needed = max(10, n_z + 2)
    if T0 == 0 or int(t_eff.min()) < min_needed:
        h_bad = horizons[int(np.argmin(t_eff))] if T0 else horizons[-1]
        raise ValueError(
            f"Insufficient effective observations ({int(t_eff.min()) if T0 else 0} at horizon "
            f"{h_bad}; at least {min_needed} needed) for smooth LP estimation. "
            "Reduce horizons or n_lags."
        )

    Z_r = Z[rows]
    x_r = xv[rows]
    Y_r = Y_lead[rows]

    W = np.zeros((T0, H_num), dtype=float)
    Yt = np.zeros((T0, H_num), dtype=float)
    U = np.zeros((T0, H_num), dtype=float)
    s = np.zeros(H_num, dtype=float)
    b = np.zeros(H_num, dtype=float)

    # Frisch-Waugh-Lovell projection on each horizon's own sample. With a
    # balanced sample the projection of the shock is identical across
    # horizons, so compute it once.
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(H_num):
        idx = np.flatnonzero(mask[:, k])
        key = idx.tobytes()
        if key in cache:
            Zh, w_t = cache[key]
        else:
            Zh = Z_r[idx]
            wh = x_r[idx]
            coef_w = np.linalg.lstsq(Zh, wh, rcond=1e-12)[0]
            w_t = wh - Zh @ coef_w
            cache[key] = (Zh, w_t)
        yh = Y_r[idx, k]
        coef_y = np.linalg.lstsq(Zh, yh, rcond=1e-12)[0]
        y_t = yh - Zh @ coef_y
        s_k = float(w_t @ w_t)
        if s_k < 1e-12:
            raise ValueError("Shock variable x has near-zero variance after partialling out controls.")
        b_k = float(w_t @ y_t) / s_k
        W[idx, k] = w_t
        Yt[idx, k] = y_t
        U[idx, k] = y_t - w_t * b_k
        s[k] = s_k
        b[k] = b_k

    return _LPData(W, Yt, s, b, U, t_eff)


# ---------------------------------------------------------------------------
# Normal equations, selection, inference
# ---------------------------------------------------------------------------
def _normal_equations(
    W: np.ndarray,
    Y: np.ndarray,
    Omega_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-horizon weight matrix A and moment vector c of the stacked system.

    The penalized normal equations are (B' A B + λ P) θ = B' c with
    A = Ω^{-1} ∘ (W' W) and c = (Ω^{-1} ∘ (W' Y)) 1 (∘ = Hadamard product); for
    PLS (Ω = I) this is A = diag(w̃_h' w̃_h) and c_h = w̃_h' ỹ_h.
    """
    G = W.T @ W
    WY = W.T @ Y
    if Omega_inv is None:
        return np.diag(np.diag(G)), np.diag(WY).copy()
    return Omega_inv * G, (Omega_inv * WY).sum(axis=1)


def _solve_penalized(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, rhs, rcond=1e-12)[0]


def _inverse(M: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(M, np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def _select_lambda(
    B: np.ndarray,
    P: np.ndarray,
    BtAB: np.ndarray,
    Btc: np.ndarray,
    data: _LPData,
    grid: np.ndarray,
    selection: str = "aic",
    Omega_inv: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Automated data-driven lambda selection via information criteria or K-fold CV.

    Returns ``(best_lambda, df_lambda, scores)``.  The CV branch re-estimates
    the *same* penalized (G)LS estimator on each training fold.
    """
    if selection not in _SELECTION_CRITERIA:
        raise ValueError(
            f"Unknown selection criterion '{selection}'. Choose from 'aic', 'bic', 'gcv', 'cv'."
        )
    W, Y, s, b_ols, u_ols, t_eff = data
    n_basis = B.shape[1]
    N = int(t_eff.sum())
    rss_ols_total = float(np.sum(u_ols**2))

    scores = np.full(len(grid), np.inf, dtype=float)

    if selection == "cv":
        T0 = W.shape[0]
        k_folds = min(5, max(2, T0 // 10))
        fold_indices = np.array_split(np.arange(T0), k_folds)
        folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for fold in fold_indices:
            train_idx = np.setdiff1d(np.arange(T0), fold)
            W_tr = W[train_idx]
            if np.any(np.sum(W_tr**2, axis=0) < 1e-12):
                continue
            A_tr, c_tr = _normal_equations(W_tr, Y[train_idx], Omega_inv)
            folds.append((B.T @ A_tr @ B, B.T @ c_tr, W[fold], Y[fold]))
        if not folds:
            raise ValueError("Cross-validation failed: every training fold has a degenerate shock.")
        for i, lam_val in enumerate(grid):
            cv_err = 0.0
            for BtAB_tr, Btc_tr, W_te, Y_te in folds:
                try:
                    theta_tr = np.linalg.solve(BtAB_tr + lam_val * P, Btc_tr)
                except np.linalg.LinAlgError:
                    cv_err = np.inf
                    break
                beta_tr = B @ theta_tr
                pred_err = Y_te - W_te * beta_tr[None, :]
                cv_err += float(np.sum(pred_err**2))
            scores[i] = cv_err
    else:
        for i, lam_val in enumerate(grid):
            try:
                inv_M = np.linalg.solve(BtAB + lam_val * P, np.eye(n_basis))
            except np.linalg.LinAlgError:
                continue
            fit_beta = B @ (inv_M @ Btc)
            # RSS of the stacked system: Σ_h [rss_h + s_h (b_h - β_h)²]
            rss = rss_ols_total + float(np.sum(s * (b_ols - fit_beta) ** 2))
            df_lam = float(np.trace(inv_M @ BtAB))
            if selection == "aic":
                scores[i] = float(np.log(max(1e-12, rss / N)) + 2.0 * df_lam / N)
            elif selection == "bic":
                scores[i] = float(np.log(max(1e-12, rss / N)) + np.log(N) * df_lam / N)
            else:  # gcv
                denom = max(1e-10, (1.0 - df_lam / N) ** 2)
                scores[i] = float((rss / N) / denom)

    if not np.any(np.isfinite(scores)):
        raise ValueError(f"Lambda selection by '{selection}' failed: no finite criterion value on the grid.")
    best_idx = int(np.argmin(scores))
    best_lam = float(grid[best_idx])
    final_df = float(np.trace(_inverse(BtAB + best_lam * P) @ BtAB))
    return best_lam, final_df, scores


def _compute_analytical_hac(
    B: np.ndarray,
    P: np.ndarray,
    BtAB: np.ndarray,
    lam: float,
    beta_smooth: np.ndarray,
    data: _LPData,
    horizons: Sequence[int],
    hac_lags: int | None = None,
    Omega_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Analytical sandwich HAC covariance of the smoothed IRF.

    V_θ = M^{-1} S_HAC M^{-1} with M = B' A B + λ P and score
    s_t = B' D_t Ω^{-1} r_t (D_t = diag(w̃_{·,t}), r_t the stacked residual).
    """
    W, Y, _, _, _, _ = data
    T0 = W.shape[0]

    R = Y - W * beta_smooth[None, :]
    if Omega_inv is None:
        S = (R * W) @ B
    else:
        S = ((R @ Omega_inv) * W) @ B

    # Newey-West Bartlett kernel HAC estimator
    L = hac_lags if hac_lags is not None else max(int(max(horizons)), 1)
    L = int(min(max(0, L), max(1, T0 - 2)))

    S_hac = S.T @ S
    for lag in range(1, L + 1):
        weight = 1.0 - lag / (L + 1.0)
        Gamma_l = S[lag:].T @ S[:-lag]
        S_hac += weight * (Gamma_l + Gamma_l.T)

    inv_M = _inverse(BtAB + lam * P)
    V_theta = inv_M @ S_hac @ inv_M
    V_theta = 0.5 * (V_theta + V_theta.T)
    V_beta = B @ V_theta @ B.T
    se = np.sqrt(np.maximum(0.0, np.diag(V_beta)))
    return se, V_beta, V_theta, L


def _compute_bootstrap_ci(
    B: np.ndarray,
    P: np.ndarray,
    lam: float,
    data: _LPData,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    Omega_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Moving block bootstrap confidence intervals (blocks of length ceil(T^(1/3)))."""
    W, Y, _, _, _, _ = data
    T0, H_num = W.shape
    n_basis = B.shape[1]

    rng = np.random.default_rng(seed)
    block_len = max(2, int(np.ceil(T0 ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(T0 / block_len))

    beta_boot = np.zeros((n_boot, H_num), dtype=float)
    theta_boot = np.zeros((n_boot, n_basis), dtype=float)

    for m in range(n_boot):
        starts = rng.integers(0, max(1, T0 - block_len + 1), size=n_blocks)
        indices = np.concatenate([np.arange(st, st + block_len) for st in starts])[:T0]
        W_b = W[indices]
        Y_b = Y[indices]
        A_b, c_b = _normal_equations(W_b, Y_b, Omega_inv)
        th_b = _solve_penalized(B.T @ A_b @ B + lam * P, B.T @ c_b)
        theta_boot[m, :] = th_b
        beta_boot[m, :] = B @ th_b

    se_boot = np.std(beta_boot, axis=0)
    lo = np.percentile(beta_boot, 100.0 * (alpha / 2.0), axis=0)
    hi = np.percentile(beta_boot, 100.0 * (1.0 - alpha / 2.0), axis=0)
    V_beta = np.atleast_2d(np.cov(beta_boot, rowvar=False))
    V_theta = np.atleast_2d(np.cov(theta_boot, rowvar=False))
    return se_boot, lo, hi, V_beta, V_theta


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------
def _normalise_lambda(lam: float | str | None) -> float | None:
    """Return None for automatic selection, else a validated non-negative float."""
    if lam is None:
        return None
    if isinstance(lam, str):
        if lam.strip().lower() == "auto":
            return None
        raise ValueError(f"lam must be 'auto', None, or a non-negative number; got {lam!r}.")
    if isinstance(lam, bool):
        raise ValueError(f"lam must be 'auto', None, or a non-negative number; got {lam!r}.")
    try:
        val = float(lam)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lam must be 'auto', None, or a non-negative number; got {lam!r}.") from exc
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"lam must be a finite non-negative number; got {lam!r}.")
    return val


def _normalise_horizons(horizons: int | Iterable[int]) -> list[int]:
    if isinstance(horizons, (int, np.integer)):
        if int(horizons) < 1:
            raise ValueError(
                f"horizons={int(horizons)} gives a single horizon; smooth_lp needs at least "
                "two horizons (horizons >= 1) to smooth across."
            )
        return list(range(0, int(horizons) + 1))
    try:
        hs = sorted({int(h) for h in horizons})
    except TypeError as exc:
        raise ValueError(f"horizons must be an int or an iterable of ints, got {horizons!r}.") from exc
    if any(h < 0 for h in hs):
        raise ValueError(f"horizons must be non-negative integers, got {hs}.")
    if len(hs) < 2:
        raise ValueError(
            f"smooth_lp needs at least two distinct horizons to smooth across, got {hs}."
        )
    return hs


def smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    lambda_: float | None = None,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> SmoothLPResult:
    """Estimate smooth local projections per Barnichon & Brownlees (2019).

    Accepts either DataFrame input ``smooth_lp(df, y='y_col', x='x_col', ...)``
    or array input ``smooth_lp(y_array, x_array, ...)`` (response first, shock
    second, exactly as :func:`puremacro.lp.lp_hac`).

    Parameters
    ----------
    df : pd.DataFrame or np.ndarray
        Dataset containing the response, the shock and the controls; or, with
        array input, the 1-D response series itself.
    y : str or np.ndarray
        Column name (or array of length ``len(df)``) of the response; with
        array input, the 1-D shock series.
    x : str or np.ndarray
        Column name (or array of length ``len(df)``) of the shock. With array
        input ``x=`` may be used instead of the second positional argument.
    horizons : int or Iterable[int], default 20
        Max horizon (if int) or explicit iterable of horizons (e.g. range(0, 21)).
        At least two horizons are required.
    n_lags : int, default 4
        Number of autoregressive lags of y, x, and controls.
    controls : Sequence[str], np.ndarray, pd.DataFrame or None, default None
        Additional exogenous controls: column names (DataFrame input), or a
        (T,) / (T, k) array, or a DataFrame with ``len(df)`` rows.
    n_knots : int or None, default None
        Number of *interior* spline knots (equally spaced between the first
        and the last horizon); the basis has ``n_knots + degree + 1``
        functions. If None, about one knot per three horizons. The basis size
        is capped at the number of horizons (so that λ → 0 reproduces the
        unpenalized local projection); an explicit ``n_knots`` above the cap is
        reduced with a warning. The effective value is stored in ``res.n_knots``.
    degree : int, default 3
        B-spline polynomial degree (default 3 for cubic B-spline).
    penalty_order : int, default 2
        Difference order of roughness penalty matrix P = D_d' D_d.
    lam : float, str, or None, default "auto"
        Smoothing parameter λ of the objective
        ``Σ_h Σ_t (ỹ_{h,t} - w̃_{h,t} B_h θ)² + λ θ'Pθ`` (stacked
        sum-of-squares scale). "auto" (case-insensitive) or None selects λ by
        the criterion in ``selection`` over the grid
        ``logspace(-5, 5, 50) * mean_h(w̃_h'w̃_h)`` (``res.lambda_grid``).
        A non-negative float fixes λ.
    selection : str, default "aic"
        Criterion for data-driven λ selection: 'aic', 'bic', 'gcv', or 'cv'
        (K-fold block cross-validation, K = min(5, T // 10)).
    alpha : float, default 0.05
        Significance level for confidence bands (default 0.05 for 95% coverage).
    ci_type : str, default "analytic"
        Confidence interval method: 'analytic' (sandwich HAC) or 'bootstrap'
        (moving block bootstrap).
    lambda_ : float or None, optional
        Backward-compatibility alias for `lam`.
    lags : int or None, optional
        Backward-compatibility alias for `n_lags`.
    horizon : int or None, optional
        Backward-compatibility alias for `horizons`.
    ci : float or None, optional
        Backward-compatibility confidence level (e.g. 0.90 sets alpha = 0.10).
    n_boot : int, default 500
        Number of bootstrap replications when ci_type='bootstrap'.
    seed : int, Generator, or None, optional
        Random seed for bootstrap replication reproducibility.
    hac_lags : int or None, optional
        Bandwidth lag order for Newey-West Bartlett kernel HAC (default: max horizon).
    gls : bool, default False
        If True, applies the cross-horizon GLS weighting matrix Ω^{-1}
        (Ω estimated from the horizon-by-horizon OLS residuals). Estimating Ω
        needs a balanced residual panel, so with ``gls=True`` every horizon
        uses the common sample ``t + H ≤ T``; with ``gls=False`` (default) each
        horizon uses its own sample ``t + h ≤ T`` as in Barnichon-Brownlees.

    Returns
    -------
    SmoothLPResult
        :class:`LPResult` (a :class:`pandas.DataFrame`) with columns
        ``['h', 'beta', 'se', 'lo', 'hi', 'lambda', 't']`` indexed by ``h`` and
        attributes ``optimal_lambda``, ``df_lambda``, ``lambda_grid``,
        ``selection_criterion`` ('fixed' when ``lam`` is a number),
        ``ci_type``, ``theta``, ``vcov``, ``vcov_theta``, ``B``, ``P``,
        ``n_knots``, ``degree``, ``penalty_order``, ``gls``, ``n_obs``.
        Provides ``.summary()`` (which reports λ), ``.plot()``,
        ``.to_markdown()``, ``.to_latex()``, ``.to_typst()``.
    """
    # Parameter normalization & backward compatibility
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, int(horizon) + 1)
    if ci is not None:
        alpha = 1.0 - float(ci)
    if lambda_ is not None:
        lam = lambda_

    if int(n_lags) < 0:
        raise ValueError(f"n_lags must be a non-negative integer, got {n_lags!r}.")
    n_lags = int(n_lags)
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}.")
    alpha = float(alpha)

    selection_clean = str(selection).lower().strip()
    if selection_clean not in _SELECTION_CRITERIA:
        raise ValueError(
            f"Unknown selection criterion '{selection}'. Choose from 'aic', 'bic', 'gcv', 'cv'."
        )
    ci_key = str(ci_type).lower().strip()
    if ci_key not in _CI_TYPES:
        raise ValueError(
            f"Unknown ci_type '{ci_type}'. Choose 'analytic' (sandwich HAC) or 'bootstrap' "
            "(moving block bootstrap)."
        )
    ci_clean = _CI_TYPES[ci_key]
    lam_fixed = _normalise_lambda(lam)
    horizons_list = _normalise_horizons(horizons)
    H_num = len(horizons_list)
    h_arr = np.array(horizons_list, dtype=float)

    # 1. Coerce inputs, prepare data and partial out controls via FWL
    work, y_col, x_col, ctl_cols, y_name, x_name = _coerce_inputs(df, y, x, controls)
    data = _prepare_lp_data(
        work, y_col, x_col, horizons_list, n_lags, ctl_cols, balanced=bool(gls),
    )
    W, Y_tilde, s_ww, b_ols, u_ols, t_eff = data

    # 2. Build B-spline basis and penalty matrix
    B, n_basis = _build_bspline_basis(h_arr, n_knots=n_knots, degree=degree)
    deg_eff = min(int(degree), H_num - 1)
    n_knots_eff = n_basis - deg_eff - 1
    P = _difference_penalty_matrix(n_basis=n_basis, order=penalty_order)

    # Optional cross-horizon GLS weighting (balanced sample)
    Omega_inv: np.ndarray | None = None
    if gls:
        T_common = int(t_eff.min())
        Omega = (u_ols.T @ u_ols) / T_common
        ridge = 1e-4 * float(np.trace(Omega)) / H_num
        Omega_reg = Omega + ridge * np.eye(H_num)
        try:
            Omega_inv = np.linalg.inv(Omega_reg)
        except np.linalg.LinAlgError:
            Omega_inv = np.linalg.pinv(Omega_reg)
        Omega_inv = 0.5 * (Omega_inv + Omega_inv.T)

    A, c = _normal_equations(W, Y_tilde, Omega_inv)
    BtAB = B.T @ A @ B
    Btc = B.T @ c

    # 3. Smoothing parameter: data-driven selection or user-fixed
    grid = _LAMBDA_GRID_NORMALISED * float(np.mean(s_ww))
    selection_scores: np.ndarray | None = None
    if lam_fixed is None:
        lam_val, df_lam, selection_scores = _select_lambda(
            B=B, P=P, BtAB=BtAB, Btc=Btc, data=data, grid=grid,
            selection=selection_clean, Omega_inv=Omega_inv,
        )
        criterion_used = selection_clean
    else:
        lam_val = lam_fixed
        df_lam = float(np.trace(_inverse(BtAB + lam_val * P) @ BtAB))
        criterion_used = "fixed"

    # 4. Final point estimates
    theta = _solve_penalized(BtAB + lam_val * P, Btc)
    beta_smooth = B @ theta

    # 5. Inference: analytical sandwich HAC vs moving block bootstrap
    L_used: int | None = None
    if ci_clean == "bootstrap":
        se, lo, hi, V_beta, V_theta = _compute_bootstrap_ci(
            B=B, P=P, lam=lam_val, data=data, alpha=alpha, n_boot=int(n_boot),
            seed=seed, Omega_inv=Omega_inv,
        )
    else:
        se, V_beta, V_theta, L_used = _compute_analytical_hac(
            B=B, P=P, BtAB=BtAB, lam=lam_val, beta_smooth=beta_smooth, data=data,
            horizons=horizons_list, hac_lags=hac_lags, Omega_inv=Omega_inv,
        )
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        lo = beta_smooth - z_crit * se
        hi = beta_smooth + z_crit * se

    # 6. Format standard LPResult
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, beta_smooth / se, np.nan)

    res = SmoothLPResult({
        "h": horizons_list,
        "beta": beta_smooth,
        "se": se,
        "lo": lo,
        "hi": hi,
        "lambda": [lam_val] * H_num,
        "t": t_stat,
    })
    res.index = res["h"]
    res.y_name = y_name
    res.x_name = x_name
    res.method = "LP-smooth"
    meta: dict[str, Any] = {
        "optimal_lambda": lam_val,
        "df_lambda": df_lam,
        "lambda_grid": grid,
        "selection_scores": selection_scores,
        "selection_criterion": criterion_used,
        "ci_type": ci_clean,
        "theta": theta,
        "vcov": V_beta,
        "vcov_theta": V_theta,
        "B": B,
        "P": P,
        "n_knots": int(n_knots_eff),
        "n_basis": int(n_basis),
        "degree": int(deg_eff),
        "penalty_order": int(min(max(1, int(penalty_order)), max(1, n_basis - 1))),
        "gls": bool(gls),
        "n_obs": t_eff.copy(),
        "hac_lags": L_used,
        "alpha": alpha,
        "sample": "balanced" if gls else "per-horizon",
    }
    for key, val in meta.items():
        object.__setattr__(res, key, val)

    return res


# Backward compatibility alias
lp_smooth = smooth_lp

__all__ = ["smooth_lp", "lp_smooth", "SmoothLPResult"]
