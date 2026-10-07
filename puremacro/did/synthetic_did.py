"""Synthetic Difference-in-Differences (Arkhangelsky-Athey-Hirshberg-
Imbens-Wager 2021).

SDID combines synthetic-control unit weights with DiD time weights.
For a single treated unit (or block of contemporaneously-treated
units) the estimator solves two weighting problems, **each with a free
intercept** (Arkhangelsky et al. 2021, eqs. 2.1 and 2.3):

  (ω̂_0, ω̂) = arg min Σ_{t<T_0} (ω_0 + Σ_i ω_i Y_{i,t} − Ȳ_{tr,t})²
                        + ζ_ω² T_pre ‖ω‖²
       s.t. ω ≥ 0,  Σ ω_i = 1                (units; pre-period match)

  (λ̂_0, λ̂) = arg min Σ_{i∈co} (λ_0 + Σ_{t<T_0} λ_t Y_{i,t} − Ȳ_{i,post})²
                        + ζ_λ² N_co ‖λ‖²
       s.t. λ ≥ 0,  Σ λ_t = 1                (times; pre vs post)

and the SDID estimate is

    τ̂_SDID = (Ȳ_{tr,post} − Σ_i ω̂_i Ȳ_{i,post})
              − Σ_t λ̂_t (Ȳ_{tr,t} − Σ_i ω̂_i Y_{i,t}).

The intercepts are what make SDID invariant to additive unit level
shifts: without ω_0 the unit weights would have to match the treated
*level*, not just its pre-trend, and a constant added to the treated
units' outcome would move both ω̂ and τ̂. Profiling out the intercept is
equivalent to centring the pre-period outcome paths (over time for ω,
over control units for λ) before solving the simplex-constrained ridge
problem; that is how it is implemented.

Regularisation (paper eq. 2.2 and footnote 3; ``synthdid`` R package):
``ζ_ω = (N_tr T_post)^{1/4} σ̂`` and ``ζ_λ = 1e-6 σ̂``, where σ̂ is the
standard deviation of the first differences ``Δ_it = Y_{i,t+1} − Y_{i,t}``
of the control units' pre-period outcomes, demeaned by their overall mean
(n − 1 divisor, as ``synthdid``'s ``noise.level``). σ̂ scales with ``y``, so
the weights do not depend on the units of the outcome and τ̂(c·y) = c·τ̂(y).

Weights solver. Each weight problem is a strictly convex quadratic
programme on the simplex. It is solved by SLSQP after dividing the
(centred) data by their root-mean-square, so the stopping rule does not
depend on the units of ``y``. The answer is then certified by the
optimality conditions: the Frank-Wolfe duality gap ``g'w − min_i g_i``
(``g`` the gradient) bounds the objective's distance to the optimum. An
SLSQP exit flagged as a failure whose iterate passes that test is kept; an
iterate that fails it is polished by accelerated projected gradient, and
if that does not converge either the solver warns and returns the best
feasible point it found. It never falls back to uniform weights silently.

Inference (paper Section 5; CIs are Gaussian, ``τ̂ ± z_{1−α/2} √V̂``, eq. 5.1):

* ``"placebo"`` (Algorithm 4) — reassign the treatment to ``N_tr`` controls
  drawn without replacement, re-estimate SDID on the controls only, and use
  the variance of those placebo estimates. The only option that works with
  one treated unit; valid under homoskedasticity across units. The
  ``"auto"`` choice for one or two treated units.
* ``"bootstrap"`` (Algorithm 2) — resample all N units (treated and
  control) with replacement, redraw samples with no treated or no control
  unit, re-estimate. For designs with several treated units (``"auto"``
  from three on).
* ``"jackknife"`` (Algorithm 3) — leave one unit out at a time with ω̂, λ̂
  held fixed. Undefined with one treated unit.

As in ``synthdid``'s ``vcov()``, every replication reuses the full-sample
ζ_ω, ζ_λ, and the variance is ``(1/B) Σ_b (τ̂_b − τ̄)²``.

This implementation supports the **single-cohort** case (one treatment
time common across treated units). For staggered designs use
:func:`puremacro.did.sdid_multi_cohort`, or the BJS / CS estimators.

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., Wager, S.
    (2021). Synthetic difference-in-differences. AER 111(12), 4088-4118.
    Equation and algorithm numbers are those of arXiv:1812.09970v4.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ._results import SyntheticDiDResult

#: Variance estimators accepted by ``synthetic_did(se_method=...)``.
_SE_METHODS = ("auto", "placebo", "bootstrap", "jackknife")
#: ``se_method="auto"`` uses the placebo estimator up to this many treated
#: units and the bootstrap above it. Monte Carlo (iid two-way FE, 30
#: controls, T = 20, nominal 90 %, 200 draws each): bootstrap coverage
#: 0.790 at N_tr = 2, 0.885 at 3, 0.890 at 4, 0.910 at 5; placebo
#: 0.885-0.905 throughout, including N_tr = 1.
_AUTO_PLACEBO_MAX_TREATED = 2

#: Target of the optimality check: Frank-Wolfe duality gap relative to
#: ``max(objective, _GAP_FLOOR * rows)`` in the rescaled problem.
_GAP_RTOL = 1e-6
#: Gap (same scale) above which the solver warns after polishing.
_GAP_WARN_RTOL = 1e-3
#: Absolute floor of the gap scale, per row of the rescaled problem, used
#: when the optimum fits the data (almost) exactly.
_GAP_FLOOR = 1e-8
#: Maximum accelerated projected-gradient steps when neither SLSQP nor the
#: support refinement passes the optimality check.
_POLISH_MAX_ITER = 5_000


# ---------------------------------------------------------------------------
# Simplex-constrained ridge regression
# ---------------------------------------------------------------------------


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, Σ w = 1}`` (sort-based)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, v.size + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def _refine_on_support(
    A: np.ndarray, b: np.ndarray, ridge: float, w: np.ndarray,
    rel_gap, target: float,
) -> np.ndarray | None:
    """Exact solution of  min ‖A w − b‖² + ridge ‖w‖²  on the simplex,
    started from the support of an approximate solution ``w``.

    On a support S the optimum solves the KKT system
    ``[A_S'A_S + ridge I, 1; 1', 0] [w_S; μ] = [A_S'b; 1]``. A coordinate
    with a negative solution leaves S; an inactive coordinate whose
    half-gradient is below the common value on S enters it (a primal
    active-set step). Returns the first feasible point with
    ``rel_gap(w) <= target``, else the feasible point with the smallest
    gap reached, else ``None``.
    """
    n = w.size
    S = w > 1e-9 * w.max()
    best = None
    for _ in range(2 * n + 2):
        idx = np.flatnonzero(S)
        k = idx.size
        As = A[:, idx]
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = As.T @ As + ridge * np.eye(k)
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        sol = np.linalg.lstsq(K, np.append(As.T @ b, 1.0), rcond=None)[0]
        w_S = sol[:k]
        if not np.all(np.isfinite(w_S)):
            return best
        if w_S.min() < 0.0:
            if k == 1:
                return best
            S[idx[np.argmin(w_S)]] = False
            continue
        w_new = np.zeros(n)
        w_new[idx] = w_S / w_S.sum()
        if rel_gap(w_new) <= target:
            return w_new
        best = w_new if best is None or rel_gap(w_new) < rel_gap(best) else best
        half_g = A.T @ (A @ w_new - b) + ridge * w_new
        out = np.flatnonzero(~S)
        if out.size == 0:
            return best
        j = out[np.argmin(half_g[out])]
        if half_g[j] >= half_g[idx].mean():
            return best
        S[j] = True
    return best


def _solve_simplex_quadratic(
    A: np.ndarray, b: np.ndarray, ridge: float, *, intercept: bool = True,
    what: str = "weights",
) -> np.ndarray:
    """Solve  min_{w, w_0} ‖A w + w_0 − b‖² + ridge ‖w‖²
    s.t.  w ≥ 0,  Σ w = 1.

    With ``intercept=True`` the free intercept ``w_0`` is profiled out
    analytically: for any ``w`` the optimal ``w_0`` is the mean residual,
    so the problem is equivalent to the same simplex ridge regression on
    row-centred ``A`` and ``b``.

    The problem is solved in units of the data's root-mean-square (``A``,
    ``b`` divided by it, ``ridge`` by its square), which leaves the
    minimiser unchanged and makes the result independent of the units of
    ``A`` and ``b``. SLSQP's iterate — whatever its exit flag — is refined
    by an exact KKT solve on its support (:func:`_refine_on_support`) and
    accepted when it satisfies the optimality conditions: Frank-Wolfe gap
    ``g'w − min_i g_i`` (``g`` the gradient; an upper bound on the
    objective's excess over the optimum) below ``_GAP_RTOL`` times
    ``max(objective, _GAP_FLOOR · rows)``. Otherwise it is polished by
    accelerated projected gradient; if the gap is still above
    ``_GAP_WARN_RTOL`` a ``UserWarning`` is raised and the best feasible
    point found is returned. ``what`` names the weights in that warning.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if intercept:
        A = A - A.mean(axis=0, keepdims=True)
        b = b - b.mean()
    m, n = A.shape
    if n == 1:
        return np.ones(1)
    scale = float(np.sqrt((np.sum(A * A) + b @ b) / (A.size + b.size)))
    if np.isfinite(scale) and scale > 0.0:
        A = A / scale
        b = b / scale
        ridge = ridge / scale ** 2

    def obj(w):
        r = A @ w - b
        return float(r @ r + ridge * (w @ w))

    def jac(w):
        return 2.0 * (A.T @ (A @ w - b)) + 2.0 * ridge * w

    floor = _GAP_FLOOR * max(m, 1)

    def rel_gap(w):
        g = jac(w)
        return float(g @ w - g.min()) / max(obj(w), floor)

    w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: np.ones_like(w)}]
    res = minimize(obj, w0, jac=jac, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = w0
    x = np.asarray(res.x, dtype=float)
    if x.shape == w0.shape and np.all(np.isfinite(x)):
        x = np.clip(x, 0.0, None)
        if x.sum() > 0.0:
            x = x / x.sum()
            if obj(x) <= obj(w0):
                w = x

    def refine(w):
        # Exact KKT solve on the support found so far (kept only if better).
        w_ref = _refine_on_support(A, b, ridge, w, rel_gap, _GAP_RTOL)
        return w_ref if w_ref is not None and rel_gap(w_ref) < rel_gap(w) else w

    w = refine(w)
    if rel_gap(w) <= _GAP_RTOL:
        return w

    # Rare fallback: accelerated projected gradient (FISTA with
    # function-value restart), then another support refinement.
    lip = 2.0 * (float(np.linalg.norm(A, 2)) ** 2 + ridge)
    if lip > 0.0:
        x, y, t, fx = w.copy(), w.copy(), 1.0, obj(w)
        for k in range(1, _POLISH_MAX_ITER + 1):
            x_new = _project_simplex(y - jac(y) / lip)
            f_new = obj(x_new)
            if f_new > fx:
                y, t = x.copy(), 1.0
                continue
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, fx, t = x_new, f_new, t_new
            if k % 25 == 0 and rel_gap(x) <= _GAP_RTOL:
                break
        if fx <= obj(w):
            w = refine(x)
    gap = rel_gap(w)
    if gap > _GAP_WARN_RTOL:
        warnings.warn(
            f"synthetic_did: the {what} solver did not reach the optimum "
            f"(SLSQP: {res.message!s}; relative optimality gap {gap:.1e} after "
            "projected-gradient polishing). The weights returned are the best "
            "feasible point found; check the panel for near-duplicate or "
            "constant series.",
            UserWarning, stacklevel=3,
        )
    return w


# ---------------------------------------------------------------------------
# SDID pieces
# ---------------------------------------------------------------------------


def _noise_level(Y_donors_pre: np.ndarray) -> float:
    """σ̂ of eq. (2.2): sd of all first differences of the controls'
    pre-period outcomes around their overall mean (n − 1 divisor, as
    ``synthdid``'s ``noise.level``; the paper writes n). 0 if undefined."""
    diffs = np.diff(np.asarray(Y_donors_pre, dtype=float), axis=1)
    return float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0


def _sdid_zeta(
    Y_donors_pre: np.ndarray, *, n_treated: int, T_post: int,
    noise_level: float | None = None,
) -> tuple[float, float]:
    """``(ζ_ω, ζ_λ) = ((N_tr T_post)^{1/4} σ̂, 1e-6 σ̂)`` (eq. 2.2, fn. 3)."""
    sigma = _noise_level(Y_donors_pre) if noise_level is None else float(noise_level)
    return (n_treated * T_post) ** 0.25 * sigma, 1e-6 * sigma


def _sdid_weights(
    Y_donors_pre: np.ndarray,
    Y_donors_post: np.ndarray,
    Y_treated_pre: np.ndarray,
    *,
    n_treated: int = 1,
    zeta: tuple[float, float] | None = None,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit weights ω (over donors, eq. 2.1) and time weights λ (over pre
    periods, eq. 2.3). ``zeta = (ζ_ω, ζ_λ)`` overrides the regularisation
    computed from ``Y_donors_pre``; replications pass the full-sample one."""
    J, T_pre = Y_donors_pre.shape
    if zeta is None:
        zeta = _sdid_zeta(Y_donors_pre, n_treated=n_treated,
                          T_post=Y_donors_post.shape[1], noise_level=noise_level)
    zeta_omega, zeta_lambda = zeta
    omega = _solve_simplex_quadratic(Y_donors_pre.T, Y_treated_pre,
                                     ridge=zeta_omega ** 2 * T_pre,
                                     what="unit-weight (omega)")
    lam = _solve_simplex_quadratic(Y_donors_pre, Y_donors_post.mean(axis=1),
                                   ridge=zeta_lambda ** 2 * J,
                                   what="time-weight (lambda)")
    return omega, lam


def _sdid_tau(
    omega: np.ndarray, lam: np.ndarray,
    Y_donors_pre: np.ndarray, Y_donors_post: np.ndarray,
    Y_treated_pre: np.ndarray, Y_treated_post: np.ndarray,
) -> float:
    delta_post = float(Y_treated_post.mean() - omega @ Y_donors_post.mean(axis=1))
    delta_pre = float((lam * (Y_treated_pre - omega @ Y_donors_pre)).sum())
    return delta_post - delta_pre


def _fit_block(Yc_pre, Yc_post, Yt_pre, Yt_post, zeta) -> float:
    """τ̂ for controls ``Yc_*`` (rows) and treated units ``Yt_*`` (rows)."""
    yt_pre = Yt_pre.mean(axis=0)
    omega, lam = _sdid_weights(Yc_pre, Yc_post, yt_pre, zeta=zeta)
    return _sdid_tau(omega, lam, Yc_pre, Yc_post, yt_pre, Yt_post.mean(axis=0))


def _placebo_taus(Yc_pre, Yc_post, n_treated, n_reps, rng, zeta) -> np.ndarray:
    """Algorithm 4: SDID on the controls with ``n_treated`` of them, drawn
    without replacement, playing the treated block."""
    J = Yc_pre.shape[0]
    taus = np.empty(n_reps)
    for r in range(n_reps):
        perm = rng.permutation(J)
        co, tr = perm[:J - n_treated], perm[J - n_treated:]
        taus[r] = _fit_block(Yc_pre[co], Yc_post[co], Yc_pre[tr], Yc_post[tr], zeta)
    return taus


def _bootstrap_taus(Yc_pre, Yc_post, Yt_pre, Yt_post, n_reps, rng, zeta) -> np.ndarray:
    """Algorithm 2: resample all N units with replacement; draws with no
    treated or no control unit are discarded and redrawn."""
    J, K = Yc_pre.shape[0], Yt_pre.shape[0]
    N = J + K
    taus = np.empty(n_reps)
    for r in range(n_reps):
        for _ in range(10_000):
            idx = rng.integers(0, N, size=N)
            co = np.sort(idx[idx < J])
            tr = np.sort(idx[idx >= J]) - J
            if co.size and tr.size:
                break
        else:  # pragma: no cover - probability < 1e-100 for any valid panel
            raise RuntimeError("bootstrap could not draw a sample with both "
                               "treated and control units")
        taus[r] = _fit_block(Yc_pre[co], Yc_post[co], Yt_pre[tr], Yt_post[tr], zeta)
    return taus


def _jackknife_taus(omega, lam, Yc_pre, Yc_post, Yt_pre, Yt_post) -> np.ndarray | None:
    """Algorithm 3: τ̂^{(−i)} for every unit with ω̂ (renormalised) and λ̂
    fixed. ``None`` when undefined: one treated unit, or a control that
    carries all of ω̂."""
    J, K = Yc_pre.shape[0], Yt_pre.shape[0]
    if K < 2:
        return None
    yt_pre, yt_post = Yt_pre.mean(axis=0), Yt_post.mean(axis=0)
    out = []
    keep = np.ones(J, dtype=bool)
    for i in range(J):
        keep[:] = True
        keep[i] = False
        w = omega[keep]
        if w.sum() <= 1e-8:
            return None
        out.append(_sdid_tau(w / w.sum(), lam, Yc_pre[keep], Yc_post[keep],
                             yt_pre, yt_post))
    keep = np.ones(K, dtype=bool)
    for k in range(K):
        keep[:] = True
        keep[k] = False
        out.append(_sdid_tau(omega, lam, Yc_pre, Yc_post,
                             Yt_pre[keep].mean(axis=0), Yt_post[keep].mean(axis=0)))
    return np.asarray(out)


_RESULT_FIELDS = {f.name for f in dataclasses.fields(SyntheticDiDResult)}


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------


def synthetic_did(
    df: pd.DataFrame,
    *,
    unit: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat_time: str = "treat_time",
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
    ci: float | None = None,
    se_method: str = "auto",
    noise_level: float | None = None,
) -> SyntheticDiDResult:
    """Synthetic-DiD for a single common treatment time.

    Identifies the treatment time as the (single) non-NaN value of
    ``treat_time`` across treated units; raises if multiple cohorts
    coexist (use :func:`sdid_multi_cohort`, BJS or CS for staggered
    designs). The panel must be **balanced** over the treated and donor
    units: a missing ``(unit, time)`` cell raises a ``ValueError``
    naming the cell.

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period (NaN for never-treated controls).
    n_boot : int, default 200
        Replications ``B`` of the placebo (Algorithm 4) or bootstrap
        (Algorithm 2) variance estimator; ignored by the jackknife. ``0``
        skips inference for every ``se_method``: ``se``, ``lo`` and ``hi``
        are NaN, the result's ``se_method`` is ``None`` and ``n_reps`` is
        0, and the bootstrap's few-treated-units warnings are not issued.
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % CIs).
    seed : int, default 0
        RNG seed for the placebo draws / bootstrap resamples.
    ci : float, optional
        Confidence-interval coverage; when given, ``alpha = 1 − ci``
        (same convention as :func:`callaway_santanna`).
    se_method : {"auto", "placebo", "bootstrap", "jackknife"}, default "auto"
        Variance estimator of Arkhangelsky et al. (2021, Section 5).
        ``"auto"`` uses ``"placebo"`` with one or two treated units (when
        there are more controls than treated units) and ``"bootstrap"``
        otherwise. In puremacro's Monte Carlo (iid two-way fixed effects, 30
        controls, nominal 90 %) the bootstrap covered 0.79 with two treated
        units and 0.885-0.91 with three to eight; the placebo covered
        0.885-0.905 from one to five.

        - ``"placebo"`` (Algorithm 4): SDID re-estimated on the controls
          only, with ``N_tr`` controls drawn without replacement as placebo
          treated units. Needs more controls than treated units; assumes
          homoskedasticity across units.
        - ``"bootstrap"`` (Algorithm 2): all units resampled with
          replacement, draws without a treated or a control unit redrawn.
          With one treated unit it is not well-defined (the paper's p. 28
          and Table 4 note) — it can only resample the controls, misses the
          treated unit's own noise and under-covers — so it warns; it also
          warns with two treated units, where it under-covers.
        - ``"jackknife"`` (Algorithm 3): leave-one-unit-out with ω̂, λ̂
          fixed; NaN (with a warning) for one treated unit or when a single
          control carries all of ω̂.
    noise_level : float, optional
        σ̂ in the outcome's units, overriding eq. (2.2). ``ζ_ω`` and ``ζ_λ``
        are proportional to it. Fixing it makes τ̂ exactly invariant to any
        common time shift of the outcome (the default σ̂ is invariant to
        common *linear* trends only).

    Returns
    -------
    SyntheticDiDResult
        Frozen dataclass with ``tau`` (point estimate), ``omega``
        (donor-unit weights), ``lambda_w`` (pre-period time weights;
        renamed from ``lambda`` because ``lambda`` is a Python reserved
        keyword), ``se`` (square root of the chosen variance estimate, in
        the outcome's units), ``lo``/``hi`` (``tau ∓ z_{1−α/2}·se``, eq.
        5.1), ``treatment_time``, plus the treated-mean and ω-weighted
        synthetic outcome paths (``y_treated``, ``y_synthetic``) used by
        ``.plot()``.

    Notes
    -----
    Both weight problems include the intercepts of Arkhangelsky et al.
    (2021), so the estimate is invariant to adding a constant to any
    unit's outcome path. It is scale-equivariant: τ̂(c·y) = c·τ̂(y), and
    so are ``se``, ``lo`` and ``hi`` for a given ``seed``.

    Up to puremacro 4.3.0 ``se``/``lo``/``hi`` came from a bootstrap
    that resampled the donors only, holding the treated units fixed, with
    a percentile interval; with one treated unit its nominal 90 % interval
    covered the truth about 50-67 % of the time. The noise level was also
    demeaned period by period rather than as in eq. (2.2), and a weight
    solve that SLSQP flagged as failed fell back silently to uniform
    weights (plain DiD), which happened for outcomes in large units.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118 (arXiv:1812.09970v4: eqs. 2.1-2.3, 5.1;
        Algorithms 2-4).
    """
    if ci is not None:
        alpha = 1.0 - ci
    if se_method not in _SE_METHODS:
        raise ValueError(f"se_method must be one of {_SE_METHODS}; got {se_method!r}")
    if noise_level is not None and not (np.isfinite(noise_level) and noise_level >= 0):
        raise ValueError(f"noise_level must be a finite number >= 0; got {noise_level!r}")
    n_boot = int(n_boot)
    if n_boot < 0:
        raise ValueError(f"n_boot must be >= 0; got {n_boot}")
    rng = np.random.default_rng(seed)
    df = df.sort_values([unit, time]).reset_index(drop=True)

    # Identify treated and control units; require a single treatment time.
    cohort_of = df.groupby(unit)[treat_time].first()
    treated_cohorts = cohort_of.dropna().unique()
    if len(treated_cohorts) != 1:
        raise ValueError(
            f"synthetic_did expects a single common treatment time; "
            f"got cohorts {treated_cohorts}. For staggered designs, "
            "use sdid_multi_cohort, or iterate per cohort / use BJS / CS."
        )
    T_treat = float(treated_cohorts[0])
    treated_units = cohort_of.index[~cohort_of.isna()].values
    donor_units = cohort_of.index[cohort_of.isna()].values
    if len(donor_units) < 2:
        raise ValueError("need at least 2 never-treated donor units")

    times_all = np.sort(df[time].unique())
    pre_mask = times_all < T_treat
    post_mask = times_all >= T_treat
    pre_times = times_all[pre_mask]
    post_times = times_all[post_mask]
    if len(pre_times) < 2 or len(post_times) < 1:
        raise ValueError("need ≥ 2 pre-treatment and ≥ 1 post-treatment periods")

    if df.duplicated(subset=[unit, time]).any():
        dup = df[df.duplicated(subset=[unit, time])].iloc[0]
        raise ValueError(
            f"duplicate ({unit}, {time}) cell: ({dup[unit]!r}, {dup[time]!r}); "
            "synthetic_did needs one row per unit and period"
        )
    Y_wide = df.pivot(index=unit, columns=time, values=outcome)
    missing = Y_wide.isna()
    if missing.to_numpy().any():
        cells = [(u, t) for u, t in zip(*np.nonzero(missing.to_numpy()))]

        def _py(v):
            return v.item() if hasattr(v, "item") else v

        named = [f"({_py(Y_wide.index[u])!r}, {_py(Y_wide.columns[t])!r})"
                 for u, t in cells[:5]]
        more = "" if len(cells) <= 5 else f" and {len(cells) - 5} more"
        raise ValueError(
            f"synthetic_did requires a balanced panel: {len(cells)} missing "
            f"({unit}, {time}) cell(s) {', '.join(named)}{more}. Drop the "
            "affected units or fill the outcome before calling."
        )
    if not np.isfinite(Y_wide.to_numpy(dtype=float)).all():
        raise ValueError(f"synthetic_did: {outcome!r} has non-finite values")
    Y_donors_pre = Y_wide.loc[donor_units, pre_times].to_numpy(dtype=float)     # (J, T_pre)
    Y_donors_post = Y_wide.loc[donor_units, post_times].to_numpy(dtype=float)   # (J, T_post)
    Yt_pre_units = Y_wide.loc[treated_units, pre_times].to_numpy(dtype=float)   # (N_tr, T_pre)
    Yt_post_units = Y_wide.loc[treated_units, post_times].to_numpy(dtype=float)  # (N_tr, T_post)
    Y_treated_pre = Yt_pre_units.mean(axis=0)
    Y_treated_post = Yt_post_units.mean(axis=0)
    n_tr, n_co = len(treated_units), len(donor_units)

    zeta = _sdid_zeta(Y_donors_pre, n_treated=n_tr, T_post=len(post_times),
                      noise_level=noise_level)
    omega, lam = _sdid_weights(Y_donors_pre, Y_donors_post, Y_treated_pre,
                               n_treated=n_tr, zeta=zeta)
    tau = _sdid_tau(omega, lam, Y_donors_pre, Y_donors_post,
                    Y_treated_pre, Y_treated_post)

    # ---------------- Inference (Section 5) ----------------
    method = se_method
    if method == "auto":
        method = ("placebo" if n_tr <= _AUTO_PLACEBO_MAX_TREATED and n_co > n_tr
                  else "bootstrap")
    if method == "placebo" and n_co <= n_tr:
        raise ValueError(
            f"se_method='placebo' needs more control units than treated units "
            f"(got {n_co} controls, {n_tr} treated); use se_method='bootstrap'"
        )
    # The bootstrap caveats describe an se; with n_boot=0 none is computed.
    if n_boot > 0 and method == "bootstrap" and n_tr == 1:
        warnings.warn(
            "synthetic_did: the bootstrap (Arkhangelsky et al. 2021, Algorithm 2) "
            "is not well-defined with one treated unit: it can only resample the "
            "controls, so its se omits the treated unit's own noise and the "
            "interval under-covers. Use se_method='placebo' (Algorithm 4), the "
            "default for one treated unit.",
            UserWarning, stacklevel=2,
        )
    elif n_boot > 0 and method == "bootstrap" and n_tr <= _AUTO_PLACEBO_MAX_TREATED:
        warnings.warn(
            f"synthetic_did: the bootstrap is unreliable with only {n_tr} treated "
            "units (Arkhangelsky et al. 2021, p. 28; a nominal 90% interval "
            "covered about 79% in puremacro's Monte Carlo with two treated "
            "units). Consider se_method='placebo' (Algorithm 4), which "
            "se_method='auto' uses for one or two treated units whenever there "
            "are more controls than treated units.",
            UserWarning, stacklevel=2,
        )
    se = lo = hi = float("nan")
    reps: np.ndarray | None = None
    if n_boot > 0:
        if method == "placebo":
            reps = _placebo_taus(Y_donors_pre, Y_donors_post, n_tr, n_boot, rng, zeta)
            se = float(np.std(reps))
        elif method == "bootstrap":
            reps = _bootstrap_taus(Y_donors_pre, Y_donors_post, Yt_pre_units,
                                   Yt_post_units, n_boot, rng, zeta)
            se = float(np.std(reps))
        else:
            reps = _jackknife_taus(omega, lam, Y_donors_pre, Y_donors_post,
                                   Yt_pre_units, Yt_post_units)
            if reps is None:
                warnings.warn(
                    "synthetic_did: the jackknife (Algorithm 3) is not defined "
                    + ("with one treated unit" if n_tr == 1 else
                       "when one control carries all of the unit weight")
                    + "; se, lo and hi are NaN. Use se_method='placebo'.",
                    UserWarning, stacklevel=2,
                )
            else:
                n = reps.size
                se = float(np.sqrt((n - 1) / n * np.sum((reps - tau) ** 2)))
        if np.isfinite(se):
            z = float(norm.ppf(1.0 - alpha / 2.0))
            lo, hi = tau - z * se, tau + z * se

    y_treated = pd.Series(
        np.concatenate([Y_treated_pre, Y_treated_post]),
        index=np.concatenate([pre_times, post_times]), name="treated",
    )
    y_synthetic = pd.Series(
        np.concatenate([omega @ Y_donors_pre, omega @ Y_donors_post]),
        index=y_treated.index, name="synthetic",
    )

    # Optional result fields, filled when SyntheticDiDResult declares them.
    # n_boot=0 computes no variance, so no estimator is recorded: summary()
    # then prints no "se method" line next to the NaN se.
    extra = {"se_method": method if n_boot > 0 else None, "alpha": float(alpha),
             "n_reps": 0 if reps is None else int(reps.size)}
    extra = {k: v for k, v in extra.items() if k in _RESULT_FIELDS}

    return SyntheticDiDResult(
        tau=float(tau),
        omega=pd.Series(omega, index=donor_units),
        lambda_w=pd.Series(lam, index=pre_times),
        se=se,
        lo=float(lo),
        hi=float(hi),
        treatment_time=T_treat,
        y_treated=y_treated,
        y_synthetic=y_synthetic,
        **extra,
    )


__all__ = ["synthetic_did"]
