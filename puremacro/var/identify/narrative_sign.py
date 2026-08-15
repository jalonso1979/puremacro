"""Narrative sign restrictions for SVARs — Antolín-Díaz & Rubio-Ramírez (2018).

Fuses the package's two flagship layers: :class:`puremacro.narrative.NarrativeEvent`
objects (or plain ``(date, shock, sign)`` tuples) become *identification
inputs* for a sign-restricted SVAR.

Method
------
Antolín-Díaz, J. and Rubio-Ramírez, J.F. (2018). "Narrative Sign
Restrictions for SVARs." *American Economic Review* 108(10), 2802-2829.

Two families of narrative restrictions are supported:

* ``'shock_sign'`` (AD-RR Restriction Type I): the identified structural
  shock ``j`` has a stated sign on a stated date, e.g. "the monetary
  shock in October 1979 was positive".
* ``'hd_dominance'`` (AD-RR Restriction Types II/III): shock ``j`` is
  the *most important* (``dominance='most'``, Type II) or the
  *overwhelming* (``dominance='overwhelming'``, Type III) contributor
  to the historical decomposition of the unexpected change in a named
  variable over a date window. With window length ``L`` (in periods),
  the contribution of shock ``k`` to the (L+1)-step-ahead forecast
  error of variable ``i`` at the window end ``t1`` is

      H[i, k] = sum_{l=0..L} (Phi_l B)[i, k] * eps[t1 - l, k],

  their equations (8)-(10); "most important" requires
  ``|H[i, j]| >= max_{k != j} |H[i, k]|`` and "overwhelming" requires
  ``|H[i, j]| >= sum_{k != j} |H[i, k]|``.

Algorithm (AD-RR Algorithm 1, with simplifications stated below)
----------------------------------------------------------------
1. Estimate the reduced-form VAR(p) by OLS; take ``P = chol(Sigma)``.
2. Draw Haar-uniform rotations ``Q`` (QR of a Gaussian matrix with
   positive-diagonal ``R`` sign fix); keep draws whose IRFs satisfy the
   traditional sign restrictions (``sign_matrix``).
3. For each traditionally-accepted draw, compute the structural shocks
   ``eps = u @ inv(B)'`` with ``B = P @ Q`` and keep the draw only if
   every narrative restriction holds on its date(s).
4. Weight each surviving draw by ``1 / omega``, where ``omega`` is the
   probability that all narrative restrictions hold when the structural
   shocks *on the restricted dates* are redrawn i.i.d. N(0, I_n)
   (all other dates' shocks held at their realized values). This is
   AD-RR's importance weight — the inverse of the probability that the
   narrative sign pattern is satisfied under the draw — which undoes
   the bias toward parameter values for which the narrative events
   would be unsurprising. When *all* restrictions are of type
   ``'shock_sign'``, omega is available in closed form as
   ``0.5 ** (number of distinct (date, shock) pairs)`` and no
   simulation is used; otherwise omega is estimated by Monte Carlo
   with ``n_weight_sims`` draws.
5. Report pointwise weighted percentiles of the IRFs across the
   surviving draws.

Stated simplifications relative to AD-RR (2018)
-----------------------------------------------
* Reduced-form parameters are **fixed at the OLS estimate** — only
  rotation (identification) uncertainty is sampled, matching the
  convention of :func:`puremacro.var.identify.sign.sign_restriction_svar`.
  AD-RR integrate over a Normal-Wishart posterior for (A, Sigma).
* Importance weights enter **weighted pointwise percentiles** directly
  instead of AD-RR's resampling (SIR) step; the two are equivalent in
  expectation, and the raw weights are returned so callers can resample.
* A Monte Carlo estimate ``omega_hat = 0`` (possible for very unlikely
  narrative patterns) is floored at ``1 / n_weight_sims`` so the weight
  is capped at ``n_weight_sims`` rather than infinite.

NarrativeEvent adapter
----------------------
A bare :class:`puremacro.narrative.NarrativeEvent` in ``restrictions``
is converted to a ``'shock_sign'`` restriction using the event's
``.date`` field (the *announcement* date) and its ``.sign`` field
(+1/-1); the restriction is attached to structural shock 0 (the first
column of the rotated impact matrix — the convention used throughout
``var.identify`` for "the" identified shock). Events with ``sign == 0``
(ambiguous) are rejected. Use an explicit
:class:`NarrativeRestriction` to target a different shock index or an
``'hd_dominance'`` restriction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import NarrativeSignSVARResult

_VALID_KINDS = {"shock_sign", "hd_dominance", "shock_bound"}
_VALID_DOMINANCE = {"most", "overwhelming"}


@dataclass(frozen=True)
class NarrativeRestriction:
    """One narrative restriction in the sense of AD-RR (2018) or Ludvigson-Ma-Ng (2021).

    Attributes
    ----------
    kind : str
        ``'shock_sign'`` (AD-RR Type I), ``'hd_dominance'``
        (AD-RR Type II with ``dominance='most'``, Type III with
        ``dominance='overwhelming'``), or ``'shock_bound'``
        (Ludvigson, Ma, and Ng 2021 magnitude inequality restrictions).
    date : int or datetime-like
        The restricted date. An integer is interpreted as a **row index
        into Y** (0-based; must be ``>= p`` so a residual exists) when
        ``narrative_sign_svar`` is called without ``dates``; anything
        else is coerced with ``pd.Timestamp`` and located in ``dates``.
    shock : int
        Structural-shock column the restriction refers to.
    sign : int or None
        +1 or -1 for signed shocks; None or 0 for unsigned magnitude bounds.
    variable : int or None
        Variable index whose historical decomposition is restricted;
        required by ``'hd_dominance'``.
    window : int
        Number of *additional* periods after ``date`` included in the
        historical-decomposition window (0 = the single date). Only
        used by ``'hd_dominance'``.
    dominance : str
        ``'most'`` (Type II) or ``'overwhelming'`` (Type III).
    min_magnitude : float or None
        Lower bound on structural shock absolute magnitude ``|eps_{t,j}| >= min_magnitude``.
        Used by ``'shock_bound'``.
    max_magnitude : float or None
        Upper bound on structural shock absolute magnitude ``|eps_{t,j}| <= max_magnitude``.
        Used by ``'shock_bound'``.
    """

    kind: str
    date: object
    shock: int
    sign: int = +1
    variable: int | None = None
    window: int = 0
    dominance: str = "most"
    min_magnitude: float | None = None
    max_magnitude: float | None = None

    def __post_init__(self):
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"NarrativeRestriction: kind {self.kind!r} not in {sorted(_VALID_KINDS)}"
            )
        if self.kind == "shock_sign" and int(self.sign) not in (-1, +1):
            raise ValueError(
                f"NarrativeRestriction: shock_sign needs sign in {{-1, +1}}; got {self.sign!r}"
            )
        if self.kind == "hd_dominance":
            if self.variable is None:
                raise ValueError(
                    "NarrativeRestriction: hd_dominance requires `variable` "
                    "(index of the variable whose historical decomposition is restricted)"
                )
            if self.dominance not in _VALID_DOMINANCE:
                raise ValueError(
                    f"NarrativeRestriction: dominance {self.dominance!r} "
                    f"not in {sorted(_VALID_DOMINANCE)}"
                )
            if int(self.window) < 0:
                raise ValueError(
                    f"NarrativeRestriction: window must be >= 0; got {self.window!r}"
                )
        if self.kind == "shock_bound":
            if self.min_magnitude is None and self.max_magnitude is None:
                raise ValueError(
                    "NarrativeRestriction: shock_bound requires at least one of "
                    "`min_magnitude` or `max_magnitude`"
                )
            if self.min_magnitude is not None and self.min_magnitude < 0:
                raise ValueError("NarrativeRestriction: min_magnitude must be >= 0")
            if self.max_magnitude is not None and self.max_magnitude < 0:
                raise ValueError("NarrativeRestriction: max_magnitude must be >= 0")

    def label(self) -> str:
        if self.kind == "shock_sign":
            return (f"shock_sign(date={self.date!r}, shock={self.shock}, "
                    f"sign={'+' if self.sign > 0 else '-'}1)")
        if self.kind == "shock_bound":
            bounds = []
            if self.min_magnitude is not None:
                bounds.append(f"|eps| >= {self.min_magnitude}")
            if self.max_magnitude is not None:
                bounds.append(f"|eps| <= {self.max_magnitude}")
            if self.sign in (+1, -1):
                bounds.append(f"sign={'+' if self.sign > 0 else '-'}1")
            return f"shock_bound(date={self.date!r}, shock={self.shock}, {', '.join(bounds)})"
        return (f"hd_dominance(date={self.date!r}, shock={self.shock}, "
                f"variable={self.variable}, window={self.window}, "
                f"dominance={self.dominance!r})")


# ---------------------------------------------------------------------------
# Input coercion helpers
# ---------------------------------------------------------------------------

def _coerce_restriction(item) -> NarrativeRestriction:
    """Adapt tuples / NarrativeEvent objects into NarrativeRestriction."""
    if isinstance(item, NarrativeRestriction):
        return item
    # NarrativeEvent adapter (duck-typed to avoid a hard import cycle):
    # uses .date (announcement date) and .sign; targets shock 0.
    if hasattr(item, "date") and hasattr(item, "sign") and hasattr(item, "scoring_method"):
        if int(item.sign) == 0:
            raise ValueError(
                "narrative_sign_svar: NarrativeEvent with sign=0 (ambiguous) "
                "cannot be used as a narrative restriction; set sign=+1/-1 or "
                "build a NarrativeRestriction explicitly"
            )
        return NarrativeRestriction(
            kind="shock_sign", date=item.date, shock=0, sign=int(item.sign)
        )
    if isinstance(item, (tuple, list)) and len(item) == 3:
        date, shock, sign = item
        return NarrativeRestriction(
            kind="shock_sign", date=date, shock=int(shock), sign=int(sign)
        )
    raise TypeError(
        "narrative_sign_svar: each narrative restriction must be a "
        "NarrativeRestriction, a (date, shock, sign) tuple, or a "
        f"NarrativeEvent; got {type(item).__name__}"
    )


def _map_date_to_eps_row(date, dates_index, *, p: int, T: int) -> int:
    """Map a restriction date to a row of the residual/shock matrix.

    Without ``dates``: integer row index into Y (must satisfy p <= t < T).
    With ``dates``: exact Timestamp match, then same year-month, then
    same calendar quarter (first match wins). The two fallbacks let
    within-period announcement dates (e.g. 1979-10-06) hit period-start
    stamped indexes (1979-10-01).
    """
    if dates_index is None:
        if not isinstance(date, (int, np.integer)):
            raise ValueError(
                "narrative_sign_svar: restriction date must be an integer row "
                "index into Y when `dates` is not supplied; got "
                f"{date!r} ({type(date).__name__})"
            )
        t = int(date)
        if t < p or t >= T:
            raise ValueError(
                f"narrative_sign_svar: restriction row index {t} out of range "
                f"[{p}, {T - 1}] (the first p={p} rows have no residual)"
            )
        return t - p
    ts = pd.Timestamp(date)
    idx = np.where(dates_index == ts)[0]
    if len(idx) == 0:
        same_month = (dates_index.year == ts.year) & (dates_index.month == ts.month)
        idx = np.where(same_month)[0]
    if len(idx) == 0:
        same_q = dates_index.to_period("Q") == pd.Period(ts, freq="Q")
        idx = np.where(same_q)[0]
    if len(idx) == 0:
        raise ValueError(
            f"narrative_sign_svar: restriction date {ts.date()} not found in "
            "`dates` (tried exact, year-month, and quarter matches); sample "
            f"runs {dates_index[0].date()} .. {dates_index[-1].date()}"
        )
    t = int(idx[0])
    if t < p:
        raise ValueError(
            f"narrative_sign_svar: restriction date {ts.date()} falls inside "
            f"the first p={p} observations, where no VAR residual exists"
        )
    return t - p


def _normalize_sign_matrix(sign_matrix, n: int, horizon: int) -> dict:
    """Normalize traditional sign restrictions to {h: (n, n) array}.

    Accepted inputs: dict {h: (n,) or (n, n)}, a single (n, n) array
    (applied at h=0), or a single (n,) vector (applied at h=0 to shock
    column 0 — the convention of ``sign.sign_restriction_svar``).
    """
    if sign_matrix is None:
        raise ValueError(
            "narrative_sign_svar: sign_matrix is required (traditional sign "
            "restrictions); pass e.g. {0: S} with S an (n, n) array of +1/-1/0"
        )
    items: Iterable[tuple[Any, Any]]
    if isinstance(sign_matrix, dict):
        items = sign_matrix.items()
    else:
        items = [(0, sign_matrix)]
    out = {}
    for h, S in items:
        h = int(h)
        if h < 0 or h > horizon:
            raise ValueError(
                f"narrative_sign_svar: sign_matrix horizon {h} outside [0, {horizon}]"
            )
        S = np.asarray(S, dtype=float)
        if S.shape == (n,):
            M = np.zeros((n, n))
            M[:, 0] = S
        elif S.shape == (n, n):
            M = S.copy()
        else:
            raise ValueError(
                f"narrative_sign_svar: sign_matrix entry at h={h} must have "
                f"shape ({n},) or ({n}, {n}); got {S.shape}"
            )
        if not np.isin(M, (-1.0, 0.0, 1.0)).all():
            raise ValueError(
                f"narrative_sign_svar: sign_matrix entries must be -1, 0 or +1 "
                f"(0 = unrestricted); offending horizon h={h}"
            )
        out[h] = M
    return out


def _check_sign_matrix(ir: np.ndarray, targets: dict) -> bool:
    for h, S in targets.items():
        mask = S != 0
        if np.any(np.sign(ir[h])[mask] != S[mask]):
            return False
    return True


# ---------------------------------------------------------------------------
# Narrative-restriction evaluation
# ---------------------------------------------------------------------------

def _eval_restrictions(specs, eps_rows: np.ndarray, ir_full: np.ndarray,
                       row_loc: dict) -> np.ndarray:
    """Evaluate all narrative restrictions on a batch of shock paths.

    Parameters
    ----------
    specs : list of (NarrativeRestriction, t_eps) with t_eps the
        eps-row of the restriction date.
    eps_rows : (S, n_rows, n) — structural shocks at the restricted
        rows only, for S paths (S=1 for the realized path).
    ir_full : (L_max+1, n, n) — Phi_l @ B stack.
    row_loc : dict mapping global eps-row -> local index into eps_rows.

    Returns
    -------
    (S, n_restrictions) boolean array of pass/fail flags.
    """
    S = eps_rows.shape[0]
    out = np.empty((S, len(specs)), dtype=bool)
    for r, (spec, t_eps) in enumerate(specs):
        if spec.kind == "shock_sign":
            vals = eps_rows[:, row_loc[t_eps], spec.shock]
            out[:, r] = np.sign(vals) == spec.sign
        elif spec.kind == "shock_bound":
            vals = eps_rows[:, row_loc[t_eps], spec.shock]
            ok = np.ones(S, dtype=bool)
            if spec.min_magnitude is not None:
                ok &= (np.abs(vals) >= spec.min_magnitude)
            if spec.max_magnitude is not None:
                ok &= (np.abs(vals) <= spec.max_magnitude)
            if spec.sign in (+1, -1):
                ok &= (np.sign(vals) == spec.sign)
            out[:, r] = ok
        else:  # hd_dominance
            L = int(spec.window)
            t1 = t_eps + L
            # M[l, k] = (Phi_l B)[variable, k]; shocks at t1-l pair with lag l.
            M = ir_full[: L + 1, spec.variable, :]           # (L+1, n)
            rows = [row_loc[t1 - l] for l in range(L + 1)]
            e = eps_rows[:, rows, :]                          # (S, L+1, n)
            contrib = np.einsum("lk,slk->sk", M, e)           # (S, n)
            absC = np.abs(contrib)
            own = absC[:, spec.shock]
            others = np.delete(absC, spec.shock, axis=1)
            if spec.dominance == "most":
                out[:, r] = own >= others.max(axis=1)
            else:  # overwhelming
                out[:, r] = own >= others.sum(axis=1)
    return out


def _is_stable_var(A_list: list[np.ndarray]) -> bool:
    """Check stability of VAR(p) companion matrix (spectral radius < 1)."""
    p = len(A_list)
    n = A_list[0].shape[0]
    if p == 1:
        eigs = np.linalg.eigvals(A_list[0])
        return bool(np.max(np.abs(eigs)) < 0.999)
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = np.hstack(A_list)
    comp[n:, :-n] = np.eye(n * (p - 1))
    eigs = np.linalg.eigvals(comp)
    return bool(np.max(np.abs(eigs)) < 0.999)


def _draw_niw_posterior(
    B_hat: np.ndarray,
    S_inv: np.ndarray,
    L_S_inv: np.ndarray,
    L_XtX_inv: np.ndarray,
    df: int,
    p: int,
    n: int,
    k_reg: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Draw (A_list, c, Sigma) from conjugate Normal-Inverse-Wishart posterior."""
    # Bartlett decomposition for Inverse-Wishart draw
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(rng.chisquare(max(1, df - i)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    Z = L_S_inv @ A
    W = Z @ Z.T
    Sigma_draw = np.linalg.inv(W)
    L_Sigma = np.linalg.cholesky(Sigma_draw)
    
    # Draw coefficients B ~ MN(B_hat, Sigma_draw, XtX_inv)
    Z_B = rng.standard_normal((k_reg, n))
    B_draw = B_hat + L_XtX_inv @ Z_B @ L_Sigma.T
    c_draw = B_draw[0, :]
    A_list_draw = [B_draw[1 + l * n : 1 + (l + 1) * n, :].T for l in range(p)]
    return A_list_draw, c_draw, Sigma_draw


def narrative_sign_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    sign_matrix,
    restrictions,
    dates=None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> NarrativeSignSVARResult:
    """Sign-restricted SVAR sharpened with AD-RR (2018) / LMN (2021) narrative restrictions.

    Parameters
    ----------
    Y : (T, n) ndarray
        Data matrix.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon H.
    sign_matrix : dict or ndarray
        Traditional sign restrictions: ``{h: S}`` with ``S`` of shape
        (n, n) — ``S[i, j]`` in {-1, 0, +1} restricts the response of
        variable i to shock j at horizon h — or of shape (n,), applied
        to shock column 0. A bare array is treated as ``{0: S}``.
    restrictions : list
        Narrative restrictions. Each item may be a
        :class:`NarrativeRestriction`, a plain ``(date, shock, sign)``
        tuple (Type I shorthand), or a
        :class:`puremacro.narrative.NarrativeEvent` (adapter uses the
        event's ``.date`` — the announcement date — and ``.sign``,
        targeting shock 0; see module docstring). An empty list
        reproduces plain traditional sign restrictions with unit
        weights — useful as a comparator.
    dates : array-like of datetime-like, length T, optional
        Calendar stamps for the rows of ``Y``. Required when any
        restriction date is not an integer row index.
    bayes_draws : bool, default False
        If True, samples reduced-form VAR parameters (A_list, Sigma) from
        the conjugate Normal-Inverse-Wishart posterior (AD-RR full Bayesian
        algorithm) for each draw. If False, conditions on OLS point estimates.
    n_draws : int
        Haar rotation draws.
    n_weight_sims : int
        Monte Carlo simulations per accepted draw for the importance
        weight (unused — closed form — when all restrictions are Type I).
    ci : float
        Pointwise band coverage (e.g. 0.9).
    seed : int
        Seed for the rotation sampler and the weight simulator.

    Returns
    -------
    NarrativeSignSVARResult
        Frozen dataclass: weighted-percentile IRF bands, acceptance and
        importance-weight diagnostics, per-draw weights.

    Raises
    ------
    RuntimeError
        If no draw satisfies the traditional restrictions, or if none
        of the traditionally-accepted draws satisfies the narrative
        restrictions (the message names the most binding restriction).
    ValueError / TypeError
        On malformed inputs (message names ``narrative_sign_svar``).
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    # Two independent child streams: the Haar rotation stream is then
    # invariant to the narrative-restriction set for a given seed, so
    # n_traditional_accepted is comparable across specifications.
    _rot_ss, _w_ss = np.random.SeedSequence(seed).spawn(2)
    rng = default_rng(_rot_ss)
    rng_w = default_rng(_w_ss)

    targets = _normalize_sign_matrix(sign_matrix, n, horizon)
    restr = [_coerce_restriction(item) for item in restrictions]

    dates_index = None
    if dates is not None:
        dates_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        if len(dates_index) != T:
            raise ValueError(
                f"narrative_sign_svar: len(dates)={len(dates_index)} does not "
                f"match T={T} rows of Y"
            )

    A_list_ols, c_ols, Sigma_ols, resid_ols, X_design = estimate_var(Y, p)
    P_ols = safe_cholesky(Sigma_ols, name="narrative_sign_svar")
    T_eff = resid_ols.shape[0]
    
    if bayes_draws:
        Y_dep = Y[p:]
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        B_hat = XtX_inv @ (X_design.T @ Y_dep)
        U_hat = Y_dep - X_design @ B_hat
        S_hat = U_hat.T @ U_hat
        df = max(n + 1, T_eff - X_design.shape[1])
        S_inv = np.linalg.inv(S_hat)
        L_S_inv = np.linalg.cholesky(S_inv)
        L_XtX_inv = np.linalg.cholesky(XtX_inv)
        k_reg = X_design.shape[1]

    # Resolve restriction dates to eps rows; validate shock/variable indices.
    specs: list[tuple[NarrativeRestriction, int]] = []
    for spec in restr:
        if not (0 <= spec.shock < n):
            raise ValueError(
                f"narrative_sign_svar: shock index {spec.shock} outside [0, {n - 1}] "
                f"for restriction {spec.label()}"
            )
        # __post_init__ guarantees `variable is not None` for hd_dominance.
        if spec.kind == "hd_dominance" and not (0 <= cast(int, spec.variable) < n):
            raise ValueError(
                f"narrative_sign_svar: variable index {spec.variable} outside "
                f"[0, {n - 1}] for restriction {spec.label()}"
            )
        t_eps = _map_date_to_eps_row(spec.date, dates_index, p=p, T=T)
        if spec.kind == "hd_dominance" and t_eps + spec.window >= T_eff:
            raise ValueError(
                f"narrative_sign_svar: hd_dominance window ending at eps row "
                f"{t_eps + spec.window} exceeds the sample (T_eff={T_eff}) "
                f"for restriction {spec.label()}"
            )
        specs.append((spec, t_eps))

    # Union of eps rows touched by any restriction (for the weight sims).
    touched: set[int] = set()
    max_L = 0
    for spec, t_eps in specs:
        if spec.kind in ("shock_sign", "shock_bound"):
            touched.add(t_eps)
        else:
            touched.update(range(t_eps, t_eps + spec.window + 1))
            max_L = max(max_L, int(spec.window))
    rows = sorted(touched)
    row_loc = {t: i for i, t in enumerate(rows)}

    all_type1 = all(spec.kind == "shock_sign" for spec, _ in specs)
    n_distinct_type1 = len({(t, spec.shock) for spec, t in specs
                            if spec.kind == "shock_sign"})

    H_full = max(horizon, max_L)
    accepted_ir: list[np.ndarray] = []
    weights: list[float] = []
    n_trad = 0
    fail_counts = np.zeros(len(specs), dtype=int)

    for _ in range(n_draws):
        if bayes_draws:
            # Sample (A_list, c, Sigma) from NIW posterior until stable
            for _attempt in range(50):
                A_list_cur, c_cur, Sigma_cur = _draw_niw_posterior(
                    B_hat, S_inv, L_S_inv, L_XtX_inv, df, p, n, k_reg, rng
                )
                if _is_stable_var(A_list_cur):
                    break
            P_cur = safe_cholesky(Sigma_cur, name="narrative_sign_svar_bayes")
            # Calculate current residuals
            B_cur = np.vstack([c_cur[None, :]] + [A_list_cur[l].T for l in range(p)])
            resid_cur = Y_dep - X_design @ B_cur
        else:
            A_list_cur = A_list_ols
            P_cur = P_ols
            resid_cur = resid_ols

        A = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        B = P_cur @ Q
        ir_full = compute_irf(A_list_cur, B, H_full)  # (H_full+1, n, n)
        if not _check_sign_matrix(ir_full, targets):
            continue
        n_trad += 1
        if not specs:
            accepted_ir.append(ir_full[: horizon + 1])
            weights.append(1.0)
            continue
        eps = resid_cur @ np.linalg.inv(B).T  # (T_eff, n) structural shocks
        eps_rows = eps[rows][None, :, :]  # (1, n_rows, n)
        ok = _eval_restrictions(specs, eps_rows, ir_full, row_loc)[0]
        fail_counts += ~ok
        if not ok.all():
            continue
        # --- AD-RR importance weight: 1 / P(narrative restrictions hold) ---
        if all_type1:
            omega = 0.5 ** n_distinct_type1
        else:
            sims = rng_w.standard_normal((n_weight_sims, len(rows), n))
            ok_sim = _eval_restrictions(specs, sims, ir_full, row_loc)
            omega = ok_sim.all(axis=1).mean()
            omega = max(omega, 1.0 / n_weight_sims)  # floor: cap weight
        accepted_ir.append(ir_full[: horizon + 1])
        weights.append(1.0 / omega)

    if n_trad == 0:
        raise RuntimeError(
            f"narrative_sign_svar: none of the {n_draws} Haar draws satisfied "
            "the traditional sign restrictions; relax sign_matrix or increase n_draws"
        )
    if not accepted_ir:
        worst = int(np.argmax(fail_counts))
        spec = specs[worst][0]
        raise RuntimeError(
            f"narrative_sign_svar: 0 of the {n_trad} traditionally-accepted "
            f"draws satisfied the narrative restrictions. Most binding: "
            f"{spec.label()}, which failed in {int(fail_counts[worst])}/{n_trad} "
            "accepted draws. Check the restriction's date/shock/sign, or "
            "increase n_draws."
        )

    draws = np.stack(accepted_ir, axis=0)          # (m, H+1, n, n)
    w = np.asarray(weights, dtype=float)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    ess = float(w.sum() ** 2 / (w ** 2).sum())

    return NarrativeSignSVARResult(
        irf_median=_weighted_quantile(draws, 0.5, w),
        irf_lower=_weighted_quantile(draws, lo_q, w),
        irf_upper=_weighted_quantile(draws, hi_q, w),
        n_draws=n_draws,
        n_traditional_accepted=n_trad,
        n_narrative_accepted=len(accepted_ir),
        weights=w,
        ess=ess,
        ci=ci,
        restriction_labels=tuple(spec.label() for spec, _ in specs),
        restriction_fail_counts=tuple(int(x) for x in fail_counts),
    )


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> np.ndarray:
    """Weighted quantile along axis 0 (midpoint-CDF rule).

    values : (m, ...) draws; weights : (m,) positive importance weights.
    For equal weights this reproduces the usual midpoint empirical
    quantile; for m=1 it returns the single draw.
    """
    m = values.shape[0]
    shape_rest = values.shape[1:]
    flat = values.reshape(m, -1)
    order = np.argsort(flat, axis=0)
    sorted_vals = np.take_along_axis(flat, order, axis=0)
    w_norm = weights / weights.sum()
    sorted_w = w_norm[order]                       # (m, K)
    cdf = np.cumsum(sorted_w, axis=0) - 0.5 * sorted_w
    out = np.empty(flat.shape[1])
    for k in range(flat.shape[1]):
        out[k] = np.interp(q, cdf[:, k], sorted_vals[:, k])
    return out.reshape(shape_rest)


__all__ = ["narrative_sign_svar", "NarrativeRestriction"]
