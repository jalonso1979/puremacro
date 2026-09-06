"""Narrative sign restrictions for SVARs — Antolín-Díaz & Rubio-Ramírez (2018).

Fuses the package's two flagship layers: :class:`puremacro.narrative.NarrativeEvent`
objects (or plain ``(date, shock, sign)`` tuples) become *identification
inputs* for a sign-restricted SVAR.

Method
------
Antolín-Díaz, J. and Rubio-Ramírez, J.F. (2018). "Narrative Sign
Restrictions for SVARs." *American Economic Review* 108(10), 2802-2829.

Three families of narrative restrictions are supported:

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
* ``'shock_bound'`` (Ludvigson, Ma and Ng 2021): the absolute magnitude
  of shock ``j`` on a date lies in a stated range,
  ``min_magnitude <= |eps[t, j]| <= max_magnitude``. The bound is
  *unsigned* unless ``sign=+1/-1`` is given explicitly.

Algorithm (AD-RR Algorithm 1, with simplifications stated below)
----------------------------------------------------------------
1. Estimate the reduced-form VAR(p) by OLS; take ``P = chol(Sigma)``.
   With ``bayes_draws=True`` a fresh ``(A, c, Sigma)`` is drawn from the
   conjugate Normal-Inverse-Wishart posterior for every rotation draw
   (redrawn until the companion matrix is stable, up to 50 attempts).
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
   surviving draws. The accepted draws' impact matrices (and, in
   Bayesian mode, their autoregressive coefficients) are stored on the
   result so that ``.irf(h)`` / ``.fevd(h)`` for ``h > H`` are the
   weighted medians of the *extended* draws, not a single draw.

Stated simplifications relative to AD-RR (2018)
-----------------------------------------------
* By default (``bayes_draws=False``) the reduced-form parameters are
  **fixed at the OLS estimate** — only rotation (identification)
  uncertainty is sampled, matching the convention of
  :func:`puremacro.var.identify.sign.sign_restriction_svar`.
  ``bayes_draws=True`` integrates over the Normal-Inverse-Wishart
  posterior for (A, Sigma) as AD-RR do.
* Importance weights enter **weighted pointwise percentiles** directly
  instead of AD-RR's resampling (SIR) step; the two are equivalent in
  expectation, and the raw weights are returned so callers can resample.
* A Monte Carlo estimate ``omega_hat = 0`` (possible for very unlikely
  narrative patterns) is floored at ``1 / n_weight_sims`` so the weight
  is capped at ``n_weight_sims`` rather than infinite. The number of
  draws on which the floor binds is reported (``n_weight_floor``) and
  a ``RuntimeWarning`` is emitted when it is positive.

Restriction dates
-----------------
An **integer** restriction date is always a 0-based row index into
``Y`` (it must be ``>= p`` so a residual exists), whether or not
``dates`` is given. Anything else (``pd.Timestamp``, ISO string,
``datetime.date``, ``np.datetime64``) is located in ``dates``; a
``DatetimeIndex``/``PeriodIndex`` on a DataFrame ``Y`` is used as
``dates`` automatically. A date that does not match a stamp exactly is
matched to the observation whose *period* contains it, where the period
length is inferred from the spacing of ``dates`` (annual, quarterly, or
monthly) — so a within-quarter announcement date such as 1979-10-06
hits 1979Q4 on a quarterly index, while on a monthly index it can only
hit October 1979 (a missing month raises instead of being remapped).

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

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ..._linalg import safe_cholesky
from .._results import VarEstimateResult
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import NarrativeSignResult, NarrativeSignSVARResult

_VALID_KINDS = {"shock_sign", "hd_dominance", "shock_bound"}
_VALID_DOMINANCE = {"most", "overwhelming"}
_MAX_STABLE_ATTEMPTS = 50
_ESS_CONCENTRATION_RATIO = 0.10


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
        The restricted date. An integer is always a **row index into
        Y** (0-based; must be ``>= p`` so a residual exists), whether or
        not ``dates`` is supplied. Anything else is coerced with
        ``pd.Timestamp`` and located in ``dates`` (exact stamp, else the
        observation whose period contains the date).
    shock : int
        Structural-shock column the restriction refers to.
    sign : int or None
        +1 or -1. Required for ``'shock_sign'``. For ``'shock_bound'``
        the default ``None`` (or 0) is an *unsigned* magnitude bound;
        pass +1/-1 to additionally restrict the sign. Ignored by
        ``'hd_dominance'``.
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
    sign: int | None = None
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
        if self.sign is not None:
            try:
                sign_int = int(self.sign)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"NarrativeRestriction: sign must be +1, -1, 0 or None; got {self.sign!r}"
                ) from exc
            if sign_int not in (-1, 0, +1):
                raise ValueError(
                    f"NarrativeRestriction: sign must be +1, -1, 0 or None; got {self.sign!r}"
                )
            # 0 means "unsigned" for magnitude bounds; normalise to None.
            object.__setattr__(self, "sign", sign_int if sign_int != 0 else None)
        if self.kind == "shock_sign" and self.sign is None:
            raise ValueError(
                "NarrativeRestriction: shock_sign needs sign in {-1, +1}; "
                f"got {self.sign!r}"
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
                    f"sign={'+' if cast(int, self.sign) > 0 else '-'}1)")
        if self.kind == "shock_bound":
            bounds = []
            if self.min_magnitude is not None:
                bounds.append(f"|eps| >= {self.min_magnitude}")
            if self.max_magnitude is not None:
                bounds.append(f"|eps| <= {self.max_magnitude}")
            if self.sign in (+1, -1):
                bounds.append(f"sign={'+' if cast(int, self.sign) > 0 else '-'}1")
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


def _is_integer_like(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _coerce_dates(dates) -> pd.DatetimeIndex:
    """Coerce ``dates`` (DatetimeIndex, PeriodIndex, or datetime-like sequence)."""
    if isinstance(dates, pd.PeriodIndex):
        return pd.DatetimeIndex(dates.to_timestamp())
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


def _index_period_freq(dates_index: pd.DatetimeIndex) -> str | None:
    """Period frequency implied by the spacing of ``dates_index``.

    Returns ``'Y'``, ``'Q'`` or ``'M'`` for (roughly) annual, quarterly
    and monthly indexes, and ``None`` for anything finer or with fewer
    than two stamps (exact matches only).
    """
    if len(dates_index) < 2:
        return None
    step = float(np.median(np.asarray((dates_index[1:] - dates_index[:-1]).days, dtype=float)))
    if step >= 300.0:
        return "Y"
    if step >= 80.0:
        return "Q"
    if step >= 27.0:
        return "M"
    return None


def _map_date_to_eps_row(date, dates_index, *, p: int, T: int) -> int:
    """Map a restriction date to a row of the residual/shock matrix.

    An integer ``date`` is a 0-based row index into Y (must satisfy
    ``p <= t < T``) regardless of ``dates_index``. Anything else needs
    ``dates_index``: an exact Timestamp match wins; otherwise the date is
    matched to the observation whose period (annual / quarterly /
    monthly, inferred from the index spacing) contains it. Finer indexes
    require an exact match.
    """
    if _is_integer_like(date):
        t = int(date)
        if t < p or t >= T:
            raise ValueError(
                f"narrative_sign_svar: restriction row index {t} out of range "
                f"[{p}, {T - 1}] (the first p={p} rows have no residual)"
            )
        return t - p
    if isinstance(date, (bool, np.bool_, float, np.floating)):
        raise ValueError(
            "narrative_sign_svar: restriction date must be an integer row "
            f"index into Y or a datetime-like; got {date!r} ({type(date).__name__})"
        )
    if dates_index is None:
        raise ValueError(
            "narrative_sign_svar: restriction date must be an integer row "
            "index into Y when no calendar is available (pass `dates=` or "
            "index the DataFrame Y by a DatetimeIndex); got "
            f"{date!r} ({type(date).__name__})"
        )
    try:
        ts = pd.Timestamp(date)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"narrative_sign_svar: cannot interpret restriction date {date!r} "
            "as a timestamp"
        ) from exc
    idx = np.where(dates_index == ts)[0]
    freq = None
    if len(idx) == 0:
        freq = _index_period_freq(dates_index)
        if freq is not None:
            same_period = dates_index.to_period(freq) == pd.Period(ts, freq=freq)
            idx = np.where(np.asarray(same_period))[0]
    if len(idx) == 0:
        tried = {"Y": "exact and same-year", "Q": "exact and same-quarter",
                 "M": "exact and same-month"}.get(cast(str, freq), "exact")
        raise ValueError(
            f"narrative_sign_svar: restriction date {ts.date()} not found in "
            f"`dates` (tried {tried} matches); sample "
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

    Accepted inputs: ``None`` (no traditional restrictions), dict
    {h: (n,) or (n, n)}, a single (n, n) array (applied at h=0), or a
    single (n,) vector (applied at h=0 to shock column 0 — the
    convention of ``sign.sign_restriction_svar``).
    """
    if sign_matrix is None:
        return {}
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


def _stack_coefficients(c: np.ndarray, A_list, p: int) -> np.ndarray:
    """Design-matrix coefficient block ``[c; A_1'; ...; A_p']`` (1 + n p, n)."""
    return np.vstack([np.asarray(c)[None, :]] + [np.asarray(A_list[l]).T for l in range(p)])


def _init_y_from_design(X_design: np.ndarray, p: int, n: int) -> np.ndarray | None:
    """Recover ``Y[:p]`` from the first row of the estimate_var design matrix.

    ``X[0] = [1, y_{p-1}, y_{p-2}, ..., y_0]`` (constant, then lag 1 .. lag p),
    so the pre-sample block is the lag blocks read backwards. Returns
    ``None`` when the design matrix does not have that layout.
    """
    if X_design.ndim != 2 or X_design.shape[1] != 1 + n * p or X_design.shape[0] == 0:
        return None
    first = X_design[0]
    return np.stack([first[1 + l * n : 1 + (l + 1) * n] for l in range(p - 1, -1, -1)])


def _min_draws_for_bands(ci: float) -> int:
    """Accepted draws needed to resolve the ``(1 - ci) / 2`` tail quantiles."""
    return max(10, math.ceil(2.0 / (1.0 - ci)))


def _warn_diagnostics(*, n_accepted: int, n_trad: int, ess: float, ci: float,
                      n_weight_floor: int, n_weight_sims: int,
                      n_unstable: int, n_draws: int) -> None:
    """Emit RuntimeWarnings for collapsed bands / concentrated weights / floors."""
    min_needed = _min_draws_for_bands(ci)
    if n_accepted < min_needed:
        if n_accepted == n_trad:
            survived = (f"only {n_accepted} of {n_draws} draws satisfied the "
                        "traditional sign restrictions")
        else:
            survived = (f"only {n_accepted} of {n_trad} traditionally-accepted "
                        "draws survived the narrative restrictions")
        warnings.warn(
            f"narrative_sign_svar: {survived}; pointwise {ci:.0%} bands need at "
            f"least {min_needed} draws to resolve the tails, so the bands may "
            "collapse onto the median. Increase n_draws.",
            RuntimeWarning, stacklevel=3,
        )
    elif ess < _ESS_CONCENTRATION_RATIO * n_accepted:
        warnings.warn(
            f"narrative_sign_svar: importance weights are concentrated — Kish "
            f"ESS {ess:.1f} of {n_accepted} accepted draws "
            f"(< {_ESS_CONCENTRATION_RATIO:.0%}); the weighted bands are driven "
            "by a handful of draws. Increase n_draws (and n_weight_sims).",
            RuntimeWarning, stacklevel=3,
        )
    if n_weight_floor > 0:
        warnings.warn(
            f"narrative_sign_svar: the Monte Carlo estimate of omega was 0 for "
            f"{n_weight_floor} of {n_accepted} accepted draws and was floored "
            f"at 1/n_weight_sims = 1/{n_weight_sims}; their importance weights "
            f"are capped at {n_weight_sims} and the ESS overstates efficiency. "
            "Increase n_weight_sims.",
            RuntimeWarning, stacklevel=3,
        )
    if n_unstable > 0:
        warnings.warn(
            f"narrative_sign_svar: {n_unstable} of {n_draws} posterior draws "
            f"were unstable (no stable VAR found in {_MAX_STABLE_ATTEMPTS} "
            "attempts) and were skipped (bayes_draws=True); the posterior puts "
            "substantial mass on explosive dynamics.",
            RuntimeWarning, stacklevel=3,
        )


def _resolve_alias(name: str, value, alias_name: str, alias_value):
    """Return the resolved value of ``name``/``alias_name``, erroring on conflict."""
    if value is not None and alias_value is not None and value != alias_value:
        raise ValueError(
            f"narrative_sign_svar: conflicting {name}={value!r} and "
            f"{alias_name}={alias_value!r}; pass only one"
        )
    return value if value is not None else alias_value


def identify_narrative_sign(
    Y: np.ndarray | pd.DataFrame | VarEstimateResult,
    restrictions: list | None = None,
    *,
    p: int | None = None,
    lags: int | None = None,
    horizon: int | None = None,
    horizons: int | None = None,
    sign_matrix: dict | np.ndarray | None = None,
    dates=None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int | None = 0,
) -> NarrativeSignResult:
    """Sign-restricted SVAR sharpened with AD-RR (2018) / LMN (2021) narrative restrictions.

    Parameters
    ----------
    Y : (T, n) ndarray, pd.DataFrame, or VarEstimateResult
        Data matrix or fitted reduced-form VAR result. A DataFrame with
        a ``DatetimeIndex`` (or ``PeriodIndex``) supplies ``dates``
        automatically and its column names become ``result.names``.
    restrictions : list, optional
        Narrative restrictions. Each item may be a
        :class:`NarrativeRestriction`, a plain ``(date, shock, sign)``
        tuple (Type I shorthand), or a
        :class:`puremacro.narrative.NarrativeEvent` (adapter uses the
        event's ``.date`` — the announcement date — and ``.sign``,
        targeting shock 0; see module docstring). An empty list
        reproduces plain traditional sign restrictions with unit
        weights — useful as a comparator.
    p : int, optional
        VAR lag order. Inferred if Y is a VarEstimateResult; required
        otherwise.
    lags : int, optional
        Alias for ``p`` (mirrors :func:`puremacro.var.estimate.estimate_var`).
        Passing both with different values raises ``ValueError``.
    horizon : int, optional
        IRF horizon H. Defaults to 20.
    horizons : int, optional
        Alias for ``horizon``. Passing both with different values raises
        ``ValueError``.
    sign_matrix : dict or ndarray, optional
        Traditional sign restrictions: ``{h: S}`` with ``S`` of shape
        (n, n) — ``S[i, j]`` in {-1, 0, +1} restricts the response of
        variable i to shock j at horizon h — or of shape (n,), applied
        to shock column 0. A bare array is treated as ``{0: S}``.
        ``None`` (the default) imposes no traditional sign restrictions.
    dates : array-like of datetime-like, length T, optional
        Calendar stamps for the rows of ``Y``. Needed only when a
        restriction date is not an integer row index and ``Y`` is not a
        DataFrame with a datetime index. An explicit ``dates`` overrides
        the DataFrame index.
    bayes_draws : bool, default False
        If True, samples reduced-form VAR parameters (A_list, Sigma) from
        the conjugate Normal-Inverse-Wishart posterior (AD-RR full Bayesian
        algorithm) for each draw; a posterior draw is redrawn until the
        VAR is stable (up to 50 attempts) and skipped — with a
        ``RuntimeWarning`` and a count in ``n_unstable_draws`` — if no
        stable draw is found. If False, conditions on OLS point estimates.
    n_draws : int
        Haar rotation draws (>= 1).
    n_weight_sims : int
        Monte Carlo simulations per accepted draw for the importance
        weight (>= 1; unused — closed form — when all restrictions are
        Type I).
    ci : float
        Pointwise band coverage, strictly between 0 and 1 (e.g. 0.9).
    seed : int or None
        Seed for the rotation sampler and the weight simulator. ``None``
        draws fresh entropy (non-reproducible).

    Returns
    -------
    NarrativeSignResult
        Frozen dataclass: weighted-percentile IRF bands, acceptance and
        importance-weight diagnostics, per-draw weights and impact
        matrices, FEVD, and the reduced-form objects of the median-target
        draw (so ``B B' = Sigma`` and the historical-decomposition
        identity hold in both OLS and Bayesian mode).

    Raises
    ------
    RuntimeError
        If no draw satisfies the traditional restrictions, if none of the
        traditionally-accepted draws satisfies the narrative restrictions
        (the message names the most binding restriction), or if every
        posterior draw was unstable in Bayesian mode.
    ValueError / TypeError
        On malformed inputs (message names ``narrative_sign_svar``);
        unknown keyword arguments raise ``TypeError``.

    Warns
    -----
    RuntimeWarning
        When too few draws survive to resolve the requested bands, when
        the importance weights are concentrated (Kish ESS below 10% of
        the accepted draws), when the omega floor binds, or when unstable
        posterior draws were skipped.
    """
    if restrictions is None:
        restrictions = []

    H_resolved = _resolve_alias("horizon", horizon, "horizons", horizons)
    horizon_int = 20 if H_resolved is None else int(H_resolved)
    if horizon_int < 0:
        raise ValueError(f"narrative_sign_svar: horizon must be >= 0; got {H_resolved!r}")
    p = _resolve_alias("p", p, "lags", lags)

    if not (0.0 < float(ci) < 1.0):
        raise ValueError(
            f"narrative_sign_svar: ci must be strictly between 0 and 1; got {ci!r}"
        )
    ci = float(ci)
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(f"narrative_sign_svar: n_draws must be >= 1; got {n_draws}")
    n_weight_sims = int(n_weight_sims)
    if n_weight_sims < 1:
        raise ValueError(
            f"narrative_sign_svar: n_weight_sims must be >= 1; got {n_weight_sims}"
        )

    init_y: np.ndarray | None
    if isinstance(Y, VarEstimateResult):
        A_list_ols = list(Y.A_list)
        c_ols = np.asarray(Y.c, dtype=float)
        Sigma_ols = np.asarray(Y.Sigma, dtype=float)
        resid_ols = np.asarray(Y.resid, dtype=float)
        X_design = np.asarray(Y.X, dtype=float)
        p = len(A_list_ols)
        names = tuple(Y.names)
        T_eff, n = resid_ols.shape
        T = T_eff + p
        B_ols = _stack_coefficients(c_ols, A_list_ols, p)
        Y_dep = resid_ols + X_design @ B_ols
        init_y = _init_y_from_design(X_design, p, n)
    else:
        names = tuple(str(c) for c in Y.columns) if hasattr(Y, "columns") else ()
        if (dates is None and isinstance(Y, pd.DataFrame)
                and isinstance(Y.index, (pd.DatetimeIndex, pd.PeriodIndex))):
            dates = Y.index
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim != 2:
            raise ValueError(
                f"narrative_sign_svar: Y must be a (T, n) matrix; got shape {Y_arr.shape}"
            )
        T, n = Y_arr.shape
        if p is None:
            raise ValueError("narrative_sign_svar: lag order p is required")
        p = int(p)
        A_list_ols, c_ols, Sigma_ols, resid_ols, X_design = estimate_var(Y_arr, p)
        A_list_ols = list(A_list_ols)
        T_eff = resid_ols.shape[0]
        Y_dep = Y_arr[p:]
        init_y = Y_arr[:p].copy()

    # Two independent child streams: the Haar rotation stream is then
    # invariant to the narrative-restriction set for a given seed, so
    # n_traditional_accepted is comparable across specifications.
    _rot_ss, _w_ss = np.random.SeedSequence(seed).spawn(2)
    rng = default_rng(_rot_ss)
    rng_w = default_rng(_w_ss)

    targets = _normalize_sign_matrix(sign_matrix, n, horizon_int)
    restr = [_coerce_restriction(item) for item in restrictions]

    dates_index = None
    if dates is not None:
        dates_index = _coerce_dates(dates)
        if len(dates_index) != T:
            raise ValueError(
                f"narrative_sign_svar: len(dates)={len(dates_index)} does not "
                f"match T={T} rows of Y"
            )

    P_ols = safe_cholesky(Sigma_ols, name="narrative_sign_svar")

    if bayes_draws:
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

    H_full = max(horizon_int, max_L)
    accepted_ir: list[np.ndarray] = []
    accepted_B: list[np.ndarray] = []
    accepted_rf: list[tuple[list[np.ndarray], np.ndarray, np.ndarray]] = []
    accepted_fevd: list[np.ndarray] = []
    weights: list[float] = []
    n_trad = 0
    n_unstable = 0
    n_weight_floor = 0
    fail_counts = np.zeros(len(specs), dtype=int)

    for _ in range(n_draws):
        if bayes_draws:
            # Sample (A_list, c, Sigma) from the NIW posterior until stable.
            stable = False
            for _attempt in range(_MAX_STABLE_ATTEMPTS):
                A_list_cur, c_cur, Sigma_cur = _draw_niw_posterior(
                    B_hat, S_inv, L_S_inv, L_XtX_inv, df, p, n, k_reg, rng
                )
                if _is_stable_var(A_list_cur):
                    stable = True
                    break
            if not stable:
                n_unstable += 1
                continue
            P_cur = safe_cholesky(Sigma_cur, name="narrative_sign_svar_bayes")
            resid_cur = Y_dep - X_design @ _stack_coefficients(c_cur, A_list_cur, p)
        else:
            A_list_cur = A_list_ols
            c_cur = c_ols
            Sigma_cur = Sigma_ols
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

        fevd_k = _fevd_from_irf(ir_full[: horizon_int + 1])

        if not specs:
            accepted_ir.append(ir_full[: horizon_int + 1])
            accepted_B.append(B.copy())
            if bayes_draws:
                accepted_rf.append((list(A_list_cur), np.asarray(c_cur), np.asarray(Sigma_cur)))
            accepted_fevd.append(fevd_k)
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
            omega = float(ok_sim.all(axis=1).mean())
            if omega < 1.0 / n_weight_sims:  # omega_hat == 0: floor, cap weight
                omega = 1.0 / n_weight_sims
                n_weight_floor += 1
        accepted_ir.append(ir_full[: horizon_int + 1])
        accepted_B.append(B.copy())
        if bayes_draws:
            accepted_rf.append((list(A_list_cur), np.asarray(c_cur), np.asarray(Sigma_cur)))
        accepted_fevd.append(fevd_k)
        weights.append(1.0 / omega)

    if bayes_draws and n_unstable == n_draws:
        raise RuntimeError(
            f"narrative_sign_svar: none of the {n_draws} Normal-Inverse-Wishart "
            f"posterior draws produced a stable VAR within {_MAX_STABLE_ATTEMPTS} "
            "attempts each; the reduced-form posterior is explosive. Use "
            "bayes_draws=False, difference or detrend the data, or reduce p."
        )
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

    irf_med = _weighted_quantile(draws, 0.5, w)
    irf_lo = _weighted_quantile(draws, lo_q, w)
    irf_hi = _weighted_quantile(draws, hi_q, w)

    fevd_med = _weighted_median_fevd(np.stack(accepted_fevd, axis=0), w)

    # Median-target representative draw: its impact matrix and — in
    # Bayesian mode — its own reduced-form (A, c, Sigma, residuals), so
    # that B B' = Sigma and the historical decomposition are coherent.
    dists = np.sum((draws - irf_med) ** 2, axis=(1, 2, 3))
    k_star = int(np.argmin(dists))
    B_star = accepted_B[k_star]
    if bayes_draws:
        A_star, c_star, Sigma_star = accepted_rf[k_star]
        resid_star = Y_dep - X_design @ _stack_coefficients(c_star, A_star, p)
        accepted_A_arr: np.ndarray | None = np.stack(
            [np.stack(rf[0], axis=0) for rf in accepted_rf], axis=0
        )  # (m, p, n, n)
    else:
        A_star, c_star, Sigma_star, resid_star = A_list_ols, c_ols, Sigma_ols, resid_ols
        accepted_A_arr = None

    m = len(accepted_ir)
    _warn_diagnostics(
        n_accepted=m, n_trad=n_trad, ess=ess, ci=ci,
        n_weight_floor=n_weight_floor, n_weight_sims=n_weight_sims,
        n_unstable=n_unstable, n_draws=n_draws,
    )

    return NarrativeSignResult(
        irf_median=irf_med,
        irf_lower=irf_lo,
        irf_upper=irf_hi,
        n_draws=n_draws,
        n_traditional_accepted=n_trad,
        n_narrative_accepted=m,
        weights=w,
        ess=ess,
        ci=ci,
        restriction_labels=tuple(spec.label() for spec, _ in specs),
        restriction_fail_counts=tuple(int(x) for x in fail_counts),
        A_list=tuple(np.asarray(a) for a in A_star),
        B=B_star,
        residuals=np.asarray(resid_star),
        intercept=np.asarray(c_star),
        fevd_median=fevd_med,
        names=names,
        Sigma=np.asarray(Sigma_star),
        init_y=init_y,
        accepted_B=np.stack(accepted_B, axis=0),
        accepted_A=accepted_A_arr,
        bayes_draws=bool(bayes_draws),
        n_unstable_draws=n_unstable,
        n_weight_floor=n_weight_floor,
    )


narrative_sign_svar = identify_narrative_sign


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


def _fevd_from_irf(ir: np.ndarray) -> np.ndarray:
    """FEVD shares (H+1, n, n) from an IRF stack (H+1, n, n); rows sum to 1."""
    cum_sq = np.cumsum(ir ** 2, axis=0)
    tot = cum_sq.sum(axis=2, keepdims=True)
    return cum_sq / np.where(tot == 0, 1.0, tot)


def _weighted_median_fevd(fevd_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pointwise weighted median of per-draw FEVDs, renormalised to sum to 1."""
    med = _weighted_quantile(fevd_stack, 0.5, weights)
    tot = med.sum(axis=2, keepdims=True)
    return med / np.where(tot == 0, 1.0, tot)


def _draw_irfs(accepted_B: np.ndarray, accepted_A: np.ndarray | None,
               A_list, horizon: int) -> np.ndarray:
    """IRFs of every accepted draw up to ``horizon``: (m, horizon+1, n, n).

    ``accepted_A`` (m, p, n, n) holds per-draw autoregressive matrices in
    Bayesian mode; when ``None`` all draws share ``A_list`` (OLS mode).
    """
    m = accepted_B.shape[0]
    shared = [np.asarray(a) for a in A_list]
    out = None
    for k in range(m):
        A_k = [np.asarray(a) for a in accepted_A[k]] if accepted_A is not None else shared
        ir_k = compute_irf(A_k, accepted_B[k], horizon)
        if out is None:
            out = np.empty((m,) + ir_k.shape)
        out[k] = ir_k
    return cast(np.ndarray, out)


__all__ = [
    "identify_narrative_sign",
    "narrative_sign_svar",
    "NarrativeRestriction",
    "NarrativeSignResult",
    "NarrativeSignSVARResult",
]
