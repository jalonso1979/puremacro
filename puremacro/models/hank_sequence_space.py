"""Sequence-Space Heterogeneous-Agent New Keynesian (HANK) Model.

Implements the Sequence-Space Jacobian framework of Auclert, Bardóczy,
Rognlie & Straub (2021, *Econometrica*):

- Stationary incomplete-markets household problem (Aiyagari-Bewley-Huggett)
  solved by the Endogenous Grid Method (EGM) to a tight fixed point, with the
  stationary distribution obtained from the exact lottery transition matrix.
- Fake News Algorithm: the household consumption Jacobians with respect to the
  real interest rate and to income are built from the genuine date-0 policy
  responses and date-1 distribution responses of the EGM household block
  (one backward pass per input, then the ABRS expectation-vector recursion).
- General-equilibrium linear transition by one ``T x T`` solve, non-linear MIT
  transitions by Broyden's method started from the inverse of that linearised
  system, and partial-equilibrium targeted fiscal transfers whose dynamics come
  from the household block itself.

Timing conventions
------------------
Households enter date ``t`` with assets ``a`` that earn the *realised* return
``r_t``: cash-on-hand is ``(1 + r_t) a + w_t s``.  The Euler equation at ``t``
uses ``r_{t+1}``, the return on assets carried into ``t + 1``.  The GE block
determines the *ex-ante* real rate ``r^{ea}_t = i_t - pi_{t+1}`` from the Taylor
rule and the Phillips curve (``M_r_Y @ dY + eps``); it is the return households
earn between ``t`` and ``t + 1``, so ``r_{t+1} = r^{ea}_t`` while ``r_0 = r_ss``
is predetermined (ABRS: ``1 + r_{t+1} = (1 + i_t) / (1 + pi_{t+1})``).  Every
``irf_rate`` array and ``jacobian_c_r`` in this module are dated by the ex-ante
convention of the GE block; ``fake_news(shock_input="r")`` returns the ABRS
matrix dated by the realised return (``J^{r}[t, s + 1] == jacobian_c_r[t, s]``).

Government
----------
Households' assets are government debt, constant at ``B = A_ss``.  The interest
bill ``r_t B`` is financed date by date by a proportional labour-income tax,
``tau_t w_t N = r_t B``, so the budget constraint is
``c + a' = (1 + r_t) a + (1 - tau_t) w_t s``.  Without this closure a rate
change would inject unbacked interest income and the linearised GE system has
no bounded solution.  ``Y = C + G`` is the only market cleared; the asset market
is not (see docs/models.md).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.plot import _palette
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

# Two-state idiosyncratic productivity process shared by every household block
# in this module.  It is fixed in the source (see docs/models.md).
_S_GRID = np.array([0.5, 1.5])
_PI_S = np.array([[0.9, 0.1], [0.1, 0.9]])
_W_SS = 1.0

# Keyword overrides accepted by solve_nonlinear_transition when a pre-solved
# steady state is supplied.
_HH_PARAM_KEYS = ("beta", "gamma", "r_ss", "n_a", "a_max")
_SS_PASSTHROUGH_KEYS = ("T", "shock_magnitude", "shock_rho")
_GE_PARAM_KEYS = ("phi_pi", "kappa")

_EGM_TOL = 1e-12
_EGM_MAX_ITER = 20000
_FD_STEP = 1e-4


# ---------------------------------------------------------------------------
# Household block primitives
# ---------------------------------------------------------------------------

def _income_process(n_s: int) -> tuple[np.ndarray, np.ndarray]:
    if n_s != len(_S_GRID):
        raise ValueError(
            f"the household block ships a {len(_S_GRID)}-state income process; got n_s={n_s}"
        )
    return _S_GRID.copy(), _PI_S.copy()


def _asset_grid(n_a: int, a_max: float) -> np.ndarray:
    return np.geomspace(1e-4, a_max + 1e-4, n_a) - 1e-4


def _egm_step(
    V_next: np.ndarray,
    a_grid: np.ndarray,
    s_grid: np.ndarray,
    pi_s: np.ndarray,
    beta: float,
    gamma: float,
    r_t: float,
    w_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One backward EGM step.

    ``V_next[a', s']`` is the marginal value of assets carried into ``t + 1``,
    ``(1 + r_{t+1}) u'(c_{t+1}(a', s'))``.  Returns ``(V_t, c_t, a_t)`` on the
    asset grid given the realised return ``r_t`` and wage ``w_t`` at date ``t``.
    """
    exp_V = V_next @ pi_s.T
    c_endo = (beta * np.maximum(exp_V, 1e-300)) ** (-1.0 / gamma)
    R = max(1.0 + r_t, 1e-6)
    c_t = np.empty_like(c_endo)
    a_t = np.empty_like(c_endo)
    for j in range(len(s_grid)):
        y = w_t * s_grid[j]
        a_endo = (c_endo[:, j] + a_grid - y) / R
        cash = R * a_grid + y
        c = np.interp(a_grid, a_endo, c_endo[:, j])
        c = np.maximum(np.minimum(c, cash), 1e-12)  # a' >= 0 binds where c would exceed cash
        c_t[:, j] = c
        a_t[:, j] = cash - c
    V_t = R * c_t ** (-gamma)
    return V_t, c_t, a_t


def _lottery(a_dest: np.ndarray, a_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolation lottery: split each destination between neighbouring grid points."""
    n_a = len(a_grid)
    idx = np.searchsorted(a_grid, a_dest)
    idx_h = np.clip(idx, 0, n_a - 1)
    idx_l = np.clip(idx - 1, 0, n_a - 1)
    span = a_grid[idx_h] - a_grid[idx_l]
    safe = np.where(span > 0, span, 1.0)
    w_h = np.where(span > 0, (a_dest - a_grid[idx_l]) / safe, 0.0)
    w_h = np.clip(w_h, 0.0, 1.0)
    return idx_l, idx_h, 1.0 - w_h, w_h


def _forward_step(D: np.ndarray, a_pol: np.ndarray, a_grid: np.ndarray, pi_s: np.ndarray) -> np.ndarray:
    """Push the distribution one period forward: D_{t+1}(a', s') = sum_s pi(s, s') Lottery(a_pol) D_t(., s)."""
    idx_l, idx_h, w_l, w_h = _lottery(a_pol, a_grid)
    n_a, n_s = D.shape
    D_end = np.zeros_like(D)
    for j in range(n_s):
        D_end[:, j] = (
            np.bincount(idx_l[:, j], weights=D[:, j] * w_l[:, j], minlength=n_a)
            + np.bincount(idx_h[:, j], weights=D[:, j] * w_h[:, j], minlength=n_a)
        )
    return D_end @ pi_s


def _build_transition_matrix(a_pol: np.ndarray, a_grid: np.ndarray, pi_s: np.ndarray) -> np.ndarray:
    """Dense (N, N) transition matrix Lambda with D_{t+1} = Lambda @ D_t, state index a_i * n_s + s_i."""
    n_a, n_s = a_pol.shape
    N = n_a * n_s
    idx_l, idx_h, w_l, w_h = _lottery(a_pol, a_grid)
    cols = np.arange(n_a)[:, None] * n_s + np.arange(n_s)[None, :]
    Lam = np.zeros((N, N))
    for s_next in range(n_s):
        p = pi_s[:, s_next][None, :]
        rows_l = idx_l * n_s + s_next
        rows_h = idx_h * n_s + s_next
        np.add.at(Lam, (rows_l.ravel(), cols.ravel()), (w_l * p).ravel())
        np.add.at(Lam, (rows_h.ravel(), cols.ravel()), (w_h * p).ravel())
    return Lam


def _stationary_distribution(Lam: np.ndarray) -> np.ndarray:
    """Exact stationary distribution of a column-stochastic Lambda (linear solve, power-iteration fallback)."""
    N = Lam.shape[0]
    A = Lam - np.eye(N)
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    ok = False
    D = np.full(N, 1.0 / N)
    try:
        D_try = np.linalg.solve(A, b)
        if np.all(np.isfinite(D_try)) and float(np.max(np.abs(Lam @ D_try - D_try))) < 1e-9 and float(D_try.min()) > -1e-9:
            D, ok = D_try, True
    except np.linalg.LinAlgError:
        ok = False
    if not ok:
        converged = False
        for _ in range(100000):
            D_new = Lam @ D
            if float(np.max(np.abs(D_new - D))) < 1e-14:
                D = D_new
                converged = True
                break
            D = D_new
        if not converged:
            warnings.warn(
                "stationary distribution iteration did not reach 1e-14; the steady state may be inexact",
                RuntimeWarning,
                stacklevel=3,
            )
    D = np.maximum(D, 0.0)
    return D / D.sum()


@dataclass
class _HouseholdBlock:
    """Steady-state household block plus the primitives needed to perturb it."""
    a_grid: np.ndarray
    s_grid: np.ndarray
    pi_s: np.ndarray
    beta: float
    gamma: float
    r_ss: float
    w_ss: float
    V_ss: np.ndarray
    c_ss: np.ndarray
    a_ss: np.ndarray
    D_ss: np.ndarray
    Lambda: np.ndarray
    B: float
    N_ss: float
    egm_iterations: int
    converged: bool

    @property
    def C_ss(self) -> float:
        return float(np.sum(self.D_ss * self.c_ss))

    @property
    def tax_rate(self) -> float:
        """Steady-state labour-income tax rate tau_ss = r_ss B / (w_ss N_ss)."""
        return self.r_ss * self.B / (self.w_ss * self.N_ss)

    def net_wage(self, r_t: float, w_t: float) -> float:
        """After-tax wage (1 - tau_t) w_t with tau_t w_t N_ss = r_t B (balanced budget)."""
        return w_t - r_t * self.B / self.N_ss


def _stationary_labour(s_grid: np.ndarray, pi_s: np.ndarray) -> float:
    """Aggregate efficiency units of labour N = sum_s pi_stat(s) s."""
    w, v = np.linalg.eig(pi_s.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    p = np.real(v[:, k])
    p = p / p.sum()
    return float(np.dot(p, s_grid))


def _egm_fixed_point(
    V0: np.ndarray, c0: np.ndarray, a_grid: np.ndarray, s_grid: np.ndarray, pi_s: np.ndarray,
    beta: float, gamma: float, r: float, w_net: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    V, c = V0, c0
    a = np.zeros_like(c)
    converged = False
    it = 0
    for it in range(1, _EGM_MAX_ITER + 1):
        V_new, c_new, a_new = _egm_step(V, a_grid, s_grid, pi_s, beta, gamma, r, w_net)
        err = float(np.max(np.abs(c_new - c)))
        V, c, a = V_new, c_new, a_new
        if err < _EGM_TOL:
            converged = True
            break
    return V, c, a, it, converged


def _solve_household_block(
    *,
    beta: float,
    gamma: float,
    r_ss: float,
    n_a: int,
    a_max: float,
    w_ss: float = _W_SS,
) -> _HouseholdBlock:
    """Steady state with constant government debt B = A_ss financed by a labour-income tax.

    Households hold the government's debt; the interest bill ``r_ss B`` is paid
    for by a proportional tax on labour income, ``tau_ss w_ss N = r_ss B``.  The
    debt level is pinned down by the households' own asset demand, ``B = A_ss``,
    which is solved as a fixed point (secant iterations, each one a full EGM
    solve warm-started from the previous policy).
    """
    a_grid = _asset_grid(n_a, a_max)
    s_grid, pi_s = _income_process(len(_S_GRID))
    N_ss = _stationary_labour(s_grid, pi_s)
    c = r_ss * a_grid[:, None] + w_ss * s_grid[None, :]
    V = (1.0 + r_ss) * c ** (-gamma)
    a = np.zeros_like(c)
    Lam = np.zeros((c.size, c.size))
    D = np.zeros_like(c)
    it_total = 0
    converged = False

    def assets_at(B: float) -> float:
        nonlocal V, c, a, Lam, D, it_total, converged
        w_net = w_ss - r_ss * B / N_ss
        if w_net <= 0:
            raise ValueError(
                f"r_ss * B / N exceeds the wage (B={B:.3f}): the interest bill cannot be financed by labour taxes"
            )
        V, c, a, it, converged = _egm_fixed_point(V, c, a_grid, s_grid, pi_s, beta, gamma, r_ss, w_net)
        it_total += it
        Lam = _build_transition_matrix(a, a_grid, pi_s)
        D = _stationary_distribution(Lam).reshape(a.shape)
        return float(np.sum(D.sum(axis=1) * a_grid))

    B_prev = 0.0
    f_prev = assets_at(B_prev) - B_prev
    B_cur = B_prev + f_prev
    f_cur = assets_at(B_cur) - B_cur
    fiscal_converged = abs(f_cur) < 1e-10 * max(1.0, abs(B_cur))
    for _ in range(50):
        if fiscal_converged:
            break
        denom = f_cur - f_prev
        step = -f_cur * (B_cur - B_prev) / denom if abs(denom) > 1e-14 else f_cur
        B_prev, f_prev = B_cur, f_cur
        B_cur = B_cur + step
        f_cur = assets_at(B_cur) - B_cur
        fiscal_converged = abs(f_cur) < 1e-10 * max(1.0, abs(B_cur))
    if not converged:
        warnings.warn(
            f"EGM steady state did not converge to {_EGM_TOL:.0e} in {_EGM_MAX_ITER} iterations "
            f"(beta*(1+r_ss)={beta * (1.0 + r_ss):.6f}); transitions will not start from an exact fixed point",
            RuntimeWarning,
            stacklevel=3,
        )
    if not fiscal_converged:
        warnings.warn(
            f"government debt fixed point B = A_ss did not converge (|A_ss - B| = {abs(f_cur):.2e})",
            RuntimeWarning,
            stacklevel=3,
        )
    return _HouseholdBlock(
        a_grid=a_grid, s_grid=s_grid, pi_s=pi_s, beta=beta, gamma=gamma, r_ss=r_ss, w_ss=w_ss,
        V_ss=V, c_ss=c, a_ss=a, D_ss=D, Lambda=Lam, B=B_cur, N_ss=N_ss,
        egm_iterations=it_total, converged=converged and fiscal_converged,
    )


def _household_block_from_result(ss: "SequenceSpaceHANKResult") -> _HouseholdBlock:
    c_ss = np.asarray(ss.policy_c, dtype=float)
    a_ss = np.asarray(ss.policy_a, dtype=float)
    D_ss = np.asarray(ss.distribution, dtype=float)
    if c_ss.ndim != 2 or c_ss.size == 0 or a_ss.shape != c_ss.shape or D_ss.shape != c_ss.shape:
        raise ValueError(
            "SequenceSpaceHANKResult must carry policy_c, policy_a and distribution of shape (n_a, n_s) "
            "(as returned by solve_hank_sequence_space)"
        )
    a_grid = np.asarray(ss.asset_grid, dtype=float)
    s_grid, pi_s = _income_process(c_ss.shape[1])
    Lam = np.asarray(ss.trans_matrix, dtype=float)
    if Lam.shape != (c_ss.size, c_ss.size):
        Lam = _build_transition_matrix(a_ss, a_grid, pi_s)
    B = float(ss.government_debt)
    if not np.isfinite(B):
        B = float(np.sum(D_ss.sum(axis=1) * a_grid))
    return _HouseholdBlock(
        a_grid=a_grid, s_grid=s_grid, pi_s=pi_s, beta=float(ss.beta), gamma=float(ss.gamma),
        r_ss=float(ss.r_ss), w_ss=float(ss.w_ss), V_ss=(1.0 + float(ss.r_ss)) * c_ss ** (-float(ss.gamma)),
        c_ss=c_ss, a_ss=a_ss, D_ss=D_ss, Lambda=Lam, B=B, N_ss=_stationary_labour(s_grid, pi_s),
        egm_iterations=0, converged=bool(ss.ss_converged),
    )


def _household_transition(
    hh: _HouseholdBlock,
    rr_seq: np.ndarray,
    w_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-linear household block along a path.

    ``rr_seq`` holds the realised returns ``r_0, ..., r_T`` (length ``T + 1``;
    ``r_T`` enters the date-T policy and hence the date ``T - 1`` Euler
    equation), ``w_seq`` the gross wages ``w_0, ..., w_{T-1}``.  The labour-income
    tax balances the government budget date by date, ``tau_t w_t N = r_t B``.
    Policies are iterated backward from the steady state at ``T`` and the
    distribution is pushed forward from ``D_ss``.  Returns the aggregate
    consumption path ``C_t`` and the policy paths ``c_t``, ``a_t``.
    """
    T = len(w_seq)
    if len(rr_seq) != T + 1:
        raise ValueError("rr_seq must have length T + 1")
    n_a, n_s = hh.c_ss.shape
    c_path = np.empty((T, n_a, n_s))
    a_path = np.empty((T, n_a, n_s))
    # Date-T policy: steady-state continuation, but the realised return r_T is whatever
    # the GE block set at T - 1 (this is exactly what the Fake News column s = T perturbs).
    V_next, _, _ = _egm_step(
        hh.V_ss, hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma, float(rr_seq[T]),
        hh.net_wage(float(rr_seq[T]), hh.w_ss),
    )
    for t in range(T - 1, -1, -1):
        V_next, c_path[t], a_path[t] = _egm_step(
            V_next, hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma, float(rr_seq[t]),
            hh.net_wage(float(rr_seq[t]), float(w_seq[t])),
        )
    D = hh.D_ss.copy()
    C = np.empty(T)
    for t in range(T):
        C[t] = float(np.sum(D * c_path[t]))
        D = _forward_step(D, a_path[t], hh.a_grid, hh.pi_s)
    return C, c_path, a_path


# ---------------------------------------------------------------------------
# Fake News Algorithm (Auclert, Bardóczy, Rognlie & Straub 2021)
# ---------------------------------------------------------------------------

def _fake_news_inputs(
    hh: _HouseholdBlock, shock_input: str, T: int, h: float = _FD_STEP
) -> tuple[np.ndarray, np.ndarray]:
    """ABRS step 1: date-0 policy responses dc_0^s and date-1 distribution responses dD_1^s.

    A unit shock to the input at date ``s`` perturbs the backward step at ``s``
    only (the realised return enters cash-on-hand, the balanced-budget tax and
    the marginal value carried to ``s - 1``; the wage enters cash-on-hand).  The date-0 policy for a shock
    at date ``s`` is that perturbed step followed by ``s`` unperturbed steps, so
    one backward pass of length ``T`` delivers every column: the perturbation
    ``dV`` is propagated through the directional derivative of the steady-state
    step, which cancels any residual drift of the fixed point.
    """
    if shock_input not in ("r", "w"):
        raise ValueError("shock_input must be 'r' or 'w'")
    args = (hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma)
    w_net_ss = hh.net_wage(hh.r_ss, hh.w_ss)
    base_V, base_c, base_a = _egm_step(hh.V_ss, *args, hh.r_ss, w_net_ss)
    D1_base = _forward_step(hh.D_ss, base_a, hh.a_grid, hh.pi_s)
    r_p = hh.r_ss + (h if shock_input == "r" else 0.0)
    w_p = hh.w_ss + (h if shock_input == "w" else 0.0)
    V_p, c_p, a_p = _egm_step(hh.V_ss, *args, r_p, hh.net_wage(r_p, w_p))
    N = hh.c_ss.size
    dc = np.zeros((T, N))
    dD1 = np.zeros((T, N))
    dV = (V_p - base_V) / h
    dc_k = (c_p - base_c) / h
    da_k = (a_p - base_a) / h
    for k in range(T):
        if k > 0:
            V_k, c_k, a_k = _egm_step(hh.V_ss + h * dV, *args, hh.r_ss, w_net_ss)
            dV = (V_k - base_V) / h
            dc_k = (c_k - base_c) / h
            da_k = (a_k - base_a) / h
        dc[k] = dc_k.ravel()
        dD1[k] = ((_forward_step(hh.D_ss, base_a + h * da_k, hh.a_grid, hh.pi_s) - D1_base) / h).ravel()
    return dc, dD1


def _fake_news_recursion(
    T: int,
    c_ss_flat: np.ndarray,
    Lam: np.ndarray,
    D_ss_flat: np.ndarray,
    dc: np.ndarray,
    dD1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ABRS steps 2-4: expectation vectors, fake-news matrix and the Jacobian recursion."""
    N = len(c_ss_flat)
    E = np.zeros((T, N))
    E[0] = c_ss_flat
    LamT = np.ascontiguousarray(Lam.T)
    for t in range(1, T):
        E[t] = LamT @ E[t - 1]
    F = np.zeros((T, T))
    F[0, :] = dc @ D_ss_flat
    if T > 1:
        F[1:, :] = E[:-1] @ dD1.T
    J = np.zeros((T, T))
    J[0, :] = F[0, :]
    for t in range(1, T):
        J[t, 0] = F[t, 0]
        J[t, 1:] = J[t - 1, :-1] + F[t, 1:]
    return J, F, E


def _fake_news(hh: _HouseholdBlock, shock_input: str, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dc, dD1 = _fake_news_inputs(hh, shock_input, T)
    return _fake_news_recursion(T, hh.c_ss.ravel(), hh.Lambda, hh.D_ss.ravel(), dc, dD1)


def _consumption_jacobians(hh: _HouseholdBlock, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(J_C_r, J_C_Y): consumption Jacobians w.r.t. the ex-ante real rate and aggregate income."""
    J_r_real, _, _ = _fake_news(hh, "r", T + 1)
    J_w, _, _ = _fake_news(hh, "w", T)
    J_r = np.ascontiguousarray(J_r_real[:T, 1:T + 1])
    J_y = J_w * (hh.w_ss / hh.C_ss)
    return J_r, J_y


def _ge_matrices(T: int, beta: float, kappa: float, phi_pi: float) -> tuple[np.ndarray, np.ndarray]:
    """NKPC (pi = K_pi @ dY) and ex-ante real-rate map (dr = M_r_Y @ dY + eps)."""
    idx = np.arange(T)
    lead = idx[None, :] - idx[:, None]
    K_pi = np.where(lead >= 0, kappa * beta ** np.maximum(lead, 0), 0.0)
    Shift_K_pi = np.zeros((T, T))
    Shift_K_pi[:-1, :] = K_pi[1:, :]
    M_r_Y = phi_pi * K_pi - Shift_K_pi
    return K_pi, M_r_Y


def _local_mpc(policy_c: np.ndarray, a_grid: np.ndarray, r_ss: float) -> np.ndarray:
    """Quarterly MPC out of a marginal cash windfall, dc / d((1 + r) a), by forward differences."""
    n_a, n_s = policy_c.shape
    mpc = np.zeros((n_a, n_s))
    d_cash = (1.0 + r_ss) * np.diff(a_grid)
    for j in range(n_s):
        mpc[:-1, j] = np.diff(policy_c[:, j]) / d_cash
        mpc[-1, j] = mpc[-2, j]
    return np.clip(mpc, 0.0, 1.0)


def _mass_window_weights(D_a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fraction of the mass at each grid point lying in the cumulative-mass window [lo, hi]."""
    mass = D_a / D_a.sum()
    cum_hi = np.cumsum(mass)
    cum_hi[-1] = 1.0
    cum_lo = cum_hi - mass
    overlap = np.clip(np.minimum(cum_hi, hi) - np.maximum(cum_lo, lo), 0.0, None)
    return np.where(mass > 0, overlap / np.where(mass > 0, mass, 1.0), 0.0)


def _decile_weights(D_a: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """(n_bins, n_a) weights splitting grid-point mass so that every bin carries 1/n_bins of households."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.vstack([_mass_window_weights(D_a, edges[d], edges[d + 1]) for d in range(n_bins)])


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    out = np.interp(x, xp, fp)
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), out)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeNewsResult:
    """Fake News Algorithm decomposition result (Auclert et al. 2021).

    Attributes
    ----------
    jacobian : np.ndarray
        Sequence-space Jacobian ``J[t, s] = dC_t / dx_s`` of aggregate consumption
        with respect to the input ``x`` at date ``s``, shape (T, T).
    fake_news : np.ndarray
        Fake-news matrix ``F`` with ``F[0, s] = D_ss' dc_0^s`` and
        ``F[t, s] = E_{t-1}' dD_1^s`` for ``t >= 1``; ``J[t, s] = J[t-1, s-1] + F[t, s]``.
    expectation_vectors : np.ndarray
        ``E[t] = (Lambda')^t c_ss``, shape (T, N).
    horizon : int
        Horizon length T.
    shock_input : str
        ``'y'`` (aggregate income, transmitted through the wage), ``'w'`` (wage) or
        ``'r'`` (real return realised at date ``s``, ABRS dating; the GE block's
        ex-ante rate ``r^{ea}_s`` is the realised return ``r_{s+1}``).
    """
    jacobian: np.ndarray
    fake_news: np.ndarray
    expectation_vectors: np.ndarray
    horizon: int
    shock_input: str = "y"

    def summary(self) -> str:
        col0 = self.jacobian[:, 0]
        lines = [
            "Fake News Algorithm Decomposition (Auclert et al. 2021)",
            "=" * 68,
            f"Horizon T                       : {self.horizon} periods",
            f"Shock input                     : {self.shock_input}",
            f"Jacobian Frobenius Norm         : {np.linalg.norm(self.jacobian):.6f}",
            f"Fake News Frobenius Norm        : {np.linalg.norm(self.fake_news):.6f}",
            f"Impact Effect (J[0, 0])         : {self.jacobian[0, 0]:.6f}",
            f"Diagonal Average (J[t, t])      : {np.mean(np.diag(self.jacobian)):.6f}",
            f"Column-0 Cumulative Response    : {float(np.sum(col0)):.6f}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def to_frame(self, which: str = "jacobian") -> pd.DataFrame:
        mat = self.fake_news if which.lower() in ("fake_news", "f") else self.jacobian
        return pd.DataFrame(
            mat,
            index=[f"t={t}" for t in range(self.horizon)],
            columns=[f"s={s}" for s in range(self.horizon)],
        )

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (10.5, 4.2)):
        """Plot heatmaps of the Fake News matrix F and Sequence-Space Jacobian J."""
        cmap = "coolwarm" if style != "grayscale" else "gray"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        im1 = ax1.imshow(self.fake_news, cmap=cmap, aspect="auto", origin="upper")
        ax1.set_title(r"Fake News Matrix $\mathcal{F}_{t,s}$", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Shock Date $s$", fontsize=9)
        ax1.set_ylabel("Outcome Date $t$", fontsize=9)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(self.jacobian, cmap=cmap, aspect="auto", origin="upper")
        ax2.set_title(rf"Sequence-Space Jacobian $\mathcal{{J}}^{{C,{self.shock_input}}}_{{t,s}}$", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Shock Date $s$", fontsize=9)
        ax2.set_ylabel("Outcome Date $t$", fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class FiscalTransferResult:
    """Partial-equilibrium simulation of a targeted one-off fiscal transfer at date 0.

    Attributes
    ----------
    irf_consumption : np.ndarray
        Aggregate consumption response path ``dC_t`` (T,), obtained by feeding the
        transfer through the household block: the date-0 consumption response of
        every recipient and the subsequent evolution of the wealth distribution
        under the steady-state policies (no general-equilibrium feedback).
    cumulative_multiplier : float
        ``sum_t dC_t / amount`` over the ``T`` periods simulated.
    impact_mpc : float
        ``dC_0 / amount``: the aggregate date-0 marginal propensity to consume out
        of the transfer.
    mpc_by_group : pd.Series
        MPC out of the same per-capita transfer for the target group, the
        non-target group and the whole economy.
    decile_incidence : pd.DataFrame
        Transfer received and date-0 consumption response by wealth decile
        (deciles carry exactly 10% of households each; grid-point mass is split
        at the boundaries).
    target_group : str
        Recipient group label.
    transfer_amount : float
        Total fiscal outlay (model units: mean quarterly labour income is 1).
    nonlinear : bool
        ``True`` if the exact response to the given ``amount`` was simulated,
        ``False`` for the first-order (per unit of transfer) response.
    """
    irf_consumption: np.ndarray
    cumulative_multiplier: float
    impact_mpc: float
    mpc_by_group: pd.Series
    decile_incidence: pd.DataFrame
    target_group: str
    transfer_amount: float
    nonlinear: bool = False

    def summary(self) -> str:
        lines = [
            f"Targeted Fiscal Transfer Simulation ({self.target_group.capitalize()})",
            "=" * 68,
            f"Total Fiscal Outlay             : {self.transfer_amount:.4f}",
            f"Response                        : {'exact non-linear' if self.nonlinear else 'first-order (per unit of transfer)'}",
            f"Impact MPC (Date 0)             : {self.impact_mpc:.4f}",
            f"Cumulative Fiscal Multiplier    : {self.cumulative_multiplier:.4f}  (sum of dC_t over {len(self.irf_consumption)} quarters / outlay)",
            "-" * 68,
            "Incidence Across Wealth Deciles (Share of Transfer & Consumption):",
            self.decile_incidence.round(4).to_string(),
            "=" * 68,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return self.decile_incidence

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (10.5, 4.2)):
        """Plot consumption impulse response and decile incidence bar chart."""
        colors = _palette(3) if style == "grayscale" else ["#1f77b4", "#ff7f0e", "#2ca02c"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Aggregate consumption path
        h = np.arange(len(self.irf_consumption))
        ax1.plot(h, self.irf_consumption, color=colors[0], lw=2.0, label=f"dC_t (Mult={self.cumulative_multiplier:.2f})")
        ax1.fill_between(h, 0, self.irf_consumption, color=colors[0], alpha=0.15)
        ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
        ax1.set_title(f"Dynamic Consumption Response ({self.target_group})", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Horizon (quarters)", fontsize=9)
        ax1.set_ylabel("Consumption Change dC", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right", frameon=False, fontsize=8)

        # 2. Decile incidence
        deciles = self.decile_incidence.index
        x = np.arange(len(deciles))
        width = 0.35
        ax2.bar(x - width/2, self.decile_incidence["Transfer"], width, label="Transfer Received", color=colors[0], alpha=0.8)
        ax2.bar(x + width/2, self.decile_incidence["Consumption"], width, label="Consumption Jump", color=colors[1], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace("Decile ", "D") for d in deciles], fontsize=8)
        ax2.set_title("Distributional Incidence by Wealth Decile", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Wealth Deciles (D1=Poorest, D10=Wealthiest)", fontsize=9)
        ax2.set_ylabel("Amount", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper right", frameon=False, fontsize=8)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class SequenceSpaceHANKResult:
    """Results from the Sequence-Space HANK general-equilibrium solve.

    Attributes
    ----------
    irf_output : np.ndarray
        General equilibrium output impulse response (T,).
    irf_consumption : np.ndarray
        Aggregate consumption impulse response (T,); equals ``irf_output`` (Y = C).
    irf_inflation : np.ndarray
        Inflation path d_pi (T,).
    irf_rate : np.ndarray
        Ex-ante real interest rate path ``d r_t = i_t - pi_{t+1}`` (T,), the return
        households earn between ``t`` and ``t + 1``.
    jacobian_c_r : np.ndarray
        Household consumption Jacobian ``dC_t / dr_s`` w.r.t. the ex-ante real
        rate at ``s`` (T, T), computed by the Fake News algorithm.
    jacobian_c_y : np.ndarray
        Household consumption Jacobian ``dC_t / dY_s`` w.r.t. aggregate income at
        ``s`` (T, T), transmitted through the wage ``w_s = w_ss (1 + dY_s / Y_ss)``.
    steady_state_mpc : float
        Aggregate quarterly MPC out of a marginal cash windfall.
    mpc_distribution : pd.Series
        Average MPC by wealth decile (each decile holds exactly 10% of households).
    asset_grid : np.ndarray
        Discretized asset grid a.
    steady_state_wealth_dist : np.ndarray
        Stationary marginal distribution over assets.
    policy_c : np.ndarray
        Steady-state consumption policy function c(a, s).
    policy_a : np.ndarray
        Steady-state asset savings policy function a'(a, s).
    distribution : np.ndarray
        Full 2D stationary distribution D(a, s).
    trans_matrix : np.ndarray
        Markov transition matrix Lambda with ``D_{t+1} = Lambda @ D_t`` (state index ``a_i * n_s + s_i``).
    steady_state_consumption : float
        Aggregate steady-state consumption ``C_ss = sum(D * c)`` (equals ``Y_ss``).
    w_ss : float
        Steady-state wage (fixed at 1.0).
    government_debt : float
        Constant real government debt ``B = A_ss`` held by households; its interest
        ``r_t B`` is financed by the proportional labour-income tax ``tau_t``.
    tax_rate : float
        Steady-state labour-income tax rate ``tau_ss = r_ss B / (w_ss N)``.
    ss_converged : bool
        Whether the EGM fixed point (1e-12) and the debt fixed point converged.
    """
    irf_output: np.ndarray
    irf_consumption: np.ndarray
    irf_inflation: np.ndarray
    irf_rate: np.ndarray
    jacobian_c_r: np.ndarray
    jacobian_c_y: np.ndarray
    steady_state_mpc: float
    mpc_distribution: pd.Series
    asset_grid: np.ndarray
    steady_state_wealth_dist: np.ndarray
    policy_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    policy_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    distribution: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    trans_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    beta: float = 0.985
    gamma: float = 1.0
    r_ss: float = 0.01
    phi_pi: float = 1.5
    kappa: float = 0.1
    steady_state_consumption: float = float("nan")
    w_ss: float = _W_SS
    government_debt: float = float("nan")
    tax_rate: float = float("nan")
    ss_converged: bool = True

    def summary(self) -> str:
        y_pk, y_at = _peak(self.irf_output)
        pi_pk, pi_at = _peak(self.irf_inflation)
        lines = [
            "Sequence-Space HANK General Equilibrium Solve (Auclert et al. 2021)",
            "=" * 68,
            f"Horizon T                       : {len(self.irf_output)} periods",
            f"Aggregate Steady-State MPC      : {self.steady_state_mpc:.4f}",
            f"Steady-State Consumption C_ss   : {self.steady_state_consumption:.4f}",
            f"Government Debt B = A_ss        : {self.government_debt:.4f} (tax rate {self.tax_rate:.4f})",
            f"Peak Output Response            : {y_pk:+.6f} (t={y_at})",
            f"Peak Inflation Response         : {pi_pk:+.6f} (t={pi_at})",
            "-" * 68,
            "MPC by Wealth Decile:",
        ]
        for decile, mpc_val in self.mpc_distribution.items():
            lines.append(f"  {decile:<20s}: {mpc_val:.4f}")
        return "\n".join(lines)

    def fake_news(self, T: int | None = None, shock_input: str = "y") -> FakeNewsResult:
        """Fake News decomposition of the household block for ``shock_input`` in {'y', 'w', 'r'}."""
        horizon = int(T or len(self.irf_output))
        return fake_news_algorithm(horizon, ss_model=self, shock_input=shock_input)

    def simulate_transfer(
        self,
        target: str | Sequence[int] = "borrowers",
        amount: float = 1.0,
        T: int | None = None,
        nonlinear: bool = False,
    ) -> FiscalTransferResult:
        """Simulate a targeted fiscal transfer through this steady state's household block."""
        horizon = int(T or len(self.irf_output))
        return simulate_targeted_transfer(
            ss_model=self, target=target, amount=amount, T=horizon, nonlinear=nonlinear,
        )

    def solve_nonlinear(
        self,
        shock_seq: Sequence[float] | np.ndarray | None = None,
        shock_var: str = "r",
        horizon: int = 300,
        max_iter: int = 100,
        tol: float = 1e-6,
        backtracking: bool = True,
        **kwargs: Any,
    ) -> NonlinearHANKResult:
        """Solve non-linear transition dynamics using Broyden's method."""
        return solve_nonlinear_transition(
            ss_model=self,
            shock_seq=shock_seq,
            shock_var=shock_var,
            horizon=horizon,
            max_iter=max_iter,
            tol=tol,
            backtracking=backtracking,
            **kwargs,
        )


def _peak(x: np.ndarray) -> tuple[float, int]:
    """Signed extremum of a path and the date at which it occurs."""
    i = int(np.argmax(np.abs(x)))
    return float(x[i]), i


@dataclass(frozen=True)
class NonlinearHANKResult:
    """Results from Non-Linear Sequence-Space HANK transition dynamics (Auclert et al. 2021).

    Attributes
    ----------
    U : np.ndarray
        Solved sequence of endogenous variables (output deviations dY) over horizon T.
    residuals : np.ndarray
        Market-clearing residual sequence H(U, Z) over horizon T.
    iterations : int
        Number of accepted Broyden steps (0 when the initial guess already satisfies ``tol``).
    converged : bool
        Whether the Broyden solver achieved ||H||_inf < tol.
    linear_path : np.ndarray
        Output path of the linearised model ``dH/dU dY = -dH/dZ Z``, with ``dH/dU``
        built from the Fake News household Jacobians (the first-order limit of
        ``nonlinear_path`` as the shock shrinks).
    nonlinear_path : np.ndarray
        General equilibrium output path from non-linear Broyden solver (equals U).
    norm_history : list[float]
        ``||H||_inf`` at the initial guess and after every accepted step (length ``iterations + 1``).
    irf_output_linear, irf_output_nonlinear : np.ndarray
        Output responses dY (T,).
    irf_consumption_linear, irf_consumption_nonlinear : np.ndarray
        Aggregate consumption responses dC (T,).
    irf_rate_linear, irf_rate_nonlinear : np.ndarray
        Ex-ante real rate responses ``d r_t = i_t - pi_{t+1}`` (T,).
    irf_inflation_linear, irf_inflation_nonlinear : np.ndarray
        Inflation responses (T,).
    shock_var : str
        Shock variable identifier ('r' for monetary, 'G' for fiscal).
    shock_seq : np.ndarray
        Exogenous shock sequence Z over horizon T.
    horizon : int
        Simulation horizon length T.
    steady_state_model : Any
        Underlying steady-state SequenceSpaceHANKResult model (re-solved if overrides were passed).
    tol : float
        Convergence tolerance used.
    jacobian_c_r, jacobian_c_y : np.ndarray
        Household Jacobians at the simulation horizon used for the linear path and ``B_0``.
    """
    U: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool
    linear_path: np.ndarray
    nonlinear_path: np.ndarray
    norm_history: list[float]
    irf_output_linear: np.ndarray
    irf_output_nonlinear: np.ndarray
    irf_consumption_linear: np.ndarray
    irf_consumption_nonlinear: np.ndarray
    irf_rate_linear: np.ndarray
    irf_rate_nonlinear: np.ndarray
    irf_inflation_linear: np.ndarray
    irf_inflation_nonlinear: np.ndarray
    shock_var: str
    shock_seq: np.ndarray
    horizon: int
    steady_state_model: Any = None
    tol: float = 1e-6
    jacobian_c_r: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    jacobian_c_y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def summary(self) -> str:
        """Produce academic text summary of non-linear transition dynamics."""
        s_name = "Monetary Policy Shock" if self.shock_var in ("r", "monetary", "interest_rate", "rate") else "Fiscal Spending Shock"
        peak_shock = float(np.max(np.abs(self.shock_seq)))
        max_res = float(np.max(np.abs(self.residuals)))
        y_lin, y_lin_at = _peak(self.linear_path)
        y_nl, y_nl_at = _peak(self.nonlinear_path)
        c_lin, c_lin_at = _peak(self.irf_consumption_linear)
        c_nl, c_nl_at = _peak(self.irf_consumption_nonlinear)
        r_nl, r_nl_at = _peak(self.irf_rate_nonlinear)
        sum_shock = float(np.sum(self.shock_seq))
        multiplier = float(np.sum(self.nonlinear_path) / sum_shock) if abs(sum_shock) > 1e-12 else 0.0

        lines = [
            "Non-Linear Sequence-Space HANK Transition Dynamics (Auclert et al. 2021)",
            "=" * 72,
            f"Horizon T                       : {self.horizon} quarters",
            f"Shock Variable                  : {self.shock_var.upper()} ({s_name})",
            f"Shock Peak Magnitude            : {peak_shock:.6f}",
            f"Broyden Solver Status           : {'CONVERGED' if self.converged else 'NOT CONVERGED'} in {self.iterations} iterations",
            f"Final Residual ||H||_inf        : {max_res:.6e} (tol {self.tol:.0e})",
            "-" * 72,
            "General Equilibrium Impulse Response Comparison (peak = largest |response|):",
            f"  Peak Output Response (Linear)     : {y_lin:+.6f} at t={y_lin_at}",
            f"  Peak Output Response (Non-linear) : {y_nl:+.6f} at t={y_nl_at}",
            f"  Peak Output Difference (NL - Lin) : {y_nl - y_lin:+.6f}",
            f"  Peak Consumption (Linear)         : {c_lin:+.6f} at t={c_lin_at}",
            f"  Peak Consumption (Non-linear)     : {c_nl:+.6f} at t={c_nl_at}",
            f"  Peak Real Rate (Non-linear)       : {r_nl:+.6f} at t={r_nl_at}",
            f"  Cumulative Output Multiplier      : {multiplier:.4f}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Convert simulation paths into a structured DataFrame."""
        data = {
            "Output_Linear": self.linear_path,
            "Output_Nonlinear": self.nonlinear_path,
            "Consumption_Linear": self.irf_consumption_linear,
            "Consumption_Nonlinear": self.irf_consumption_nonlinear,
            "Rate_Linear": self.irf_rate_linear,
            "Rate_Nonlinear": self.irf_rate_nonlinear,
            "Inflation_Linear": self.irf_inflation_linear,
            "Inflation_Nonlinear": self.irf_inflation_nonlinear,
            "Residual": self.residuals,
        }
        return pd.DataFrame(
            data,
            index=[f"t={t}" for t in range(self.horizon)],
        )

    def to_markdown(self, **kwargs: Any) -> str:
        """Render simulation paths as Markdown table."""
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render simulation paths as LaTeX tabular environment."""
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render simulation paths as Typst table markup."""
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (11.0, 8.0)):
        """Plot 4-panel comparison of linear vs non-linear general equilibrium paths.

        Panels:
        1. Output Y (Linear vs Non-Linear)
        2. Consumption C (Linear vs Non-Linear)
        3. Ex-ante real rate r (Linear vs Non-Linear)
        4. Inflation pi (Linear vs Non-Linear)

        The figure title reports the final market-clearing residual ``||H||_inf``,
        the tolerance and the iteration count.
        """
        colors = _palette(3) if style == "grayscale" else ["#1f77b4", "#d62728", "#2ca02c"]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        t_grid = np.arange(self.horizon)

        panels = [
            (axes[0, 0], self.linear_path, self.nonlinear_path,
             r"Output Path $\mathbf{Y}$ ($Y_t - Y_{ss}$)", "Output Deviation"),
            (axes[0, 1], self.irf_consumption_linear, self.irf_consumption_nonlinear,
             r"Aggregate Consumption $\mathbf{C}$ ($C_t - C_{ss}$)", "Consumption Deviation"),
            (axes[1, 0], self.irf_rate_linear, self.irf_rate_nonlinear,
             r"Ex-ante Real Rate $\mathbf{r}$ ($r_t - r_{ss}$)", "Real Rate Deviation"),
            (axes[1, 1], self.irf_inflation_linear, self.irf_inflation_nonlinear,
             r"Inflation $\mathbf{\pi}$ ($\pi_t - \pi_{ss}$)", "Inflation Deviation"),
        ]
        for ax, lin, nl, title, ylabel in panels:
            ax.plot(t_grid, lin, label="Linear Path", color=colors[0], linestyle="--", lw=1.8)
            ax.plot(t_grid, nl, label="Non-Linear Path", color=colors[1], lw=2.2)
            ax.axhline(0, color="gray", linestyle=":", lw=0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Horizon (quarters)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="best", frameon=False, fontsize=9)

        max_res = float(np.max(np.abs(self.residuals)))
        status = "converged" if self.converged else "NOT converged"
        fig.suptitle(
            rf"Non-linear HANK transition: $\|H\|_\infty = {max_res:.2e}$ (tol {self.tol:.0e}), "
            f"{status} in {self.iterations} Broyden iterations",
            fontsize=11,
        )
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fake_news_algorithm(
    T: int,
    policy_c: np.ndarray | None = None,
    trans_matrix: np.ndarray | None = None,
    D_ss: np.ndarray | None = None,
    *,
    ss_model: SequenceSpaceHANKResult | None = None,
    shock_input: str = "y",
    dc_shocks: np.ndarray | None = None,
    dtrans_shocks: np.ndarray | None = None,
    beta: float = 0.985,
    r_ss: float = 0.01,
    **ss_kwargs: Any,
) -> FakeNewsResult:
    """Execute the Fake News Algorithm of Auclert et al. (2021, Econometrica).

    Computes, for a unit shock to ``shock_input`` at each date ``s``:

    1. the date-0 policy responses ``dc_0^s`` and the date-1 distribution
       responses ``dD_1^s = dLambda_s D_ss`` by one backward EGM pass of the
       household block (perturbed step at ``s`` followed by ``s`` unperturbed steps);
    2. the expectation vectors ``E_t = (Lambda')^t c_ss`` in O(T) matrix-vector products;
    3. the fake-news matrix ``F[0, s] = D_ss' dc_0^s``, ``F[t, s] = E_{t-1}' dD_1^s``;
    4. the Jacobian by the recursion ``J[t, s] = J[t-1, s-1] + F[t, s]`` (Proposition 1).

    The household block comes from ``ss_model`` (a ``SequenceSpaceHANKResult``);
    when neither ``ss_model`` nor explicit ``dc_shocks``/``dtrans_shocks`` are
    given, the steady state is solved here from ``beta``, ``r_ss`` and any
    ``solve_hank_sequence_space`` keyword (``n_a``, ``gamma``, ``a_max``, ...).
    Nothing is approximated by closed-form heuristics.

    Parameters
    ----------
    T : int
        Horizon length for sequences.
    policy_c, trans_matrix, D_ss : np.ndarray, optional
        Steady-state consumption policy (n_a, n_s), transition matrix (N, N) with
        ``D_{t+1} = Lambda @ D_t`` and stationary distribution (n_a, n_s).  Only
        used together with ``dc_shocks``/``dtrans_shocks``; otherwise the block is
        taken from ``ss_model`` or solved internally.
    ss_model : SequenceSpaceHANKResult, optional
        Solved steady state supplying the full household block.
    shock_input : {'y', 'w', 'r'}, default 'y'
        ``'y'``: aggregate income (through the wage ``w = w_ss (1 + dY / Y_ss)``);
        ``'w'``: the wage; ``'r'``: the real return realised at date ``s`` (ABRS
        dating; ``J[t, s + 1]`` equals ``jacobian_c_r[t, s]`` of the GE block).
    dc_shocks, dtrans_shocks : np.ndarray, optional
        Custom ``dc_0^s`` (T, N) and ``dD_1^s`` (T, N); both must be given together
        with ``policy_c``, ``trans_matrix`` and ``D_ss``.
    beta, r_ss : float
        Used only when the steady state is solved internally.
    **ss_kwargs
        Further ``solve_hank_sequence_space`` keywords for the internal solve.

    Returns
    -------
    FakeNewsResult
    """
    T = int(T)
    if T < 1:
        raise ValueError("T must be a positive integer")
    if (dc_shocks is None) != (dtrans_shocks is None):
        raise ValueError("dc_shocks and dtrans_shocks must be supplied together")

    if dc_shocks is not None and dtrans_shocks is not None:
        if policy_c is None or trans_matrix is None or D_ss is None:
            raise ValueError("policy_c, trans_matrix and D_ss are required with custom dc_shocks/dtrans_shocks")
        c_vec = np.asarray(policy_c, dtype=float).ravel()
        d_vec = np.asarray(D_ss, dtype=float).ravel()
        N = len(d_vec)
        Lam = np.asarray(trans_matrix, dtype=float)
        if c_vec.shape != (N,) or Lam.shape != (N, N):
            raise ValueError(
                f"shape mismatch: policy_c has {c_vec.size} states, D_ss has {N}, trans_matrix is {Lam.shape}; "
                f"expected ({N},), ({N},) and ({N}, {N})"
            )
        dc = np.asarray(dc_shocks, dtype=float)
        dD = np.asarray(dtrans_shocks, dtype=float)
        if dc.shape != (T, N) or dD.shape != (T, N):
            raise ValueError(f"dc_shocks and dtrans_shocks must have shape ({T}, {N}); got {dc.shape} and {dD.shape}")
        d_sum = float(d_vec.sum())
        if d_sum > 0:
            d_vec = d_vec / d_sum
        J, F, E = _fake_news_recursion(T, c_vec, Lam, d_vec, dc, dD)
        return FakeNewsResult(jacobian=J, fake_news=F, expectation_vectors=E, horizon=T, shock_input=str(shock_input))

    if ss_model is None:
        if policy_c is not None or trans_matrix is not None or D_ss is not None:
            raise ValueError(
                "fake_news_algorithm cannot back-iterate the household problem from policy_c/trans_matrix/D_ss "
                "alone: pass ss_model=solve_hank_sequence_space(...), or supply dc_shocks and dtrans_shocks"
            )
        ss_model = solve_hank_sequence_space(T=int(ss_kwargs.pop("T", T)), beta=beta, r_ss=r_ss, **ss_kwargs)
    elif not isinstance(ss_model, SequenceSpaceHANKResult):
        raise TypeError(f"ss_model must be a SequenceSpaceHANKResult, got {type(ss_model)}")

    key = str(shock_input).lower().strip()
    hh = _household_block_from_result(ss_model)
    if key in ("y", "income"):
        J, F, E = _fake_news(hh, "w", T)
        scale = hh.w_ss / hh.C_ss
        J, F = J * scale, F * scale
        key = "y"
    elif key in ("w", "wage"):
        J, F, E = _fake_news(hh, "w", T)
        key = "w"
    elif key in ("r", "rate", "interest_rate"):
        J, F, E = _fake_news(hh, "r", T)
        key = "r"
    else:
        raise ValueError(f"Unknown shock_input {shock_input!r}: choose 'y', 'w' or 'r'")
    return FakeNewsResult(jacobian=J, fake_news=F, expectation_vectors=E, horizon=T, shock_input=key)


def simulate_targeted_transfer(
    *,
    D: np.ndarray | None = None,
    policy_c: np.ndarray | None = None,
    asset_grid: np.ndarray | None = None,
    policy_a: np.ndarray | None = None,
    ss_model: SequenceSpaceHANKResult | None = None,
    target: str | Sequence[int] = "borrowers",
    amount: float = 1.0,
    T: int = 40,
    r_ss: float = 0.01,
    nonlinear: bool = False,
) -> FiscalTransferResult:
    """Simulate a targeted one-off fiscal transfer through the household block (partial equilibrium).

    Recipients receive the same per-capita transfer ``tau = amount / eligible_mass``
    at date 0.  Their date-0 consumption response is read off the steady-state
    policy evaluated at the higher cash-on-hand (``c_ss(a + tau / (1 + r), s)``),
    the rest is saved, and the wealth distribution is then pushed forward under
    the steady-state policies, which produces the dynamic consumption path and
    the cumulative multiplier.  Prices are held at their steady-state values (no
    general-equilibrium feedback).

    Parameters
    ----------
    D, policy_c, policy_a, asset_grid : np.ndarray, optional
        Stationary distribution (n_a, n_s), consumption and savings policies
        (n_a, n_s) and asset grid (n_a,).  All four are taken from ``ss_model``
        when it is given; otherwise all four are required.
    ss_model : SequenceSpaceHANKResult, optional
        Solved steady state (also supplies ``r_ss``).
    target : str or sequence of int, default 'borrowers'
        ``'borrowers'`` / ``'hand_to_mouth'`` / ``'bottom_quartile'``: the poorest
        25% of households; ``'unconstrained'`` / ``'wealthy'``: the richest 50%;
        ``'all'`` / ``'universal'``: everyone; or a list of wealth deciles
        (``1`` = poorest ... ``10`` = richest), e.g. ``[1, 2, 3]``.  Groups are
        defined by cumulative mass, splitting grid-point mass at the boundary.
    amount : float, default 1.0
        Total fiscal outlay in model units (mean quarterly labour income is 1).
    T : int, default 40
        Horizon of the dynamic consumption response.
    r_ss : float, default 0.01
        Quarterly real interest rate (ignored when ``ss_model`` is given).
    nonlinear : bool, default False
        ``False``: first-order response per unit of transfer (marginal MPCs and
        the linearised distribution dynamics; scale-free in ``amount``).
        ``True``: exact non-linear response to the given ``amount``.

    Returns
    -------
    FiscalTransferResult
    """
    if ss_model is not None:
        if not isinstance(ss_model, SequenceSpaceHANKResult):
            raise TypeError(f"ss_model must be a SequenceSpaceHANKResult, got {type(ss_model)}")
        D, policy_c, policy_a, asset_grid = ss_model.distribution, ss_model.policy_c, ss_model.policy_a, ss_model.asset_grid
        r_ss = float(ss_model.r_ss)
    if D is None or policy_c is None or asset_grid is None:
        raise ValueError("D, policy_c and asset_grid are required (or pass ss_model=...)")
    if policy_a is None:
        raise ValueError(
            "policy_a (the steady-state savings policy) is required for the dynamic response: "
            "pass policy_a=ss.policy_a or ss_model=ss"
        )
    D_arr = np.asarray(D, dtype=float)
    c_ss = np.asarray(policy_c, dtype=float)
    a_ss = np.asarray(policy_a, dtype=float)
    a_grid = np.asarray(asset_grid, dtype=float)
    if c_ss.ndim != 2 or D_arr.shape != c_ss.shape or a_ss.shape != c_ss.shape or a_grid.shape != (c_ss.shape[0],):
        raise ValueError("D, policy_c and policy_a must share shape (n_a, n_s) and asset_grid must have length n_a")
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("amount must be a positive number")
    T = int(T)
    if T < 1:
        raise ValueError("T must be a positive integer")
    n_a, n_s = c_ss.shape
    _, pi_s = _income_process(n_s)
    D_a = D_arr.sum(axis=1)
    W = _decile_weights(D_a)

    # 1. Eligibility weights e_i: fraction of households at grid point i that receive the transfer
    if isinstance(target, str):
        tgt = target.lower().strip()
        if tgt in ("borrowers", "hand_to_mouth", "constrained", "bottom_quartile", "p25"):
            elig = _mass_window_weights(D_a, 0.0, 0.25)
        elif tgt in ("unconstrained", "wealthy"):
            elig = _mass_window_weights(D_a, 0.5, 1.0)
        elif tgt in ("all", "universal", "lump_sum"):
            elig = np.ones(n_a)
        else:
            raise ValueError(
                f"Unknown target group: {target!r}. Choose 'borrowers', 'unconstrained', 'bottom_quartile', "
                f"'all', or a list of wealth deciles (1..10)."
            )
        label = target
    else:
        deciles = [int(d) for d in target]
        if not deciles or any(d < 1 or d > 10 for d in deciles):
            raise ValueError("target deciles must be integers between 1 (poorest) and 10 (richest)")
        elig = np.clip(W[[d - 1 for d in sorted(set(deciles))]].sum(axis=0), 0.0, 1.0)
        label = "deciles " + ",".join(str(d) for d in sorted(set(deciles)))
    eligible_mass = float(np.sum(elig * D_a))
    if eligible_mass <= 0:
        raise ValueError(f"target group {target!r} has zero mass")
    tau = float(amount) / eligible_mass
    R = 1.0 + r_ss
    C_ss = float(np.sum(D_arr * c_ss))
    transfer_grid = np.repeat((elig * tau)[:, None], n_s, axis=1)

    # 2. Household responses
    dC = np.zeros(T)
    if not nonlinear:
        mpc = _local_mpc(c_ss, a_grid, r_ss)
        dc0 = transfer_grid * mpc
        dC[0] = float(np.sum(D_arr * dc0))
        direction = elig[:, None] * (1.0 - mpc)
        h = _FD_STEP
        dD = tau * (_forward_step(D_arr, a_ss + h * direction, a_grid, pi_s) - _forward_step(D_arr, a_ss, a_grid, pi_s)) / h
        for t in range(1, T):
            dC[t] = float(np.sum(dD * c_ss))
            dD = _forward_step(dD, a_ss, a_grid, pi_s)
        target_mpc = dC[0] / float(amount)
        non_mass = float(np.sum((1.0 - elig) * D_a))
        non_target_mpc = float(np.sum((1.0 - elig)[:, None] * D_arr * mpc) / non_mass) if non_mass > 1e-12 else float("nan")
        agg_mpc = float(np.sum(D_arr * mpc))
    else:
        shift = tau / R
        c_tr = np.column_stack([_interp_extrap(a_grid + shift, a_grid, c_ss[:, j]) for j in range(n_s)])
        a_tr = np.column_stack([_interp_extrap(a_grid + shift, a_grid, a_ss[:, j]) for j in range(n_s)])
        a_tr = np.maximum(a_tr, 0.0)
        dc_full = c_tr - c_ss
        dc0 = elig[:, None] * dc_full
        dC[0] = float(np.sum(D_arr * dc0))
        D_next = (
            _forward_step(elig[:, None] * D_arr, a_tr, a_grid, pi_s)
            + _forward_step((1.0 - elig)[:, None] * D_arr, a_ss, a_grid, pi_s)
        )
        for t in range(1, T):
            dC[t] = float(np.sum(D_next * c_ss)) - C_ss
            D_next = _forward_step(D_next, a_ss, a_grid, pi_s)
        target_mpc = dC[0] / float(amount)
        non_mass = float(np.sum((1.0 - elig) * D_a))
        non_target_mpc = (
            float(np.sum((1.0 - elig)[:, None] * D_arr * dc_full) / (non_mass * tau)) if non_mass > 1e-12 else float("nan")
        )
        agg_mpc = float(np.sum(D_arr * dc_full) / tau)

    impact_mpc = dC[0] / float(amount)
    cumulative_multiplier = float(np.sum(dC) / float(amount))
    mpc_by_group = pd.Series({
        "Target Group": target_mpc,
        "Non-Target Group": non_target_mpc,
        "Aggregate Economy": agg_mpc,
    })

    # 3. Decile incidence (exact 10% mass bins)
    transfer_by_point = (D_arr * transfer_grid).sum(axis=1)
    cons_by_point = (D_arr * dc0).sum(axis=1)
    tot_transfer = W @ transfer_by_point
    tot_cons = W @ cons_by_point
    decile_mpc = np.where(tot_transfer > 1e-15, tot_cons / np.where(tot_transfer > 1e-15, tot_transfer, 1.0), 0.0)
    df_deciles = pd.DataFrame(
        {"Transfer": tot_transfer, "Consumption": tot_cons, "Decile_MPC": decile_mpc},
        index=pd.Index([f"Decile {d}" for d in range(1, 11)], name="Decile"),
    )

    return FiscalTransferResult(
        irf_consumption=dC,
        cumulative_multiplier=cumulative_multiplier,
        impact_mpc=impact_mpc,
        mpc_by_group=mpc_by_group,
        decile_incidence=df_deciles,
        target_group=str(label),
        transfer_amount=float(amount),
        nonlinear=bool(nonlinear),
    )


def solve_hank_sequence_space(
    *,
    T: int = 40,
    beta: float = 0.985,
    gamma: float = 1.0,
    r_ss: float = 0.01,
    phi_pi: float = 1.5,
    kappa: float = 0.1,
    shock_magnitude: float = 0.0025,
    shock_rho: float = 0.7,
    n_a: int = 50,
    a_max: float = 30.0,
) -> SequenceSpaceHANKResult:
    """Solve the HANK steady state, its Fake News Jacobians and the linear GE response.

    Parameters
    ----------
    T : int, default 40
        Horizon for impulse responses and sequence matrices.
    beta : float, default 0.985
        Household discount factor (also the NKPC discount factor).
    gamma : float, default 1.0
        Relative risk aversion (CRRA parameter).
    r_ss : float, default 0.01
        Steady-state quarterly real interest rate (e.g. 1% quarterly = ~4% annual).
    phi_pi : float, default 1.5
        Taylor rule inflation coefficient.
    kappa : float, default 0.1
        New Keynesian Phillips curve slope.
    shock_magnitude : float, default 0.0025
        Monetary policy shock (e.g. 25 bps = 0.0025).
    shock_rho : float, default 0.7
        Persistence of the monetary policy shock.
    n_a : int, default 50
        Number of points on asset grid.
    a_max : float, default 30.0
        Maximum asset limit.

    Returns
    -------
    SequenceSpaceHANKResult
    """
    T = int(T)
    if T < 1:
        raise ValueError("T must be a positive integer")
    hh = _solve_household_block(beta=float(beta), gamma=float(gamma), r_ss=float(r_ss), n_a=int(n_a), a_max=float(a_max))
    D = hh.D_ss
    D_a = D.sum(axis=1)

    # MPCs: marginal quarterly MPC on the grid, aggregate and by exact wealth decile
    mpc_grid = _local_mpc(hh.c_ss, hh.a_grid, hh.r_ss)
    agg_mpc = float(np.sum(mpc_grid * D))
    W = _decile_weights(D_a)
    mpc_by_point = (mpc_grid * D).sum(axis=1)
    decile_mass = W @ D_a
    decile_mpc = (W @ mpc_by_point) / np.where(decile_mass > 0, decile_mass, 1.0)
    mpc_series = pd.Series({f"Decile {d + 1}": float(decile_mpc[d]) for d in range(10)})

    # Household Jacobians (Fake News algorithm) and the linear GE solve
    J_C_r, J_C_Y = _consumption_jacobians(hh, T)
    K_pi, M_r_Y = _ge_matrices(T, float(beta), float(kappa), float(phi_pi))
    shock_seq = float(shock_magnitude) * (float(shock_rho) ** np.arange(T))
    LHS = np.eye(T) - J_C_Y - J_C_r @ M_r_Y
    dY = np.linalg.solve(LHS, J_C_r @ shock_seq)
    dC = dY.copy()
    dpi = K_pi @ dY
    dr = M_r_Y @ dY + shock_seq

    return SequenceSpaceHANKResult(
        irf_output=dY,
        irf_consumption=dC,
        irf_inflation=dpi,
        irf_rate=dr,
        jacobian_c_r=J_C_r,
        jacobian_c_y=J_C_Y,
        steady_state_mpc=agg_mpc,
        mpc_distribution=mpc_series,
        asset_grid=hh.a_grid,
        steady_state_wealth_dist=D_a,
        policy_c=hh.c_ss,
        policy_a=hh.a_ss,
        distribution=D,
        trans_matrix=hh.Lambda,
        beta=float(beta),
        gamma=float(gamma),
        r_ss=float(r_ss),
        phi_pi=float(phi_pi),
        kappa=float(kappa),
        steady_state_consumption=hh.C_ss,
        w_ss=hh.w_ss,
        government_debt=hh.B,
        tax_rate=hh.tax_rate,
        ss_converged=hh.converged,
    )


def _resolve_steady_state(
    ss_model: SequenceSpaceHANKResult | Mapping[str, Any] | None,
    horizon: int,
    kwargs: dict[str, Any],
) -> tuple[SequenceSpaceHANKResult, float, float]:
    """Return (steady state, phi_pi, kappa), re-solving the steady state when overrides require it."""
    ge_over = {k: float(kwargs.pop(k)) for k in list(kwargs) if k in _GE_PARAM_KEYS}
    if ss_model is None:
        params: dict[str, Any] = {"T": min(horizon, 40)}
        params.update(kwargs)
        params.update(ge_over)
        ss = solve_hank_sequence_space(**params)
    elif isinstance(ss_model, SequenceSpaceHANKResult):
        unknown = [k for k in kwargs if k not in _HH_PARAM_KEYS and k not in _SS_PASSTHROUGH_KEYS]
        if unknown:
            raise TypeError(
                f"solve_nonlinear_transition() got unexpected keyword argument(s) {unknown}; with a pre-solved "
                f"ss_model the accepted overrides are {list(_HH_PARAM_KEYS + _GE_PARAM_KEYS + _SS_PASSTHROUGH_KEYS)}"
            )
        current: dict[str, Any] = {
            "beta": float(ss_model.beta), "gamma": float(ss_model.gamma), "r_ss": float(ss_model.r_ss),
            "n_a": int(len(ss_model.asset_grid)), "a_max": float(ss_model.asset_grid[-1]),
        }
        changed = {k: v for k, v in kwargs.items() if k in _HH_PARAM_KEYS and not np.isclose(float(v), float(current[k]))}
        passthrough = {k: v for k, v in kwargs.items() if k in _SS_PASSTHROUGH_KEYS}
        if changed or passthrough:
            params = {"T": len(ss_model.irf_output), "phi_pi": float(ss_model.phi_pi), "kappa": float(ss_model.kappa)}
            params.update(current)
            params.update(changed)
            params.update(passthrough)
            params.update(ge_over)
            ss = solve_hank_sequence_space(**params)
        else:
            ss = ss_model
    elif isinstance(ss_model, Mapping):
        params = dict(ss_model)
        params.update(kwargs)
        params.update(ge_over)
        ss = solve_hank_sequence_space(**params)
    else:
        raise TypeError(
            f"ss_model must be SequenceSpaceHANKResult, Mapping, or None, got {type(ss_model)}"
        )
    phi_pi = ge_over.get("phi_pi", float(ss.phi_pi))
    kappa = ge_over.get("kappa", float(ss.kappa))
    return ss, phi_pi, kappa


def solve_nonlinear_transition(
    ss_model: SequenceSpaceHANKResult | Mapping[str, Any] | None = None,
    shock_seq: Sequence[float] | np.ndarray | None = None,
    shock_var: str = "r",
    horizon: int = 300,
    max_iter: int = 100,
    tol: float = 1e-6,
    backtracking: bool = True,
    **kwargs: Any,
) -> NonlinearHANKResult:
    """Solve Non-Linear General Equilibrium Transition Dynamics for large MIT shocks.

    Implements the sequence-space Broyden Quasi-Newton method of Auclert, Bardóczy,
    Rognlie & Straub (2021, Econometrica):

    1. Evaluates the non-linear household consumption function C(Y, r) over horizon T
       via backward Endogenous Grid Method and forward simulation of the household distribution.
    2. Constructs the market-clearing residual sequence
       ``H_t = Y_t - C_t(Y, r(Y, Z)) - G_t = 0``.
    3. Solves H(U, Z) = 0 via Broyden's Quasi-Newton method with Sherman-Morrison
       rank-1 inverse Jacobian updates:

       - Initial inverse Jacobian ``B_0 = J_ss^{-1}`` with
         ``J_ss = dH/dU = I - J_C_Y - J_C_r @ M_r_Y`` built from the Fake News
         household Jacobians (the same matrix solves the linear path).
       - Iteration step ``Delta U_k = - B_k @ H(U_k)``.
       - Monotone backtracking line search: the step is halved (up to twelve
         times) until the Euclidean residual norm ``||H||_2`` decreases; if no
         contraction is found the inverse Jacobian is reset to ``B_0`` once, and
         the solver stops with a ``RuntimeWarning`` if it still cannot make
         progress.  Convergence is tested in the sup norm ``||H||_inf < tol``.
       - Sherman-Morrison rank-1 update
         ``B_{k+1} = B_k + ((dU - B_k dH) (dU' B_k)) / (dU' B_k dH)``.
       - Termination when ``||H||_inf < tol``.

    Timing: the GE block's real rate ``dr_t = i_t - pi_{t+1}`` is the ex-ante
    return between ``t`` and ``t + 1``; households earn it on assets carried
    into ``t + 1`` (``r_{t+1} = r_ss + dr_t``, ``r_0 = r_ss`` predetermined).
    Government debt is constant at ``B = A_ss`` and its interest ``r_t B`` is
    financed by a proportional labour-income tax, so a rate change has no
    unbacked aggregate income effect (only redistribution and substitution).
    Government spending ``G`` is not tax-financed within the horizon.

    A zero shock returns the steady state exactly (``||H(0)||_inf ~ 1e-12``, zero
    iterations).  Non-convergence within ``max_iter`` is reported with
    ``converged=False`` **and** a ``RuntimeWarning``.

    Parameters
    ----------
    ss_model : SequenceSpaceHANKResult, dict, or None, optional
        Pre-solved steady-state HANK model result or parameters. If None,
        solves steady-state problem automatically.
    shock_seq : Sequence[float] or np.ndarray, optional
        Exogenous MIT shock path. Shorter sequences are zero-padded to ``horizon``;
        a longer one extends the horizon to its length. If None, defaults to a
        100 bps monetary shock ``0.01 * 0.7 ** t``.
    shock_var : {'r', 'G', 'monetary', 'fiscal'}, default 'r'
        Type of shock: 'r' for monetary policy shock, 'G' for fiscal spending shock.
    horizon : int, default 300
        Simulation horizon length T (quarters).
    max_iter : int, default 100
        Maximum number of Broyden iterations.
    tol : float, default 1e-6
        Convergence tolerance on ||H||_inf.
    backtracking : bool, default True
        Whether to perform the monotone line search (``False`` takes full Broyden
        steps and may diverge, which is reported by the warning).
    **kwargs : Any
        Parameter overrides. With ``ss_model=None`` or a dict they go to
        ``solve_hank_sequence_space``. With a pre-solved ``ss_model``, ``phi_pi``
        and ``kappa`` change only the GE block, while ``beta``, ``gamma``,
        ``r_ss``, ``n_a`` and ``a_max`` trigger a re-solve of the steady state
        (the result's ``steady_state_model`` is the re-solved one); any other key
        raises ``TypeError``.

    Returns
    -------
    NonlinearHANKResult
        Structured result containing linear vs non-linear general equilibrium paths,
        residuals, iterations, convergence status, and .plot().
    """
    if not isinstance(shock_var, str):
        raise TypeError(f"shock_var must be a string ('r' or 'G'), got {type(shock_var).__name__}")
    s_var = shock_var.lower().strip()
    is_monetary = s_var in ("r", "monetary", "interest_rate", "rate")
    is_fiscal = s_var in ("g", "fiscal", "spending", "transfer")
    if not is_monetary and not is_fiscal:
        raise ValueError(
            f"Unknown shock_var: {shock_var!r}. Must be 'r' (monetary) or 'G' (fiscal)."
        )
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    tol = float(tol)
    if not tol > 0:
        raise ValueError("tol must be positive")

    # 1. Shock sequence
    if shock_seq is None:
        shock_seq_full = 0.01 * (0.7 ** np.arange(horizon))
    else:
        shock_arr = np.asarray(shock_seq, dtype=float).ravel()
        if not np.all(np.isfinite(shock_arr)):
            raise ValueError("shock_seq must be finite")
        if len(shock_arr) > horizon:
            horizon = len(shock_arr)
            shock_seq_full = shock_arr.copy()
        else:
            shock_seq_full = np.zeros(horizon)
            shock_seq_full[:len(shock_arr)] = shock_arr

    # 2. Steady state (re-solved on conflicting overrides) and household block
    ss, phi_pi, kappa = _resolve_steady_state(ss_model, horizon, dict(kwargs))
    hh = _household_block_from_result(ss)
    beta = hh.beta
    r_ss = hh.r_ss
    w_ss = hh.w_ss
    C_ss = hh.C_ss
    Y_ss = C_ss

    # 3. GE matrices and household Jacobians at the simulation horizon
    K_pi, M_r_Y = _ge_matrices(horizon, beta, kappa, phi_pi)
    if ss.jacobian_c_r.shape[0] >= horizon and ss.jacobian_c_y.shape[0] >= horizon:
        J_C_r = np.ascontiguousarray(ss.jacobian_c_r[:horizon, :horizon])
        J_C_Y = np.ascontiguousarray(ss.jacobian_c_y[:horizon, :horizon])
    else:
        J_C_r, J_C_Y = _consumption_jacobians(hh, horizon)
    J_ss = np.eye(horizon) - J_C_Y - J_C_r @ M_r_Y
    B0 = np.linalg.inv(J_ss)

    # 4. Linear sequence-space solution (first-order limit of the non-linear path)
    if is_monetary:
        dY_linear = np.linalg.solve(J_ss, J_C_r @ shock_seq_full)
        dC_linear = dY_linear.copy()
        dr_linear = M_r_Y @ dY_linear + shock_seq_full
    else:
        dY_linear = np.linalg.solve(J_ss, shock_seq_full)
        dC_linear = dY_linear - shock_seq_full
        dr_linear = M_r_Y @ dY_linear
    dpi_linear = K_pi @ dY_linear

    # 5. Non-linear household block along the path
    shock_r = shock_seq_full if is_monetary else np.zeros(horizon)
    shock_G = shock_seq_full if is_fiscal else np.zeros(horizon)

    def compute_C(dY: np.ndarray) -> np.ndarray:
        dr = M_r_Y @ dY + shock_r
        rr_seq = np.concatenate(([r_ss], r_ss + dr))          # realised returns r_0..r_T
        w_seq = np.maximum(w_ss * (1.0 + dY / Y_ss), 1e-6)
        C, _, _ = _household_transition(hh, rr_seq, w_seq)
        return C

    def H_func(dY: np.ndarray) -> np.ndarray:
        return dY - (compute_C(dY) - C_ss) - shock_G

    # 6. Broyden solver with monotone backtracking and Sherman-Morrison updates
    U = np.zeros(horizon)
    H_val = H_func(U)
    norm = float(np.max(np.abs(H_val)))
    merit = float(np.linalg.norm(H_val))
    norm_history: list[float] = [norm]
    B = B0.copy()
    iterations = 0
    stall_reason = ""
    reset_done = False
    while norm >= tol and iterations < max_iter:
        dU = -B @ H_val
        accepted = False
        if backtracking:
            step = 1.0
            for _ in range(12):
                U_try = U + step * dU
                H_try = H_func(U_try)
                m_try = float(np.linalg.norm(H_try)) if np.all(np.isfinite(H_try)) else np.inf
                if m_try <= (1.0 - 1e-4 * step) * merit:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                if not reset_done:
                    B = B0.copy()
                    reset_done = True
                    continue
                stall_reason = "the line search found no norm-decreasing step even from the steady-state Jacobian"
                break
        else:
            U_try = U + dU
            H_try = H_func(U_try)
            if not np.all(np.isfinite(H_try)):
                stall_reason = "the Broyden step produced non-finite residuals (backtracking=False)"
                break
            m_try = float(np.linalg.norm(H_try))
        delta_U = U_try - U
        delta_H = H_try - H_val
        u_vec = delta_U - B @ delta_H
        v_vec = delta_U @ B
        denom = float(np.dot(v_vec, delta_H))
        if abs(denom) > 1e-14:
            B = B + np.outer(u_vec, v_vec) / denom
        U, H_val, merit = U_try, H_try, m_try
        norm = float(np.max(np.abs(H_val)))
        norm_history.append(norm)
        iterations += 1
        if accepted:
            reset_done = False

    converged = bool(norm < tol)
    if not converged:
        reason = stall_reason or f"max_iter={max_iter} reached"
        warnings.warn(
            f"solve_nonlinear_transition did not converge: ||H||_inf = {norm:.3e} >= tol = {tol:.1e} after "
            f"{iterations} Broyden iterations ({reason}). The returned paths do not clear the goods market; "
            f"raise max_iter, shorten the horizon or reduce the shock.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 7. Non-linear paths
    dY_nonlinear = U.copy()
    dC_nonlinear = compute_C(dY_nonlinear) - C_ss
    dpi_nonlinear = K_pi @ dY_nonlinear
    dr_nonlinear = M_r_Y @ dY_nonlinear + shock_r

    return NonlinearHANKResult(
        U=U,
        residuals=H_val,
        iterations=iterations,
        converged=converged,
        linear_path=dY_linear,
        nonlinear_path=dY_nonlinear,
        norm_history=norm_history,
        irf_output_linear=dY_linear,
        irf_output_nonlinear=dY_nonlinear,
        irf_consumption_linear=dC_linear,
        irf_consumption_nonlinear=dC_nonlinear,
        irf_rate_linear=dr_linear,
        irf_rate_nonlinear=dr_nonlinear,
        irf_inflation_linear=dpi_linear,
        irf_inflation_nonlinear=dpi_nonlinear,
        shock_var=s_var,
        shock_seq=shock_seq_full,
        horizon=horizon,
        steady_state_model=ss,
        tol=tol,
        jacobian_c_r=J_C_r,
        jacobian_c_y=J_C_Y,
    )


__all__ = [
    "SequenceSpaceHANKResult",
    "FakeNewsResult",
    "FiscalTransferResult",
    "NonlinearHANKResult",
    "solve_hank_sequence_space",
    "fake_news_algorithm",
    "simulate_targeted_transfer",
    "solve_nonlinear_transition",
]
