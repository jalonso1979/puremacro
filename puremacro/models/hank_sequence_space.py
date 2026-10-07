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

import numpy as np
import pandas as pd

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
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

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
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

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
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

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




# ---------------------------------------------------------------------------
# Two-Asset HANK Sequence-Space Engine (Kaplan, Moll & Violante 2018; Auclert et al. 2021)
# ---------------------------------------------------------------------------
#
# Household problem (quarterly).  A household enters date t with liquid assets
# b, illiquid assets a and productivity s, and solves
#
#     V_t(s, b, a) = max  u(c) + beta E[V_{t+1}(s', b', a') | s]
#     c + b' + a' = (1 + r^b_t) b + (1 + r^a_t) a + z_t s - chi(d, a),
#     d = a' - (1 + r^a_t) a,     b' >= b_min,     a' >= 0,
#
# with CRRA utility u(c) = c^(1-gamma)/(1-gamma) (log for gamma = 1) and the
# convex, kink-free portfolio adjustment cost (KMV 2018 adjustment-cost
# structure without the linear term, so that policies are differentiable)
#
#     chi(d, a) = chi_0 / (1 + chi_1) * |d|^(1 + chi_1) / (a + a_bar)^chi_1.
#
# r^b_t and r^a_t are the returns *realised* at t on assets carried into t
# (the convention of the one-asset block above).  The backward iteration is
# the endogenous-grid method on the two marginal values V_a and V_b:
# for every (a, b', s) the first-order condition for a',
# W_a(a', b', s) = W_b(a', b', s) (1 + chi_d(a', a)), with W_x = beta E[V_x'],
# is solved for a' (safeguarded Newton inside the bracketing grid interval),
# the Euler equation u'(c) = W_b gives c, the budget constraint gives the
# endogenous b, and linear interpolation maps it back to the b grid.  States
# for which b' = b_min binds solve u'(c) (1 + chi_d) = W_a(a', b_min, s) for a'
# directly.  Envelope conditions: V_b = (1 + r^b) u'(c) and
# V_a = (1 + r^a - chi_a(a', a)) u'(c), with chi_a the derivative of the cost
# with respect to a at fixed a'.  Policies are kept inside the grids and c
# comes from the budget constraint, so the bilinear lottery is exactly
# mean-preserving and no wealth is created or destroyed at the grid edges.
#
# Closure.  The returns on both assets are paid out of aggregate income, on the
# holdings households actually carry into t, so household non-financial
# income per efficiency unit is
#
#     z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1},        N = E[s],
#
# with B_{t-1} = sum D b', A_{t-1} = sum D a' chosen at t-1 (B_{-1}, A_{-1}
# at the steady state).  Aggregate household income is then exactly Y_t:
# rate changes redistribute income between asset holders and workers, and the
# supply of each asset accommodates the portfolio split households choose.
# Adjustment costs are a resource cost, so the goods market clears as
# Y_t = C_t + CHI_t, and the aggregate household budget
# C_t + CHI_t + W_t - W_{t-1} = Y_t (W = A + B) then gives W_t = W_{-1}: total
# wealth stays at A_ss + B_ss (Walras's law) while its composition moves.  This
# is the two-asset analogue of the balanced-budget labour tax of the one-asset
# block, where Walras's law pins the single asset.  (Paying the returns on the
# fixed steady-state stocks instead, z_t N = Y_t - r^b_t B - r^a_t A, leaves
# household income above output by r_b dB_{t-1} + r_a dA_{t-1}, and total wealth
# then follows dW_t = (1 + r_b) dW_{t-1} + (r_a - r_b) dA_{t-1}, an explosive
# root that contaminates long-horizon IRFs.)
#
# GE block (identical to the one-asset block): NKPC pi = K_pi dY, Taylor rule
# i_t = phi_pi pi_t + eps_t, ex-ante liquid real rate r^{b,ea}_t = i_t - pi_{t+1}
# (= M_r_Y dY + eps), ex-ante illiquid rate r^{a,ea}_t = r^a_ss +
# alpha_ab (r^{b,ea}_t - r^b_ss).  Ex-ante rates are realised one period later,
# r^b_{t+1} = r^{b,ea}_t, and date-0 returns are predetermined.  The Jacobians
# of _two_asset_jacobians and TwoAssetSequenceSpaceHANKResult are those of the
# household sector *closed* by the income rule above (inputs: Y and the ex-ante
# rates; z is solved out), dated by the ex-ante convention;
# _two_asset_household_jacobians returns the raw household-block Jacobians
# with respect to the realised inputs (r^b_t, r^a_t, z_t), without closure.
#
# Supported range: the adjustment-cost curvature chi_1 >= 1 (chi_1 = 1 is
# quadratic).  For chi_1 < 1 the marginal cost |d|^chi_1 has an infinite slope
# at d = 0 and the EGM iteration cycles instead of converging.

_TA_EGM_TOL = 1e-11
_TA_EGM_MAX_ITER = 10000
# The EGM iteration is stopped early (converged=False) when the best residual of the last
# _TA_EGM_STALL_WINDOW iterations is not below _TA_EGM_STALL_FACTOR times the best residual before
# them (checked from iteration 2 * window on).  Every validated calibration converges in < 1000
# iterations; a contraction slower than 0.5**(1/500) per iteration would need > 17000 iterations.
_TA_EGM_STALL_WINDOW = 500
_TA_EGM_STALL_FACTOR = 0.5
_TA_C_FLOOR = 1e-12
# Central-difference steps of the Fake-News Algorithm, per input.  Rates are ~1e-2 per quarter, so a
# 1e-4 step would be 1-2% of the rate and cross kinks of the linear interpolants; 1e-5 gives the
# local derivative (the direct method at 1e-5 and 1e-6 agrees to ~1e-8).
_TA_FD_STEP = {"r_b": 1e-5, "r_a": 1e-5, "z": 1e-4}
_TA_GRID_TOP_TOL = 1e-8
_TA_INPUTS = ("r_b", "r_a", "z")
_TA_OUTPUTS = ("C", "D", "A", "B", "CHI")
_TA_INPUT_ALIASES = {
    "r_b": "r_b", "rb": "r_b", "r": "r_b",
    "r_a": "r_a", "ra": "r_a",
    "z": "z", "w": "z", "y": "z",
}


def _transaction_cost(
    d: np.ndarray | float,
    a: np.ndarray | float,
    chi_0: float = 0.25,
    chi_1: float = 1.0,
    a_bar: float | None = None,
) -> np.ndarray | float:
    """Portfolio adjustment cost chi(d, a) of a deposit d into the illiquid account.

    ``d = a' - (1 + r_a) a`` is the net deposit (negative for a withdrawal) and
    ``a`` the illiquid holding at the start of the period, both in units of
    goods (steady-state non-financial income per efficiency unit is 1)::

        chi(d, a) = chi_0 / (1 + chi_1) * |d|^(1 + chi_1) / (a + a_bar)^chi_1

    which for ``chi_1 == 1`` is the quadratic cost ``0.5 chi_0 d^2 / (a + a_bar)``.
    This is the Kaplan, Moll & Violante (2018) cost structure without the linear
    ``|d|`` term, so that policies are differentiable (a requirement of the
    Fake-News Jacobians).  ``a_bar > 0`` keeps the cost of depositing into an
    empty account finite.  With ``a_bar=None`` (the legacy formula, kept for
    backward compatibility; the solver always passes ``a_bar``) the denominator
    is ``max(a, 1e-4)``, which makes depositing into an empty account
    prohibitively expensive.
    """
    abs_d = np.abs(d)
    if a_bar is None:
        denom = np.maximum(a, 1e-4)
    else:
        denom = np.asarray(a, dtype=float) + float(a_bar)
    if chi_1 == 1.0:
        return 0.5 * chi_0 * (abs_d ** 2) / denom
    return (chi_0 / abs(1.0 + chi_1)) * ((abs_d / denom) ** (1.0 + chi_1)) * denom


def _two_asset_cost(
    a_next: np.ndarray,
    a: np.ndarray,
    r_a: float,
    chi_0: float,
    chi_1: float,
    a_bar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adjustment cost and derivatives as a function of (a', a).

    Returns ``(chi, chi_d, chi_a, chi_dd)``: the cost of moving from ``a`` to
    ``a'`` (``d = a' - (1 + r_a) a``), its derivative with respect to ``a'``,
    its derivative with respect to ``a`` at fixed ``a'``
    (``-(1 + r_a) chi_d - chi_1 chi / (a + a_bar)``) and the second derivative
    with respect to ``a'``.
    """
    d = a_next - (1.0 + r_a) * a
    den = a + a_bar
    abs_d = np.abs(d)
    ratio = abs_d / den
    core = ratio ** chi_1
    chi = chi_0 / (1.0 + chi_1) * abs_d * core
    chi_d = chi_0 * np.sign(d) * core
    chi_a = -(1.0 + r_a) * chi_d - chi_1 * chi / den
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_dd = chi_0 * chi_1 * ratio ** (chi_1 - 1.0) / den
    return chi, chi_d, chi_a, chi_dd


def _two_asset_grid(lo: float, hi: float, n: int, curvature: float = 2.0) -> np.ndarray:
    """Grid on [lo, hi] with points concentrated near ``lo`` (x_i = lo + (hi - lo) (i/(n-1))^curvature)."""
    x = np.linspace(0.0, 1.0, int(n))
    return lo + (hi - lo) * x ** curvature


def _lottery_2d(
    a_dest: np.ndarray,
    b_dest: np.ndarray,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D bilinear lottery: split (a_dest, b_dest) among the 4 neighbouring grid points.

    Mean-preserving for destinations inside the grids; a destination outside a
    grid is clipped to its edge (the solver keeps every policy inside the grids).
    """
    na = len(a_grid)
    idx_a = np.clip(np.searchsorted(a_grid, a_dest), 0, na - 1)
    idx_al = np.clip(idx_a - 1, 0, na - 1)
    span_a = np.maximum(a_grid[idx_a] - a_grid[idx_al], 1e-6)
    w_ah = np.clip((a_dest - a_grid[idx_al]) / span_a, 0.0, 1.0)
    w_al = 1.0 - w_ah

    nb = len(b_grid)
    idx_b = np.clip(np.searchsorted(b_grid, b_dest), 0, nb - 1)
    idx_bl = np.clip(idx_b - 1, 0, nb - 1)
    span_b = np.maximum(b_grid[idx_b] - b_grid[idx_bl], 1e-6)
    w_bh = np.clip((b_dest - b_grid[idx_bl]) / span_b, 0.0, 1.0)
    w_bl = 1.0 - w_bh

    return idx_al, idx_a, w_al, w_ah, idx_bl, idx_b, w_bl, w_bh


def _build_transition_matrix_2d(
    a_pol: np.ndarray,
    b_pol: np.ndarray,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    pi_s: np.ndarray,
) -> np.ndarray:
    """Dense (N, N) column-stochastic transition matrix Lambda with state index a_i * (n_b * n_s) + b_i * n_s + s_i."""
    na, nb, ns = a_pol.shape
    N = na * nb * ns
    idx_al, idx_ah, w_al, w_ah, idx_bl, idx_bh, w_bl, w_bh = _lottery_2d(a_pol, b_pol, a_grid, b_grid)
    cols = (
        np.arange(na)[:, None, None] * (nb * ns)
        + np.arange(nb)[None, :, None] * ns
        + np.arange(ns)[None, None, :]
    )
    Lam = np.zeros((N, N))
    for s_next in range(ns):
        p = pi_s[:, s_next][None, None, :]
        row_ll = idx_al * (nb * ns) + idx_bl * ns + s_next
        row_lh = idx_al * (nb * ns) + idx_bh * ns + s_next
        row_hl = idx_ah * (nb * ns) + idx_bl * ns + s_next
        row_hh = idx_ah * (nb * ns) + idx_bh * ns + s_next

        np.add.at(Lam, (row_ll.ravel(), cols.ravel()), (w_al * w_bl * p).ravel())
        np.add.at(Lam, (row_lh.ravel(), cols.ravel()), (w_al * w_bh * p).ravel())
        np.add.at(Lam, (row_hl.ravel(), cols.ravel()), (w_ah * w_bl * p).ravel())
        np.add.at(Lam, (row_hh.ravel(), cols.ravel()), (w_ah * w_bh * p).ravel())
    return Lam


def _stationary_distribution_2d(Lam: np.ndarray) -> np.ndarray:
    """Exact stationary distribution of a column-stochastic 2D transition matrix Lambda.

    Linear solve with the normalisation replacing one equation; falls back to
    power iteration (tolerance 1e-13, warning if not reached) when the solve is
    singular or its residual ``max |Lambda D - D|`` exceeds 1e-10.
    """
    N = Lam.shape[0]
    A = Lam - np.eye(N)
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    try:
        D = np.linalg.solve(A, b)
        if (
            np.all(np.isfinite(D))
            and D.min() > -1e-9
            and float(np.max(np.abs(Lam @ D - D))) < 1e-10
        ):
            D = np.maximum(D, 0.0)
            return D / D.sum()
    except np.linalg.LinAlgError:
        pass
    D = np.full(N, 1.0 / N)
    converged = False
    for _ in range(200000):
        D_new = Lam @ D
        if np.max(np.abs(D_new - D)) < 1e-13:
            D = D_new
            converged = True
            break
        D = D_new
    if not converged:
        warnings.warn(
            "two-asset stationary distribution: power iteration did not reach 1e-13",
            RuntimeWarning,
            stacklevel=3,
        )
    D = np.maximum(D, 0.0)
    return D / D.sum()


def _two_asset_forward(
    D: np.ndarray, a_pol: np.ndarray, b_pol: np.ndarray, a_grid: np.ndarray, b_grid: np.ndarray, pi_s: np.ndarray,
) -> np.ndarray:
    """D_{t+1} = Lambda(a_pol, b_pol) D_t without forming Lambda (same lottery as _build_transition_matrix_2d)."""
    na, nb, ns = D.shape
    idx_al, idx_ah, w_al, w_ah, idx_bl, idx_bh, w_bl, w_bh = _lottery_2d(a_pol, b_pol, a_grid, b_grid)
    D_end = np.zeros_like(D)
    for s in range(ns):
        mass = D[:, :, s].ravel()
        tot = np.zeros(na * nb)
        for ia, wa in ((idx_al, w_al), (idx_ah, w_ah)):
            for ib, wb in ((idx_bl, w_bl), (idx_bh, w_bh)):
                tot += np.bincount(
                    (ia[:, :, s] * nb + ib[:, :, s]).ravel(),
                    weights=mass * (wa[:, :, s] * wb[:, :, s]).ravel(),
                    minlength=na * nb,
                )
        D_end[:, :, s] = tot.reshape(na, nb)
    return D_end @ pi_s


def _two_asset_expectation(
    E: np.ndarray, a_pol: np.ndarray, b_pol: np.ndarray, a_grid: np.ndarray, b_grid: np.ndarray, pi_s: np.ndarray,
) -> np.ndarray:
    """Lambda' E: expected next-period value of E for every current state (transpose of _two_asset_forward)."""
    idx_al, idx_ah, w_al, w_ah, idx_bl, idx_bh, w_bl, w_bh = _lottery_2d(a_pol, b_pol, a_grid, b_grid)
    EV = E @ pi_s.T
    s_idx = np.arange(E.shape[2])[None, None, :]
    return (
        w_al * w_bl * EV[idx_al, idx_bl, s_idx]
        + w_al * w_bh * EV[idx_al, idx_bh, s_idx]
        + w_ah * w_bl * EV[idx_ah, idx_bl, s_idx]
        + w_ah * w_bh * EV[idx_ah, idx_bh, s_idx]
    )


@dataclass
class _TwoAssetHouseholdBlock:
    """Steady-state two-asset household block (arrays indexed [a, b, s]).

    ``w_ss`` is steady-state non-financial income per efficiency unit (``z``);
    ``a_ss``/``b_ss`` are the next-period illiquid/liquid policies, ``d_ss`` the
    net deposit ``a' - (1 + r_a) a`` and ``chi_ss`` the adjustment cost paid.
    ``Va_ss``/``Vb_ss`` are the marginal values of illiquid and liquid assets.
    ``iterations``/``residual`` report the EGM fixed point
    (sup-norm change of c, a', b' in the last iteration); ``converged`` is True
    only if that residual fell below ``tol`` and the stationary distribution
    was solved exactly.
    """
    a_grid: np.ndarray
    b_grid: np.ndarray
    s_grid: np.ndarray
    pi_s: np.ndarray
    beta: float
    gamma: float
    r_b_ss: float
    r_a_ss: float
    w_ss: float
    chi_0: float
    chi_1: float
    a_bar: float
    Va_ss: np.ndarray
    Vb_ss: np.ndarray
    c_ss: np.ndarray
    d_ss: np.ndarray
    a_ss: np.ndarray
    b_ss: np.ndarray
    chi_ss: np.ndarray
    D_ss: np.ndarray
    Lambda: np.ndarray
    converged: bool
    iterations: int = 0
    residual: float = float("nan")
    tol: float = _TA_EGM_TOL

    @property
    def C_ss(self) -> float:
        return float(np.sum(self.D_ss * self.c_ss))

    @property
    def CHI_ss(self) -> float:
        """Aggregate adjustment cost (a resource cost)."""
        return float(np.sum(self.D_ss * self.chi_ss))

    @property
    def Y_ss(self) -> float:
        """Steady-state output Y = C + CHI (= w_ss N + r_b B + r_a A by the aggregate budget)."""
        return self.C_ss + self.CHI_ss

    @property
    def N_ss(self) -> float:
        """Aggregate efficiency units E[s]."""
        return float(np.sum(self.D_ss.sum(axis=(0, 1)) * self.s_grid))

    @property
    def D_flow_ss(self) -> float:
        return float(np.sum(self.D_ss * self.d_ss))

    @property
    def A_ss(self) -> float:
        return float(np.sum(self.D_ss.sum(axis=(1, 2)) * self.a_grid))

    @property
    def B_ss(self) -> float:
        return float(np.sum(self.D_ss.sum(axis=(0, 2)) * self.b_grid))

    @property
    def marginal_distribution_a(self) -> np.ndarray:
        return self.D_ss.sum(axis=(1, 2))

    @property
    def marginal_distribution_b(self) -> np.ndarray:
        return self.D_ss.sum(axis=(0, 2))

    @property
    def joint_distribution(self) -> np.ndarray:
        return self.D_ss.sum(axis=2)

    @property
    def deposit_distribution(self) -> np.ndarray:
        return np.sum(self.D_ss * self.d_ss, axis=(1, 2))

    @property
    def mass_at_a_max(self) -> float:
        """Mass on the top illiquid grid point (should be ~0: the grid must contain the ergodic set)."""
        return float(self.marginal_distribution_a[-1])

    @property
    def mass_at_b_max(self) -> float:
        """Mass on the top liquid grid point (should be ~0)."""
        return float(self.marginal_distribution_b[-1])

    @property
    def htm_share(self) -> float:
        """Share of households choosing b' = b_min (liquidity-constrained, 'hand-to-mouth')."""
        return float(self.D_ss[self.b_ss <= self.b_grid[0] + 1e-12].sum())

    @property
    def wealthy_htm_share(self) -> float:
        """Share of households with b' = b_min but positive illiquid wealth a' > 0."""
        mask = (self.b_ss <= self.b_grid[0] + 1e-12) & (self.a_ss > 1e-10)
        return float(self.D_ss[mask].sum())

    @property
    def budget_residual(self) -> float:
        """C + CHI - (w N + r_b B + r_a A): zero up to rounding when no wealth leaks at the grid edges."""
        return self.Y_ss - (self.w_ss * self.N_ss + self.r_b_ss * self.B_ss + self.r_a_ss * self.A_ss)

    def mpc(self, h: float = 1e-5) -> np.ndarray:
        """Quarterly MPC out of a one-off lump-sum transfer, per state (central difference of size h)."""
        up = _two_asset_backward(self, self.Va_ss, self.Vb_ss, transfer=h)[2]
        dn = _two_asset_backward(self, self.Va_ss, self.Vb_ss, transfer=-h)[2]
        return (up - dn) / (2.0 * h)


def _two_asset_refine_root(fun, lo, hi, g_lo, g_hi, max_iter: int = 60, rtol: float = 1e-15):
    """Vectorised safeguarded Newton for a decreasing g with g(lo) >= 0 > g(hi).

    ``fun(x)`` returns ``(g, dg)``; a Newton step that leaves the current
    bracket (or is not finite) is replaced by bisection.
    """
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = lo + (hi - lo) * np.clip(g_lo / (g_lo - g_hi), 0.0, 1.0)
    x = np.where(np.isfinite(x), x, 0.5 * (lo + hi))
    for _ in range(max_iter):
        g, dg = fun(x)
        pos = g >= 0
        lo = np.where(pos, x, lo)
        hi = np.where(pos, hi, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_new = x - g / dg
        bad = ~np.isfinite(x_new) | (x_new < lo) | (x_new > hi)
        x_new = np.where(bad, 0.5 * (lo + hi), x_new)
        step = np.abs(x_new - x)
        x = x_new
        if np.all(step <= rtol * (1.0 + np.abs(x))):
            break
    return x


def _two_asset_step(
    Va_next: np.ndarray,
    Vb_next: np.ndarray,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    s_grid: np.ndarray,
    pi_s: np.ndarray,
    beta: float,
    gamma: float,
    r_b: float,
    r_a: float,
    z: float,
    chi_0: float,
    chi_1: float,
    a_bar: float,
    transfer: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One backward EGM step of the two-asset household block.

    ``Va_next``/``Vb_next`` (shape ``(n_a, n_b, n_s)``) are the marginal values
    of illiquid and liquid assets carried into ``t + 1``.  ``r_b``, ``r_a`` are
    the returns realised at ``t``, ``z`` non-financial income per efficiency
    unit and ``transfer`` a lump-sum transfer at ``t``.  Returns
    ``(Va_t, Vb_t, c_t, a'_t, b'_t, d_t, chi_t)`` on the ``(a, b, s)`` grid.
    """
    na, nb, ns = len(a_grid), len(b_grid), len(s_grid)
    a = a_grid
    Wa = beta * (Va_next @ pi_s.T)  # W[a', b', s] = beta E[V(a', b', s') | s]
    Wb = beta * (Vb_next @ pi_s.T)
    # Cost on the (a' = a_j, a = a_i) node pairs
    chi_n, chid_n, _, _ = _two_asset_cost(a[:, None], a[None, :], r_a, chi_0, chi_1, a_bar)

    ia = np.arange(na)[:, None, None]
    ib = np.arange(nb)[None, :, None]
    is_ = np.arange(ns)[None, None, :]
    income = z * s_grid[None, None, :] + transfer

    # --- 1. unconstrained a'(a, b', s): W_a(a') = W_b(a') (1 + chi_d(a', a))
    G = Wa[:, None, :, :] - Wb[:, None, :, :] * (1.0 + chid_n[:, :, None, None])  # [j, a, b', s]
    neg = G < 0
    any_neg = neg.any(axis=0)
    first = np.argmax(neg, axis=0)
    corner_lo = any_neg & (first == 0)
    corner_hi = ~any_neg
    j1 = np.where(any_neg & (first > 0), first, 1)
    j0 = j1 - 1
    x0 = a[j0]
    dx = a[j1] - x0
    Wa0, Wa1 = Wa[j0, ib, is_], Wa[j1, ib, is_]
    Wb0, Wb1 = Wb[j0, ib, is_], Wb[j1, ib, is_]
    sWa = (Wa1 - Wa0) / dx
    sWb = (Wb1 - Wb0) / dx
    a_now = np.broadcast_to(a[:, None, None], (na, nb, ns))

    def g_unc(x):
        Wa_x = Wa0 + sWa * (x - x0)
        Wb_x = Wb0 + sWb * (x - x0)
        _, cd, _, cdd = _two_asset_cost(x, a_now, r_a, chi_0, chi_1, a_bar)
        return Wa_x - Wb_x * (1.0 + cd), sWa - sWb * (1.0 + cd) - Wb_x * cdd

    ap_endo = _two_asset_refine_root(g_unc, x0, a[j1], G[j0, ia, ib, is_], G[j1, ia, ib, is_])
    ap_endo = np.where(corner_lo, a[0], np.where(corner_hi, a[-1], ap_endo))
    Wb_at = np.where(
        corner_lo, Wb[0, ib, is_],
        np.where(corner_hi, Wb[-1, ib, is_], Wb0 + sWb * (np.clip(ap_endo, x0, x0 + dx) - x0)),
    )
    c_endo = Wb_at ** (-1.0 / gamma)
    chi_endo = _two_asset_cost(ap_endo, a_now, r_a, chi_0, chi_1, a_bar)[0]
    # endogenous current liquid holdings b(a, b', s) from the budget constraint
    b_endo = (c_endo + ap_endo + b_grid[None, :, None] + chi_endo - (1.0 + r_a) * a_now - income) / (1.0 + r_b)

    # --- 2. invert b_endo(a, ., s) onto the b grid (linear, extrapolating above the top point)
    be = np.moveaxis(b_endo, 1, 2)  # [a, s, b']
    ae = np.moveaxis(ap_endo, 1, 2)
    k = np.clip((be[:, :, None, :] <= b_grid[None, None, :, None]).sum(axis=-1) - 1, 0, nb - 2)  # [a, s, b]
    xb0 = np.take_along_axis(be, k, axis=-1)
    xb1 = np.take_along_axis(be, k + 1, axis=-1)
    span = xb1 - xb0  # b_endo is strictly increasing in b' (Euler equation); guard a degenerate interval
    w = (b_grid[None, None, :] - xb0) / np.where(np.abs(span) > 1e-300, span, 1e-300)
    bp = np.moveaxis(b_grid[k] + w * (b_grid[k + 1] - b_grid[k]), 2, 1)
    ap = np.moveaxis(
        np.take_along_axis(ae, k, axis=-1) * (1.0 - w) + np.take_along_axis(ae, k + 1, axis=-1) * w, 2, 1,
    )
    cash = (1.0 + r_b) * b_grid[None, :, None] + (1.0 + r_a) * a[:, None, None] + income

    # --- 3. liquidity-constrained states (b' = b_min): u'(c(a')) (1 + chi_d) = W_a(a', b_min, s)
    constrained = b_grid[None, :, None] < b_endo[:, 0:1, :]
    if constrained.any():
        ic, kc, sc = np.nonzero(constrained)
        cash_c = cash[ic, kc, sc] - b_grid[0]
        c_nodes = cash_c[None, :] - a[:, None] - chi_n[:, ic]
        slope_n = 1.0 + chid_n[:, ic]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            H = Wa[:, 0, sc] - np.maximum(c_nodes, 1e-300) ** (-gamma) * slope_n
        H = np.where(c_nodes > 0, H, np.where(slope_n > 0, -np.inf, np.inf))
        negc = H < 0
        any_c = negc.any(axis=0)
        fc = np.argmax(negc, axis=0)
        c_lo = any_c & (fc == 0)
        c_hi = ~any_c
        m1 = np.where(any_c & (fc > 0), fc, 1)
        m0 = m1 - 1
        x0c = a[m0]
        Wa0c = Wa[m0, 0, sc]
        sWac = (Wa[m1, 0, sc] - Wa0c) / (a[m1] - x0c)
        a_c = a[ic]

        def h_con(x):
            chi_x, cd, _, cdd = _two_asset_cost(x, a_c, r_a, chi_0, chi_1, a_bar)
            c = cash_c - x - chi_x
            cp = np.maximum(c, 1e-300)
            with np.errstate(over="ignore", invalid="ignore"):
                u1 = cp ** (-gamma)
                h = Wa0c + sWac * (x - x0c) - u1 * (1.0 + cd)
                dh = sWac - (gamma * cp ** (-gamma - 1.0) * (1.0 + cd) ** 2 + u1 * cdd)
            h = np.where(c > 0, h, np.where(1.0 + cd > 0, -np.inf, np.inf))
            return h, np.where(c > 0, dh, np.nan)

        cols = np.arange(len(ic))
        h_lo = H[m0, cols]
        h_hi = H[m1, cols]
        h_lo = np.where(np.isfinite(h_lo), h_lo, 1.0)
        h_hi = np.where(np.isfinite(h_hi), h_hi, -1.0)
        ap_c = _two_asset_refine_root(h_con, x0c, a[m1], h_lo, h_hi)
        ap[ic, kc, sc] = np.where(c_lo, a[0], np.where(c_hi, a[-1], ap_c))
        bp[ic, kc, sc] = b_grid[0]

    # --- 4. keep policies on the grids and read consumption off the budget constraint
    # (the floor _TA_C_FLOOR only guards the power below; callers detect a binding floor with
    # _two_asset_floored_states, because a floored c violates the budget constraint)
    ap = np.clip(ap, a[0], a[-1])
    bp = np.clip(bp, b_grid[0], b_grid[-1])
    chi, _, chi_a, _ = _two_asset_cost(ap, a[:, None, None], r_a, chi_0, chi_1, a_bar)
    c = np.maximum(cash - ap - bp - chi, _TA_C_FLOOR)
    uc = c ** (-gamma)
    Vb = (1.0 + r_b) * uc
    Va = (1.0 + r_a - chi_a) * uc
    d = ap - (1.0 + r_a) * a[:, None, None]
    return Va, Vb, c, ap, bp, d, chi


def _two_asset_backward(
    hh: "_TwoAssetHouseholdBlock",
    Va_next: np.ndarray,
    Vb_next: np.ndarray,
    r_b: float | None = None,
    r_a: float | None = None,
    z: float | None = None,
    transfer: float = 0.0,
):
    """_two_asset_step at the household block's parameters (inputs default to their steady-state values)."""
    return _two_asset_step(
        Va_next, Vb_next, hh.a_grid, hh.b_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma,
        hh.r_b_ss if r_b is None else float(r_b),
        hh.r_a_ss if r_a is None else float(r_a),
        hh.w_ss if z is None else float(z),
        hh.chi_0, hh.chi_1, hh.a_bar, float(transfer),
    )


def _two_asset_outputs(step: tuple) -> dict[str, np.ndarray]:
    _, _, c, ap, bp, d, chi = step
    return {"C": c, "D": d, "A": ap, "B": bp, "CHI": chi}


def _two_asset_floored_states(
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    s_grid: np.ndarray,
    r_b: float,
    r_a: float,
    z: float,
    ap: np.ndarray,
    bp: np.ndarray,
    chi: np.ndarray,
    transfer: float = 0.0,
) -> int:
    """Number of states where the consumption floor of _two_asset_step binds.

    There ``c = _TA_C_FLOOR > cash - a' - b' - chi``, so the budget constraint
    fails; a correct solution has none.
    """
    cash = (
        (1.0 + r_b) * b_grid[None, :, None]
        + (1.0 + r_a) * a_grid[:, None, None]
        + z * s_grid[None, None, :]
        + transfer
    )
    return int(np.count_nonzero(cash - ap - bp - chi < _TA_C_FLOOR))


def _solve_two_asset_household_block(
    *,
    beta: float = 0.98,
    gamma: float = 1.0,
    r_b_ss: float = 0.005,
    r_a_ss: float = 0.0125,
    w_ss: float = 1.0,
    n_a: int = 25,
    n_b: int = 25,
    a_max: float = 40.0,
    b_max: float = 10.0,
    b_min: float = 0.0,
    chi_0: float = 1.0,
    chi_1: float = 1.0,
    max_iter: int = _TA_EGM_MAX_ITER,
    tol: float = _TA_EGM_TOL,
    a_bar: float = 0.25,
) -> _TwoAssetHouseholdBlock:
    """Solve the two-asset stationary household problem and its stationary distribution.

    Iterates the EGM step on ``(V_a, V_b)`` until the sup-norm change of the
    policies ``c``, ``a'`` and ``b'`` falls below ``tol``, then solves for the
    exact stationary distribution of the lottery transition matrix.
    ``converged`` is False, with a RuntimeWarning, if ``max_iter`` is reached
    first, if the residual stalls (no halving of the best residual within
    ``_TA_EGM_STALL_WINDOW`` iterations; the iteration then stops early) or if
    the consumption floor binds anywhere (the budget constraint would fail).
    Returns are quarterly; income is in units of steady-state non-financial
    income per efficiency unit (``w_ss``).  Warns if more than 1e-8 of the mass
    sits on the top point of either grid (the grid does not contain the
    ergodic set, e.g. because ``beta (1 + r_a_ss) >= 1``).

    Supported range: ``chi_1 >= 1``.  For ``chi_1 < 1`` a RuntimeWarning is
    issued up front: the marginal adjustment cost has an infinite slope at
    ``d = 0`` and the EGM iteration cycles (residual ~2e-2 for ``chi_1 = 0.5``).
    """
    if not (gamma > 0 and chi_0 > 0 and chi_1 > 0 and a_bar > 0 and w_ss > 0):
        raise ValueError("gamma, chi_0, chi_1, a_bar and w_ss must be positive")
    if chi_1 < 1.0:
        warnings.warn(
            f"two-asset household block: chi_1={chi_1:g} < 1 is outside the supported range chi_1 >= 1; "
            "the marginal adjustment cost has an infinite slope at d = 0 and the EGM iteration is expected "
            "to cycle instead of converging (converged is then False)",
            RuntimeWarning,
            stacklevel=2,
        )
    if not (a_max > 0 and b_max > b_min and int(n_a) >= 3 and int(n_b) >= 3):
        raise ValueError("need a_max > 0, b_max > b_min and at least 3 points on each grid")
    if r_b_ss <= -1 or r_a_ss <= -1:
        raise ValueError("returns must exceed -100%")
    s_grid, pi_s = _income_process(len(_S_GRID))
    if b_min < 0 and (1.0 + r_b_ss) * b_min - b_min + w_ss * s_grid.min() <= 0:
        raise ValueError("b_min is below the natural borrowing limit: consumption cannot stay positive")
    a_grid = _two_asset_grid(0.0, float(a_max), int(n_a))
    b_grid = _two_asset_grid(float(b_min), float(b_max), int(n_b))

    c = (
        w_ss * s_grid[None, None, :]
        + r_b_ss * (b_grid[None, :, None] - min(b_min, 0.0))
        + r_a_ss * a_grid[:, None, None]
    )
    Va = (1.0 + r_a_ss) * c ** (-gamma)
    Vb = (1.0 + r_b_ss) * c ** (-gamma)
    ap = np.zeros_like(c)
    bp = np.zeros_like(c)
    d = np.zeros_like(c)
    chi = np.zeros_like(c)
    params = (a_grid, b_grid, s_grid, pi_s, beta, gamma, r_b_ss, r_a_ss, w_ss, chi_0, chi_1, a_bar)

    converged = False
    stalled = False
    resid = float("inf")
    it = 0
    history: list[float] = []
    win = _TA_EGM_STALL_WINDOW
    for it in range(1, int(max_iter) + 1):
        Va, Vb, c_new, ap_new, bp_new, d, chi = _two_asset_step(Va, Vb, *params)
        resid = float(max(
            np.max(np.abs(c_new - c)), np.max(np.abs(ap_new - ap)), np.max(np.abs(bp_new - bp)),
        ))
        c, ap, bp = c_new, ap_new, bp_new
        history.append(resid)
        if not np.isfinite(resid):
            break
        if resid < tol:
            converged = True
            break
        if it >= 2 * win and it % 50 == 0:
            if min(history[-win:]) > _TA_EGM_STALL_FACTOR * min(history[:-win]):
                stalled = True
                break
    if not converged:
        why = "the residual stalled (the iteration cycles) at" if stalled else "sup-norm policy change"
        warnings.warn(
            f"two-asset household block did not converge: {why} {resid:.2e} after {it} "
            f"iterations (tol {tol:.0e}; beta*(1+r_a)={beta * (1.0 + r_a_ss):.4f}, "
            f"beta*(1+r_b)={beta * (1.0 + r_b_ss):.4f}, chi_1={chi_1:g})",
            RuntimeWarning,
            stacklevel=2,
        )
    n_floor = _two_asset_floored_states(a_grid, b_grid, s_grid, r_b_ss, r_a_ss, w_ss, ap, bp, chi)
    if n_floor:
        warnings.warn(
            f"two-asset household block: the consumption floor binds in {n_floor} states, so the budget "
            "constraint fails there (converged set to False)",
            RuntimeWarning,
            stacklevel=2,
        )
    Lam = _build_transition_matrix_2d(ap, bp, a_grid, b_grid, pi_s)
    D_ss = _stationary_distribution_2d(Lam).reshape(ap.shape)
    stat_ok = bool(np.all(np.isfinite(D_ss)) and abs(D_ss.sum() - 1.0) < 1e-10)

    hh = _TwoAssetHouseholdBlock(
        a_grid=a_grid, b_grid=b_grid, s_grid=s_grid, pi_s=pi_s, beta=float(beta), gamma=float(gamma),
        r_b_ss=float(r_b_ss), r_a_ss=float(r_a_ss), w_ss=float(w_ss), chi_0=float(chi_0), chi_1=float(chi_1),
        a_bar=float(a_bar), Va_ss=Va, Vb_ss=Vb, c_ss=c, d_ss=d, a_ss=ap, b_ss=bp, chi_ss=chi, D_ss=D_ss,
        Lambda=Lam, converged=bool(converged and stat_ok and n_floor == 0), iterations=int(it), residual=resid,
        tol=float(tol),
    )
    top = [
        (name, mass, bound)
        for name, mass, bound in (
            ("illiquid", hh.mass_at_a_max, f"a_max={a_max:g}"),
            ("liquid", hh.mass_at_b_max, f"b_max={b_max:g}"),
        )
        if mass > _TA_GRID_TOP_TOL
    ]
    if top:
        where = " and ".join(f"a share {mass:.3e} of households on the top {name} grid point ({bound})"
                             for name, mass, bound in top)
        warnings.warn(
            f"two-asset steady state: {where}; the grid does not contain the ergodic set and the grid edge "
            f"acts as a binding constraint (beta*(1+r_a)={beta * (1.0 + r_a_ss):.4f}; increase "
            f"{'/'.join(b.split('=')[0] for _, _, b in top)} or lower beta or r_a_ss)",
            RuntimeWarning,
            stacklevel=2,
        )
    return hh


def _two_asset_fake_news_inputs(
    hh: _TwoAssetHouseholdBlock,
    shock_input: str,
    T: int,
    h: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """ABRS step 1: date-0 outcome responses dy_0^s and date-1 distribution responses dD_1^s.

    A unit shock to the realised input (``'r_b'``, ``'r_a'`` or ``'z'``) at
    date ``s`` perturbs the backward step at ``s``; the date-0 policies respond
    through the perturbed continuation marginal values ``(V_a, V_b)``, which
    are propagated one step at a time (``dV_k``), so one backward pass of
    length ``T`` gives every column.  Central differences of size ``h``
    (default: ``_TA_FD_STEP[input]``, 1e-5 for rates and 1e-4 for income).
    Returns ``(dy, dD1)`` with ``dy[o]`` of shape ``(T, N)`` for each outcome in
    ``('C', 'D', 'A', 'B', 'CHI')`` and ``dD1`` of shape ``(T, N)``.
    """
    try:
        inp = _TA_INPUT_ALIASES[shock_input]
    except KeyError:
        raise ValueError(f"shock_input must be one of {sorted(_TA_INPUT_ALIASES)}; got {shock_input!r}") from None
    h = _TA_FD_STEP[inp] if h is None else float(h)
    base = {"r_b": hh.r_b_ss, "r_a": hh.r_a_ss, "z": hh.w_ss}
    N = hh.c_ss.size
    dy = {o: np.zeros((T, N)) for o in _TA_OUTPUTS}
    dD1 = np.zeros((T, N))
    dVa = dVb = None
    for k in range(T):
        if k == 0:
            up = _two_asset_backward(hh, hh.Va_ss, hh.Vb_ss, **{inp: base[inp] + h})
            dn = _two_asset_backward(hh, hh.Va_ss, hh.Vb_ss, **{inp: base[inp] - h})
        else:
            up = _two_asset_backward(hh, hh.Va_ss + h * dVa, hh.Vb_ss + h * dVb)
            dn = _two_asset_backward(hh, hh.Va_ss - h * dVa, hh.Vb_ss - h * dVb)
        dVa = (up[0] - dn[0]) / (2.0 * h)
        dVb = (up[1] - dn[1]) / (2.0 * h)
        o_up, o_dn = _two_asset_outputs(up), _two_asset_outputs(dn)
        for o in _TA_OUTPUTS:
            dy[o][k] = ((o_up[o] - o_dn[o]) / (2.0 * h)).ravel()
        D_up = _two_asset_forward(hh.D_ss, o_up["A"], o_up["B"], hh.a_grid, hh.b_grid, hh.pi_s)
        D_dn = _two_asset_forward(hh.D_ss, o_dn["A"], o_dn["B"], hh.a_grid, hh.b_grid, hh.pi_s)
        dD1[k] = ((D_up - D_dn) / (2.0 * h)).ravel()
    return dy, dD1


def _two_asset_expectation_vectors(hh: _TwoAssetHouseholdBlock, y: np.ndarray, T: int) -> np.ndarray:
    """E[t] = (Lambda')^t y, shape (T, N)."""
    E = np.empty((T,) + hh.c_ss.shape)
    E[0] = y
    for t in range(1, T):
        E[t] = _two_asset_expectation(E[t - 1], hh.a_ss, hh.b_ss, hh.a_grid, hh.b_grid, hh.pi_s)
    return E.reshape(T, -1)


def _two_asset_household_jacobians(
    hh: _TwoAssetHouseholdBlock,
    T: int,
    inputs: Sequence[str] = _TA_INPUTS,
    outputs: Sequence[str] = _TA_OUTPUTS,
    h: float | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Fake-News Jacobians of the household block alone, dated by the *realised* inputs.

    ``J[(o, x)][t, s] = dO_t / dx_s`` for aggregate outcome ``O`` (C, D, A = sum a',
    B = sum b', CHI) and realised input ``x`` (``r_b``, ``r_a``, ``z``), with no
    income closure.  ABRS (2021) steps 1-4: ``F[0, s] = D_ss' dy_0^s``,
    ``F[t, s] = E_{t-1}' dD_1^s``, ``J[t, s] = J[t-1, s-1] + F[t, s]``.
    """
    T = int(T)
    D_flat = hh.D_ss.ravel()
    ss_outputs = {"C": hh.c_ss, "D": hh.d_ss, "A": hh.a_ss, "B": hh.b_ss, "CHI": hh.chi_ss}
    E = {o: _two_asset_expectation_vectors(hh, ss_outputs[o], T) for o in outputs}
    J: dict[tuple[str, str], np.ndarray] = {}
    for x in inputs:
        dy, dD1 = _two_asset_fake_news_inputs(hh, x, T, h)
        for o in outputs:
            F = np.empty((T, T))
            F[0] = dy[o] @ D_flat
            if T > 1:
                F[1:] = E[o][:-1] @ dD1.T
            Jo = np.empty((T, T))
            Jo[0] = F[0]
            for t in range(1, T):
                Jo[t, 0] = F[t, 0]
                Jo[t, 1:] = Jo[t - 1, :-1] + F[t, 1:]
            J[(o, _TA_INPUT_ALIASES[x])] = Jo
    return J


def _two_asset_closure_matrices(
    hh: _TwoAssetHouseholdBlock,
    raw: dict[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Response of non-financial income ``dz`` under the income closure, per realised input.

    Linearising ``z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}`` around the steady
    state, with ``dA``/``dB`` the households' own responses
    (``dA = J_Az dz + J_A,rb dr^b + J_A,ra dr^a``, likewise ``dB``), gives

        (N I + L (r_b J_Bz + r_a J_Az)) dz
            = dY - (B I + L (r_b J_B,rb + r_a J_A,rb)) dr^b - (A I + L (r_b J_B,ra + r_a J_A,ra)) dr^a,

    with ``L`` the lag operator ``(L x)_t = x_{t-1}`` and realised rates.
    Returns ``{'Y': G_Y, 'r_b': G_rb, 'r_a': G_ra}`` with ``dz = G_Y dY + G_rb dr^b + G_ra dr^a``
    on the horizon of ``raw``.
    """
    T = raw[("A", "z")].shape[0]
    N, A, B = hh.N_ss, hh.A_ss, hh.B_ss
    rb, ra = hh.r_b_ss, hh.r_a_ss
    eye = np.eye(T)
    lag = np.eye(T, k=-1)

    def interest(x: str) -> np.ndarray:  # r_b dB + r_a dA per unit of realised input x
        return rb * raw[("B", x)] + ra * raw[("A", x)]

    Mz = N * eye + lag @ interest("z")
    return {
        "Y": np.linalg.solve(Mz, eye),
        "r_b": -np.linalg.solve(Mz, B * eye + lag @ interest("r_b")),
        "r_a": -np.linalg.solve(Mz, A * eye + lag @ interest("r_a")),
    }


def _two_asset_jacobians(hh: _TwoAssetHouseholdBlock, T: int) -> dict[str, np.ndarray]:
    """GE-ready Jacobians of the household sector closed by the income rule, ex-ante dated.

    Inputs are aggregate output ``Y`` and the ex-ante liquid/illiquid real
    rates ``r^{b,ea}_s``, ``r^{a,ea}_s`` (realised by households at ``s + 1``).
    Non-financial income is not an input: it is solved out through the closure
    ``z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}`` on the households' own lagged
    holdings (see :func:`_two_asset_closure_matrices`), so household income
    equals ``Y``.  With ``J^{real}`` the raw household Jacobians of
    :func:`_two_asset_household_jacobians` on ``T + 1`` dates and ``G`` the
    closure matrices,

        J_O_Y = (J^{real}_{O,z} G_Y)[:T, :T]
        J_O_rb = (J^{real}_{O,r_b} + J^{real}_{O,z} G_rb)[:T, 1:T+1]   (ex-ante shift)
        J_O_ra = (J^{real}_{O,r_a} + J^{real}_{O,z} G_ra)[:T, 1:T+1]

    for outcome ``O`` in (C, D, A, B, CHI).  ``J_z_Y``, ``J_z_rb`` and ``J_z_ra``
    give the implied response of ``z``; ``J_C_r`` is an alias of ``J_C_rb``.
    The aggregate budget then reads ``J_Q + (I - L) J_W = I`` for ``Y`` and
    ``0`` for the rates (``Q = C + CHI``, ``W = A + B``), so in general
    equilibrium (``dY = dQ``) total wealth does not move.
    """
    T = int(T)
    raw = _two_asset_household_jacobians(hh, T + 1)
    G = _two_asset_closure_matrices(hh, raw)
    out: dict[str, np.ndarray] = {}
    for o in _TA_OUTPUTS:
        Jz = raw[(o, "z")]
        out[f"J_{o}_Y"] = np.ascontiguousarray((Jz @ G["Y"])[:T, :T])
        out[f"J_{o}_rb"] = np.ascontiguousarray((raw[(o, "r_b")] + Jz @ G["r_b"])[:T, 1:T + 1])
        out[f"J_{o}_ra"] = np.ascontiguousarray((raw[(o, "r_a")] + Jz @ G["r_a"])[:T, 1:T + 1])
    out["J_z_Y"] = np.ascontiguousarray(G["Y"][:T, :T])
    out["J_z_rb"] = np.ascontiguousarray(G["r_b"][:T, 1:T + 1])
    out["J_z_ra"] = np.ascontiguousarray(G["r_a"][:T, 1:T + 1])
    out["J_C_r"] = out["J_C_rb"]
    return out


def _two_asset_transition(
    hh: _TwoAssetHouseholdBlock,
    r_b_path: np.ndarray,
    r_a_path: np.ndarray,
    z_path: np.ndarray,
    transfer_path: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Non-linear household block along realised input paths (direct method).

    Paths have length ``T`` and hold the realised returns and non-financial
    income at dates ``0..T-1``; after ``T`` the economy is back at the steady
    state.  Policies are iterated backward from ``(Va_ss, Vb_ss)`` and the
    distribution is pushed forward from ``D_ss``.  Returns aggregate paths of
    C, D, A (= sum D a'), B (= sum D b') and CHI.
    """
    T = len(r_b_path)
    if transfer_path is None:
        transfer_path = np.zeros(T)
    Va, Vb = hh.Va_ss, hh.Vb_ss
    pols: list[dict[str, np.ndarray]] = [None] * T  # type: ignore[list-item]
    n_floor = 0
    for t in range(T - 1, -1, -1):
        step = _two_asset_backward(hh, Va, Vb, r_b=r_b_path[t], r_a=r_a_path[t], z=z_path[t],
                                   transfer=float(transfer_path[t]))
        Va, Vb = step[0], step[1]
        pols[t] = _two_asset_outputs(step)
        n_floor += _two_asset_floored_states(
            hh.a_grid, hh.b_grid, hh.s_grid, float(r_b_path[t]), float(r_a_path[t]), float(z_path[t]),
            step[3], step[4], step[6], float(transfer_path[t]),
        )
    if n_floor:
        warnings.warn(
            f"two-asset transition: the consumption floor binds in {n_floor} (state, date) pairs, so the "
            "budget constraint fails there; the shock is too large for this grid",
            RuntimeWarning,
            stacklevel=2,
        )
    D = hh.D_ss.copy()
    agg = {o: np.empty(T) for o in _TA_OUTPUTS}
    for t in range(T):
        for o in _TA_OUTPUTS:
            agg[o][t] = float(np.sum(D * pols[t][o]))
        D = _two_asset_forward(D, pols[t]["A"], pols[t]["B"], hh.a_grid, hh.b_grid, hh.pi_s)
    return agg


def _two_asset_closed_transition(
    hh: _TwoAssetHouseholdBlock,
    Y_path: np.ndarray,
    r_b_path: np.ndarray,
    r_a_path: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 200,
) -> dict[str, np.ndarray]:
    """Non-linear household block under the income closure (direct method).

    ``Y_path`` is output in levels and ``r_b_path``/``r_a_path`` the returns
    *realised* at dates ``0..T-1``.  Non-financial income solves the fixed point

        z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1},   B_{-1} = B_ss, A_{-1} = A_ss,

    with ``A_t``/``B_t`` the aggregate holdings chosen at ``t`` along the
    non-linear transition; it is found by fixed-point iteration on the ``z``
    path (sup-norm tolerance ``tol``; RuntimeWarning if not reached).  Returns
    the aggregates of :func:`_two_asset_transition` plus ``'z'``.
    """
    Y_path = np.asarray(Y_path, dtype=float)
    r_b_path = np.asarray(r_b_path, dtype=float)
    r_a_path = np.asarray(r_a_path, dtype=float)
    N = hh.N_ss
    B_lag = np.full(len(Y_path), hh.B_ss)
    A_lag = np.full(len(Y_path), hh.A_ss)
    z = (Y_path - r_b_path * B_lag - r_a_path * A_lag) / N
    step = float("inf")
    for _ in range(int(max_iter)):
        agg = _two_asset_transition(hh, r_b_path, r_a_path, z)
        B_lag[1:], A_lag[1:] = agg["B"][:-1], agg["A"][:-1]
        z_new = (Y_path - r_b_path * B_lag - r_a_path * A_lag) / N
        step = float(np.max(np.abs(z_new - z)))
        z = z_new
        if step < tol:
            break
    else:
        warnings.warn(f"two-asset closed transition: z fixed point not reached (step {step:.1e})",
                      RuntimeWarning, stacklevel=2)
    agg = _two_asset_transition(hh, r_b_path, r_a_path, z)
    agg["z"] = z
    return agg


@dataclass(frozen=True)
class TwoAssetSequenceSpaceHANKResult:
    """Results from the two-asset sequence-space HANK general-equilibrium solve.

    IRFs are deviations from the steady state in levels (``irf_output``,
    ``irf_consumption``, ``irf_deposit``: the aggregate net deposit flow
    ``sum D (a' - (1 + r_a) a)``) or in quarterly rates (``irf_inflation``,
    ``irf_rate_b``, ``irf_rate_a``: the *ex-ante* real returns, realised one
    quarter later).  The Jacobians are ``(T, T)`` Jacobians of the household
    sector closed by the income rule ``z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}``
    (inputs: output ``Y`` and the ex-ante rates; see :func:`_two_asset_jacobians`
    and the module comment above ``_transaction_cost``).  In equilibrium total
    wealth ``A + B`` does not move; only its composition does.  ``steady_state`` also carries the
    household-block diagnostics (``hh_iterations``, ``hh_residual``,
    ``mass_at_a_max``, ``mass_at_b_max``, ``htm_share``, ``wealthy_htm_share``,
    ``mpc``, ``budget_residual``).  ``converged`` is True only if the household
    fixed point reached its tolerance and the GE solve is finite.
    """
    irf_output: np.ndarray
    irf_consumption: np.ndarray
    irf_deposit: np.ndarray
    irf_inflation: np.ndarray
    irf_rate_b: np.ndarray
    irf_rate_a: np.ndarray
    jacobian_c_rb: np.ndarray
    jacobian_c_ra: np.ndarray
    jacobian_c_y: np.ndarray
    jacobian_d_rb: np.ndarray
    jacobian_d_ra: np.ndarray
    jacobian_d_y: np.ndarray
    asset_grid: np.ndarray
    liquid_asset_grid: np.ndarray
    joint_distribution: np.ndarray
    marginal_distribution_a: np.ndarray
    marginal_distribution_b: np.ndarray
    deposit_distribution: np.ndarray
    steady_state: dict[str, float]
    policy_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    policy_d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    policy_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    policy_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    distribution: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    trans_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    horizon: int = 40
    beta: float = 0.98
    gamma: float = 1.0
    r_b_ss: float = 0.005
    r_a_ss: float = 0.0125
    chi_0: float = 1.0
    chi_1: float = 1.0
    converged: bool = True

    def summary(self) -> str:
        y_pk, y_at = _peak(self.irf_output)
        c_pk, c_at = _peak(self.irf_consumption)
        d_pk, d_at = _peak(self.irf_deposit)
        pi_pk, pi_at = _peak(self.irf_inflation)
        ss = self.steady_state
        lines = [
            "Two-Asset Sequence-Space HANK General Equilibrium Solve (Kaplan et al. 2018; Auclert et al. 2021)",
            "=" * 78,
            f"Horizon T                       : {self.horizon} periods",
            f"Steady-State Illiquid Return r_a: {self.r_a_ss:.4f}",
            f"Steady-State Liquid Return r_b  : {self.r_b_ss:.4f}",
            f"Adjustment Cost Parameters      : chi_0={self.chi_0:.2f}, chi_1={self.chi_1:.2f}",
            f"Steady-State Output Y_ss        : {ss.get('Y', np.nan):.4f}",
            f"Steady-State Consumption C_ss   : {ss.get('C', np.nan):.4f}",
            f"Illiquid / Liquid Wealth A, B   : {ss.get('A', np.nan):.4f}, {ss.get('B', np.nan):.4f}",
            f"Hand-to-Mouth Share (wealthy)   : {ss.get('htm_share', np.nan):.3f} ({ss.get('wealthy_htm_share', np.nan):.3f})",
            f"Household Block Converged       : {self.converged} "
            f"({int(ss.get('hh_iterations', 0))} iterations, residual {ss.get('hh_residual', np.nan):.1e})",
            f"Peak Output Response            : {y_pk:+.6f} (t={y_at})",
            f"Peak Consumption Response       : {c_pk:+.6f} (t={c_at})",
            f"Peak Deposit Flow Response      : {d_pk:+.6f} (t={d_at})",
            f"Peak Inflation Response         : {pi_pk:+.6f} (t={pi_at})",
            "=" * 78,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        data = {
            "Y": self.irf_output,
            "C": self.irf_consumption,
            "D": self.irf_deposit,
            "pi": self.irf_inflation,
            "r_b": self.irf_rate_b,
            "r_a": self.irf_rate_a,
        }
        return pd.DataFrame(data)

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, figsize: tuple[float, float] = (12.0, 7.0)):
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        ax = axes.ravel()
        t_grid = np.arange(self.horizon)

        ax[0].plot(t_grid, self.irf_output, color="#1f77b4", lw=2)
        ax[0].set_title("Output dY", fontweight="bold")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_grid, self.irf_consumption, color="#2ca02c", lw=2)
        ax[1].set_title("Consumption dC", fontweight="bold")
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(t_grid, self.irf_deposit, color="#9467bd", lw=2)
        ax[2].set_title("Portfolio Deposits dD", fontweight="bold")
        ax[2].grid(True, alpha=0.3)

        ax[3].plot(t_grid, self.irf_inflation, color="#d62728", lw=2)
        ax[3].set_title("Inflation dpi", fontweight="bold")
        ax[3].grid(True, alpha=0.3)

        ax[4].plot(t_grid, self.irf_rate_b, color="#ff7f0e", lw=2, label="Liquid dr_b")
        ax[4].plot(t_grid, self.irf_rate_a, color="#8c564b", lw=2, linestyle="--", label="Illiquid dr_a")
        ax[4].set_title("Real Interest Rates", fontweight="bold")
        ax[4].legend(frameon=False)
        ax[4].grid(True, alpha=0.3)

        A, B = np.meshgrid(self.asset_grid, self.liquid_asset_grid, indexing="ij")
        cp = ax[5].contourf(A, B, self.joint_distribution, cmap="viridis")
        fig.colorbar(cp, ax=ax[5], fraction=0.046, pad=0.04)
        ax[5].set_title(r"Joint Wealth Distribution $\mathcal{D}^*(a, b)$", fontweight="bold")
        ax[5].set_xlabel("Illiquid $")
        ax[5].set_ylabel("Liquid $")

        fig.tight_layout()
        return fig


def _two_asset_steady_state_dict(hh: _TwoAssetHouseholdBlock) -> dict[str, float]:
    """Aggregate steady state and household-block diagnostics of a two-asset block."""
    return {
        "Y": hh.Y_ss,
        "C": hh.C_ss,
        "CHI": hh.CHI_ss,
        "D": hh.D_flow_ss,
        "r_b": hh.r_b_ss,
        "r_a": hh.r_a_ss,
        "pi": 0.0,
        "i": hh.r_b_ss,
        "w": hh.w_ss,
        "A": hh.A_ss,
        "B": hh.B_ss,
        "N": hh.N_ss,
        "mpc": float(np.sum(hh.D_ss * hh.mpc())),
        "htm_share": hh.htm_share,
        "wealthy_htm_share": hh.wealthy_htm_share,
        "mass_at_a_max": hh.mass_at_a_max,
        "mass_at_b_max": hh.mass_at_b_max,
        "budget_residual": hh.budget_residual,
        "hh_iterations": float(hh.iterations),
        "hh_residual": hh.residual,
    }


def solve_two_asset_hank_sequence_space(
    T: int = 40,
    beta: float = 0.98,
    gamma: float = 1.0,
    r_b_ss: float = 0.005,
    r_a_ss: float = 0.0125,
    phi_pi: float = 1.5,
    kappa: float = 0.1,
    chi_0: float = 1.0,
    chi_1: float = 1.0,
    shock_magnitude: float = -0.0025,
    shock_rho: float = 0.5,
    n_a: int = 25,
    n_b: int = 25,
    a_max: float = 40.0,
    b_max: float = 10.0,
    b_min: float = 0.0,
    alpha_ab: float = 0.5,
    a_bar: float = 0.25,
) -> TwoAssetSequenceSpaceHANKResult:
    """Solve the two-asset HANK steady state, its Fake-News Jacobians and the linear GE response.

    Households choose consumption, liquid bonds ``b`` and an illiquid asset
    ``a`` that can be adjusted only at the convex cost
    ``chi(d, a) = chi_0/(1+chi_1) |d|^(1+chi_1) / (a + a_bar)^chi_1``
    (``d = a' - (1 + r_a) a``; Kaplan, Moll & Violante 2018).  The steady
    state is solved by EGM on ``(V_a, V_b)`` to a sup-norm tolerance of 1e-11,
    the household Jacobians by the Fake-News Algorithm (Auclert, Bardóczy,
    Rognlie & Straub 2021) with anticipation effects, and the general
    equilibrium by one ``T x T`` solve of the goods-market condition
    ``dY = d(C + CHI)`` with ``dr_b = M_r_Y dY + eps`` (NKPC, Taylor rule and
    Fisher equation, as in :func:`solve_hank_sequence_space`) and
    ``dr_a = alpha_ab dr_b``.  Asset returns are paid out of aggregate income
    on the holdings households carry into the period
    (``z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}``), so household income
    equals output, a rate change redistributes income between savers and
    workers, and in equilibrium total wealth ``A + B`` stays at its steady
    state while its composition moves (Walras's law).  See the comment block
    above ``_transaction_cost`` for the timing and closure conventions.

    This is a stylised teaching model (Kaplan-Moll-Violante cost structure,
    Auclert et al. solution method); it is not a replication of either paper's
    calibration.  Supported adjustment-cost curvature: ``chi_1 >= 1``.

    Parameters
    ----------
    T : int, default 40
        Horizon of the IRFs and of the ``(T, T)`` Jacobians (quarters).
    beta : float, default 0.98
        Household discount factor (also the NKPC discount factor).  The
        illiquid-wealth distribution is interior only if ``beta (1 + r_a_ss) < 1``.
    gamma : float, default 1.0
        Relative risk aversion (CRRA; 1 is log utility).
    r_b_ss, r_a_ss : float, default 0.005, 0.0125
        Steady-state quarterly real returns on the liquid and illiquid assets
        (2% and 5% a year).
    phi_pi, kappa : float, default 1.5, 0.1
        Taylor-rule inflation coefficient and NKPC slope.
    chi_0, chi_1, a_bar : float, default 1.0, 1.0, 0.25
        Adjustment-cost scale, curvature (``chi_1 = 1`` is quadratic; values
        below 1 are unsupported and warn) and the shift that keeps depositing
        into an empty account affordable.
    shock_magnitude, shock_rho : float, default -0.0025, 0.5
        Monetary shock ``eps_t = shock_magnitude * shock_rho**t`` to the
        Taylor rule (-0.0025 is a 25 bp quarterly cut).
    n_a, n_b, a_max, b_max, b_min : grid sizes and bounds
        Illiquid grid on ``[0, a_max]`` and liquid grid on ``[b_min, b_max]``,
        both denser near the lower bound.  A RuntimeWarning is issued if the
        grids do not contain the ergodic set.
    alpha_ab : float, default 0.5
        Pass-through of the ex-ante liquid real rate to the illiquid rate.

    Returns
    -------
    TwoAssetSequenceSpaceHANKResult
    """
    T = int(T)
    if T < 1:
        raise ValueError("T must be a positive integer")
    hh = _solve_two_asset_household_block(
        beta=float(beta),
        gamma=float(gamma),
        r_b_ss=float(r_b_ss),
        r_a_ss=float(r_a_ss),
        n_a=int(n_a),
        n_b=int(n_b),
        a_max=float(a_max),
        b_max=float(b_max),
        b_min=float(b_min),
        chi_0=float(chi_0),
        chi_1=float(chi_1),
        a_bar=float(a_bar),
    )
    shock_seq = float(shock_magnitude) * (float(shock_rho) ** np.arange(T))
    return _two_asset_ge_solve(hh, T, float(phi_pi), float(kappa), shock_seq, float(alpha_ab))


def _two_asset_ge_paths(
    hh: _TwoAssetHouseholdBlock,
    T: int,
    phi_pi: float,
    kappa: float,
    shock_seq: np.ndarray,
    alpha_ab: float = 0.5,
    jacobians: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Linear GE paths of the two-asset model (deviations from the steady state).

    Returns a dict with the ``(T,)`` paths ``Y``, ``C``, ``CHI``, ``D``, ``A``,
    ``B``, ``z`` (non-financial income per efficiency unit), ``r_b``, ``r_a``
    (ex-ante real rates), ``pi``, ``i`` and the Jacobians used
    (``'jacobians'``; computed with :func:`_two_asset_jacobians` unless given).
    By construction ``dY = dC + dCHI`` and ``dA + dB = 0`` up to rounding.
    """
    T = int(T)
    shock_seq = np.asarray(shock_seq, dtype=float)
    if jacobians is None:
        jacobians = _two_asset_jacobians(hh, T=T)
    J = jacobians
    alpha_ab = float(alpha_ab)
    K_pi, M_r_Y = _ge_matrices(T, hh.beta, float(kappa), float(phi_pi))

    # Goods market: dY = d(C + CHI); spending Jacobians with respect to Y and to the ex-ante liquid rate
    JQ_Y = J["J_C_Y"] + J["J_CHI_Y"]
    JQ_r = (J["J_C_rb"] + J["J_CHI_rb"]) + alpha_ab * (J["J_C_ra"] + J["J_CHI_ra"])
    LHS = np.eye(T) - JQ_Y - JQ_r @ M_r_Y
    dY = np.linalg.solve(LHS, JQ_r @ shock_seq)
    dr_b = M_r_Y @ dY + shock_seq
    dr_a = alpha_ab * dr_b
    dpi = K_pi @ dY
    out: dict[str, Any] = {
        o: J[f"J_{o}_Y"] @ dY + J[f"J_{o}_rb"] @ dr_b + J[f"J_{o}_ra"] @ dr_a for o in _TA_OUTPUTS + ("z",)
    }
    out.update({"Y": dY, "r_b": dr_b, "r_a": dr_a, "pi": dpi, "i": float(phi_pi) * dpi + shock_seq,
                "jacobians": J})
    return out


def _two_asset_ge_solve(
    hh: _TwoAssetHouseholdBlock,
    T: int,
    phi_pi: float,
    kappa: float,
    shock_seq: np.ndarray,
    alpha_ab: float = 0.5,
) -> TwoAssetSequenceSpaceHANKResult:
    """Linear GE response of a solved two-asset household block to a Taylor-rule shock path.

    Goods market ``dY = d(C + CHI)`` with ``dC + dCHI = J_Y dY + J_rb dr_b + J_ra dr_a``
    (closed household-sector Jacobians of :func:`_two_asset_jacobians`),
    ``dr_b = M_r_Y dY + eps`` and ``dr_a = alpha_ab dr_b`` (ex-ante rates):
    ``(I - J_Y - (J_rb + alpha_ab J_ra) M_r_Y) dY = (J_rb + alpha_ab J_ra) eps``.
    """
    paths = _two_asset_ge_paths(hh, T, phi_pi, kappa, shock_seq, alpha_ab)
    jacobians = paths["jacobians"]
    dY = paths["Y"]
    ss_dict = _two_asset_steady_state_dict(hh)
    converged = bool(hh.converged and np.all(np.isfinite(dY)))

    return TwoAssetSequenceSpaceHANKResult(
        irf_output=dY,
        irf_consumption=paths["C"],
        irf_deposit=paths["D"],
        irf_inflation=paths["pi"],
        irf_rate_b=paths["r_b"],
        irf_rate_a=paths["r_a"],
        jacobian_c_rb=jacobians["J_C_rb"],
        jacobian_c_ra=jacobians["J_C_ra"],
        jacobian_c_y=jacobians["J_C_Y"],
        jacobian_d_rb=jacobians["J_D_rb"],
        jacobian_d_ra=jacobians["J_D_ra"],
        jacobian_d_y=jacobians["J_D_Y"],
        asset_grid=hh.a_grid,
        liquid_asset_grid=hh.b_grid,
        joint_distribution=hh.joint_distribution,
        marginal_distribution_a=hh.marginal_distribution_a,
        marginal_distribution_b=hh.marginal_distribution_b,
        deposit_distribution=hh.deposit_distribution,
        steady_state=ss_dict,
        policy_c=hh.c_ss,
        policy_d=hh.d_ss,
        policy_a=hh.a_ss,
        policy_b=hh.b_ss,
        distribution=hh.D_ss,
        trans_matrix=hh.Lambda,
        horizon=T,
        beta=hh.beta,
        gamma=hh.gamma,
        r_b_ss=hh.r_b_ss,
        r_a_ss=hh.r_a_ss,
        chi_0=hh.chi_0,
        chi_1=hh.chi_1,
        converged=converged,
    )


__all__ = [
    "SequenceSpaceHANKResult",
    "TwoAssetSequenceSpaceHANKResult",
    "FakeNewsResult",
    "FiscalTransferResult",
    "NonlinearHANKResult",
    "solve_hank_sequence_space",
    "solve_two_asset_hank_sequence_space",
    "fake_news_algorithm",
    "simulate_targeted_transfer",
    "solve_nonlinear_transition",
]
