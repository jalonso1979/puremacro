"""Krusell-Smith (1998): heterogeneous agents with aggregate shocks.

Agents perceive the aggregate via mean capital K and forecast K' with a
log-linear rule per aggregate state. Folding K (on a grid) into the exogenous
state makes the household a standard VFIProblem (endogenous a; exogenous
(z, Z, K)); the equilibrium forecast rule is a fixed point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from puremacro.vfi.discretize import markov_stationary
from puremacro.vfi.distribution import lottery_push, stationary_distribution
from puremacro.vfi.problem import VFIProblem


def ks_exog_transition(z_vals, P_z, Z_vals, P_Z, K_grid, b0, b1):
    """Combined exogenous transition over the C-order product (z, Z, K).

    ``P[(z,Z,K) -> (z',Z',K')] = P_z[z,z'] * P_Z[Z,Z'] * w_lottery(K' | K'(Z,K))``
    where the forecast ``K'(Z,K) = exp(b0[Z] + b1[Z]*log K)`` is scattered onto the
    K-grid by linear (mean-preserving) lottery weights, clamped to the grid range.
    Returns ``(P_combined (N,N), values (N,3))`` with ``N = nz*nZ*nK`` and
    ``values[s] = (z, Z, K)`` enumerated in C-order (flat = (iz*nZ + iZ)*nK + iK).
    """
    z_vals = np.asarray(z_vals, dtype=float)
    P_z = np.asarray(P_z, dtype=float)
    Z_vals = np.asarray(Z_vals, dtype=float)
    P_Z = np.asarray(P_Z, dtype=float)
    K = np.asarray(K_grid, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    nz, nZ, nK = z_vals.size, Z_vals.size, K.size

    # K-lottery per aggregate state: KL[iZ, iK, iK'] (rows sum to 1)
    logK = np.log(K)
    KL = np.zeros((nZ, nK, nK))
    for iZ in range(nZ):
        Kp = np.exp(b0[iZ] + b1[iZ] * logK)             # (nK,) forecast next K
        j = np.clip(np.searchsorted(K, Kp) - 1, 0, nK - 2)
        w_lo = np.clip((K[j + 1] - Kp) / (K[j + 1] - K[j]), 0.0, 1.0)
        rows = np.arange(nK)
        KL[iZ, rows, j] += w_lo
        KL[iZ, rows, j + 1] += 1.0 - w_lo

    # P6[iz,iZ,iK, iz',iZ',iK'] = P_z[iz,iz'] P_Z[iZ,iZ'] KL[iZ,iK,iK']
    P6 = (P_z.reshape(nz, 1, 1, nz, 1, 1)
          * P_Z.reshape(1, nZ, 1, 1, nZ, 1)
          * KL.reshape(1, nZ, nK, 1, 1, nK))
    N = nz * nZ * nK
    P_combined = P6.reshape(N, N)

    mz, mZ, mK = np.meshgrid(z_vals, Z_vals, K, indexing="ij")
    values = np.stack([mz.reshape(-1), mZ.reshape(-1), mK.reshape(-1)], axis=1)
    return P_combined, values


def ks_simulate(policy_aprime, a_grid, P_z, K_grid, Z_path, *, mu0=None):
    """Simulate the wealth distribution forward along an aggregate-shock path.

    ``policy_aprime`` is (n_a, nz, nZ, nK) int a'-indices (the household policy
    reshaped over the flat (z,Z,K) exogenous state). Each period: the realized
    aggregate ``K_t = E[a]`` under the current measure; the policy is interpolated
    in K at the realized ``K_t`` (continuous a'); the measure is pushed via
    ``lottery_push`` (a-lottery + idiosyncratic ``P_z``). Returns ``K_path`` of
    length ``len(Z_path)+1`` (``K_path[t]`` is mean capital entering period t).
    """
    pol = np.asarray(policy_aprime)
    a = np.asarray(a_grid, dtype=float)
    P = np.asarray(P_z, dtype=float)
    Kg = np.asarray(K_grid, dtype=float)
    n_a, nz, nZ, nK = pol.shape
    Zp = np.asarray(Z_path, dtype=int)
    T = Zp.size
    aval = a[pol]                                        # (n_a, nz, nZ, nK) realized a'
    if mu0 is None:
        mu = np.zeros((n_a, nz))
        mu[0, :] = markov_stationary(P)                  # all at a=0; z ~ stationary
    else:
        mu = np.asarray(mu0, dtype=float).copy()
        mu = mu / mu.sum()
    K_path = np.empty(T + 1)
    for t in range(T):
        K_t = float(np.sum(mu * a[:, None]))
        K_path[t] = K_t
        iZ = int(Zp[t])
        Kc = min(max(K_t, Kg[0]), Kg[-1])
        j = int(np.clip(np.searchsorted(Kg, Kc) - 1, 0, nK - 2))
        w = (Kg[j + 1] - Kc) / (Kg[j + 1] - Kg[j])
        ap_val = w * aval[:, :, iZ, j] + (1.0 - w) * aval[:, :, iZ, j + 1]
        mu = lottery_push(mu, ap_val, a, P)
    K_path[T] = float(np.sum(mu * a[:, None]))
    return K_path


@dataclass(frozen=True)
class KSEquilibrium:
    """A solved Krusell-Smith approximate equilibrium."""
    b0: np.ndarray           # forecast intercept per aggregate state
    b1: np.ndarray           # forecast log-K slope per aggregate state
    r_squared: np.ndarray    # forecast R^2 per aggregate state
    K_path: np.ndarray       # simulated mean-capital path
    mean_K: float            # post-burn-in mean of K
    no_agg_risk_K: float     # the no-aggregate-risk (Aiyagari) capital K*
    n_outer: int
    converged: bool


def _firm_prices(Z, K, L, alpha, delta):
    KL = K / L
    return alpha * Z * KL ** (alpha - 1.0) - delta, (1.0 - alpha) * Z * KL ** alpha


def _util(c, gamma, xp):
    cc = xp.maximum(c, 1e-12)
    if gamma == 1.0:
        return xp.log(cc)
    return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _solve_household(b0, b1, *, z_vals, P_z, Z_vals, P_Z, K_grid, a_grid, L,
                     alpha, delta, beta, gamma, options):
    P_comb, _ = ks_exog_transition(z_vals, P_z, Z_vals, P_Z, K_grid, b0, b1)

    def rf(ap, a, zc, Zc, Kc, xp=np):
        r, w = _firm_prices(Zc, Kc, L, alpha, delta)
        c = (1.0 + r) * a + w * zc - ap
        return xp.where(c > 0.0, _util(c, gamma, xp), -np.inf)

    sol = VFIProblem(a_grid=a_grid, z_grid=[z_vals, Z_vals, K_grid], P_z=P_comb,
                     return_fn=rf, beta=beta, options=options).solve("numpy")
    return sol.policy_aprime.reshape(a_grid.size, z_vals.size, Z_vals.size, K_grid.size)


def _no_agg_risk_K(K_bracket, *, z_vals, P_z, L, alpha, delta, beta, gamma, a_grid,
                   options):
    """The no-aggregate-risk (single Z=1) Aiyagari capital: K* with household
    saving == K* (the reduction reference; also centers the K-grid)."""
    def excess(K):
        r, w = _firm_prices(1.0, K, L, alpha, delta)

        def rf(ap, a, zc, xp=np):
            c = (1.0 + r) * a + w * zc - ap
            return xp.where(c > 0.0, _util(c, gamma, xp), -np.inf)

        sol = VFIProblem(a_grid=a_grid, z_grid=z_vals, P_z=P_z, return_fn=rf,
                         beta=beta, options=options).solve("numpy")
        mu = stationary_distribution(sol.policy_aprime, P_z)
        return float(np.sum(mu * a_grid[:, None])) - K

    return float(brentq(excess, K_bracket[0], K_bracket[1], xtol=1e-4))


def _simulate_markov_path(P, T, rng):
    cdf = np.cumsum(np.asarray(P, dtype=float), axis=1)
    path = np.empty(T, dtype=int)
    s = 0
    for t in range(T):
        path[t] = s
        s = int(np.searchsorted(cdf[s], rng.random()))
    return path


def _regress_forecast(K_path, Z_path, nZ, burn_in):
    logK = np.log(K_path)
    x = logK[burn_in:-1]                                 # log K_t
    y = logK[burn_in + 1:]                               # log K_{t+1}
    zt = np.asarray(Z_path)[burn_in:]                    # Z_t
    b0 = np.zeros(nZ)
    b1 = np.ones(nZ)
    R2 = np.ones(nZ)
    for iZ in range(nZ):
        m = zt == iZ
        xi, yi = x[m], y[m]
        if xi.size < 2 or float(np.var(xi)) < 1e-14:   # no K-variation (e.g. no agg risk)
            b0[iZ] = float(np.mean(yi)) if yi.size else 0.0
            b1[iZ] = 0.0
            R2[iZ] = 1.0
            continue
        A = np.vstack([np.ones_like(xi), xi]).T
        coef, *_ = np.linalg.lstsq(A, yi, rcond=None)
        b0[iZ], b1[iZ] = float(coef[0]), float(coef[1])
        ss_tot = float(np.sum((yi - yi.mean()) ** 2))
        ss_res = float(np.sum((yi - A @ coef) ** 2))
        R2[iZ] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return b0, b1, R2


def krusell_smith(*, alpha: float = 0.36, delta: float = 0.025, beta: float = 0.99,
                  gamma: float = 1.0, e=(0.5, 1.1), P_z=((0.6, 0.4), (0.2, 0.8)),
                  Z_vals=(0.99, 1.01), P_Z=((0.875, 0.125), (0.125, 0.875)),
                  n_a: int = 200, a_max=None, n_K: int = 7, T: int = 3000,
                  burn_in: int = 300, seed: int = 0, damping: float = 0.5,
                  tol: float = 1e-3, max_outer: int = 60, solve_options=None):
    """Solve a Krusell-Smith approximate equilibrium (see module docstring).

    Iterates the per-aggregate-state log-linear forecast ``log K' = b0[Z] +
    b1[Z] log K`` to a fixed point: solve the household (K folded into the
    exogenous state), simulate the wealth distribution along a drawn aggregate
    path, OLS-update the forecast, damp, repeat. v1: 2-state idiosyncratic
    productivity, 2-state aggregate TFP, Cobb-Douglas firm. Returns a KSEquilibrium.
    """
    z_vals = np.asarray(e, dtype=float)
    P_z = np.asarray(P_z, dtype=float)
    Z_vals = np.asarray(Z_vals, dtype=float)
    P_Z = np.asarray(P_Z, dtype=float)
    nZ = Z_vals.size
    L = float(markov_stationary(P_z) @ z_vals)
    options = solve_options or dict(tol=1e-7, n_howard=40)

    K_cm = L * (alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    if a_max is None:
        a_max = 4.0 * K_cm
    a_grid = np.linspace(0.0, float(a_max), n_a)
    Kstar = _no_agg_risk_K((0.4 * K_cm, 4.0 * K_cm), z_vals=z_vals, P_z=P_z, L=L,
                           alpha=alpha, delta=delta, beta=beta, gamma=gamma,
                           a_grid=a_grid, options=options)
    K_grid = np.linspace(0.75 * Kstar, 1.25 * Kstar, n_K)

    # start the simulation at the no-aggregate-risk stationary (a,z) distribution
    # (K_0 ~ K*) -- avoids a long transient from zero and keeps K inside the grid.
    r0, w0 = _firm_prices(1.0, Kstar, L, alpha, delta)

    def _rf0(ap, a, zc, xp=np):
        c = (1.0 + r0) * a + w0 * zc - ap
        return xp.where(c > 0.0, _util(c, gamma, xp), -np.inf)

    sol0 = VFIProblem(a_grid=a_grid, z_grid=z_vals, P_z=P_z, return_fn=_rf0,
                      beta=beta, options=options).solve("numpy")
    mu0 = stationary_distribution(sol0.policy_aprime, P_z)

    Z_path = _simulate_markov_path(P_Z, T, np.random.default_rng(seed))
    b0 = np.zeros(nZ)
    b1 = np.ones(nZ)
    converged = False
    K_path = np.array([Kstar])
    R2 = np.ones(nZ)
    n_outer = 0
    for outer in range(max_outer):
        n_outer = outer + 1
        policy = _solve_household(b0, b1, z_vals=z_vals, P_z=P_z, Z_vals=Z_vals,
                                  P_Z=P_Z, K_grid=K_grid, a_grid=a_grid, L=L,
                                  alpha=alpha, delta=delta, beta=beta, gamma=gamma,
                                  options=options)
        K_path = ks_simulate(policy, a_grid, P_z, K_grid, Z_path, mu0=mu0)
        nb0, nb1, R2 = _regress_forecast(K_path, Z_path, nZ, burn_in)
        diff = max(float(np.max(np.abs(nb0 - b0))), float(np.max(np.abs(nb1 - b1))))
        b0 = damping * nb0 + (1.0 - damping) * b0
        b1 = damping * nb1 + (1.0 - damping) * b1
        if diff < tol:
            converged = True
            break
    return KSEquilibrium(b0=b0, b1=b1, r_squared=R2, K_path=K_path,
                         mean_K=float(np.mean(K_path[burn_in:])),
                         no_agg_risk_K=Kstar, n_outer=n_outer, converged=converged)


__all__ = ["ks_exog_transition", "ks_simulate", "krusell_smith", "KSEquilibrium"]
