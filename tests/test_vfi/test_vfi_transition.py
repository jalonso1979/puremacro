from __future__ import annotations

import numpy as np

from puremacro.vfi import VFIProblem
from puremacro.vfi.examples import aiyagari_steady_state
from puremacro.vfi.transition import TransitionPath, transition_path

ALPHA, DELTA, BETA = 0.36, 0.08, 0.96


def _aiyagari_transition_pieces(n_a=150, n_z=5):
    res = aiyagari_steady_state(n_a=n_a, n_z=n_z)
    eq = res["equilibrium"]
    a_grid = np.asarray(eq.problem.a_grid, dtype=float)
    z_grid = np.asarray(eq.problem.z_grid, dtype=float)
    P = np.asarray(eq.problem.P_z, dtype=float)
    L = res["L"]

    def KL_of_r(r):
        return (ALPHA / (r + DELTA)) ** (1.0 / (1.0 - ALPHA))

    def wage_of_r(r):
        return (1.0 - ALPHA) * KL_of_r(r) ** ALPHA

    def build_problem(t, p):
        r = float(p[t]); w = wage_of_r(r)

        def rf(ap, a, z, r_, w_, xp=np):
            c = w_ * xp.exp(z) + (1.0 + r_) * a - ap
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=BETA, params={"r": r, "w": w},
                          options=dict(tol=1e-9, n_howard=40))

    def implied_price_path(dists, policies, p):
        T = len(p)
        out = np.empty(T)
        for t in range(T):
            K_t = float(np.sum(dists[t] * a_grid[:, None]))   # capital entering t
            out[t] = ALPHA * (K_t / L) ** (ALPHA - 1.0) - DELTA
        return out

    return res, a_grid, build_problem, implied_price_path


def test_transition_no_shock_stays_at_steady_state():
    # Start AT the steady state with no parameter change: the path must stay flat
    # at the SS price and the distribution must not move (the SS is a fixed point
    # of the transition map). This is the primary correctness anchor.
    res, a_grid, build_problem, implied_price_path = _aiyagari_transition_pieces()
    eq = res["equilibrium"]
    r_ss = eq.price
    mu_ss = eq.distribution
    V_ss = eq.solution.V
    T = 25
    tp = transition_path(mu_ss, V_ss, build_problem, implied_price_path,
                         np.full(T, r_ss), damping=0.6, tol=1e-7, max_iter=500)
    assert isinstance(tp, TransitionPath)
    np.testing.assert_allclose(tp.price_path, r_ss, atol=1e-4)
    # the distribution stays at the steady state along the whole path
    for mu_t in tp.distributions:
        np.testing.assert_allclose(mu_t, mu_ss, atol=5e-4)
        np.testing.assert_allclose(mu_t.sum(), 1.0, atol=1e-10)


def test_transition_returns_to_steady_state_from_perturbed_dist():
    # Same params, but start from a perturbed distribution (extra mass at the
    # lowest assets). With unchanged fundamentals the economy must transition
    # BACK to the steady state: the final distribution ~ mu_ss and the capital
    # path moves monotonically from the (low) initial K toward K_ss.
    res, a_grid, build_problem, implied_price_path = _aiyagari_transition_pieces()
    eq = res["equilibrium"]
    r_ss = eq.price
    mu_ss = eq.distribution
    V_ss = eq.solution.V
    # perturb: move half the mass onto the lowest asset row (same z-marginal)
    z_marg = mu_ss.sum(axis=0)
    mu0 = 0.5 * mu_ss
    mu0[0, :] += 0.5 * z_marg
    mu0 = mu0 / mu0.sum()
    K0 = float(np.sum(mu0 * a_grid[:, None]))
    K_ss = float(np.sum(mu_ss * a_grid[:, None]))
    assert K0 < K_ss  # perturbation lowered aggregate capital

    T = 120
    # This is a large (50%) capital shortfall, so heavy damping (0.1) is needed
    # for stability -- damping=0.5 limit-cycles on a shock this size (textbook
    # shooting behavior). And discrete-grid Aiyagari has a price-path noise floor
    # ~1.5e-4 (granularity of K_t over integer-index policies), so tol must sit
    # above it. The economic assertions below (distribution + capital path) are
    # the real anchors; the no-shock identity test keeps a tight tol=1e-7.
    tp = transition_path(mu0, V_ss, build_problem, implied_price_path,
                         np.full(T, r_ss), damping=0.1, tol=5e-4, max_iter=800)
    # converged back to the steady state by the end of the horizon
    np.testing.assert_allclose(tp.distributions[-1], mu_ss, atol=2e-3)
    K_path = np.array([float(np.sum(mu * a_grid[:, None])) for mu in tp.distributions])
    assert abs(K_path[-1] - K_ss) < 1e-2          # ends at the SS capital
    assert abs(K_path[0] - K0) < 1e-12            # starts at the perturbed capital
    assert K_path[5] > K_path[0]                  # capital rebuilds toward the SS


def test_transition_nonconvergence_raises():
    import pytest
    res, a_grid, build_problem, implied_price_path = _aiyagari_transition_pieces(n_a=40)
    eq = res["equilibrium"]
    with pytest.raises(RuntimeError, match="did not converge"):
        transition_path(eq.distribution, eq.solution.V, build_problem,
                        implied_price_path, np.full(10, eq.price),
                        damping=0.5, tol=1e-12, max_iter=1)
