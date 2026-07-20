import numpy as np, pytest
from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import welfare as W
from puremacro.models.nested_dmp import estimation as E


def test_sigma_grid_nonneg_and_stochastic():
    p = NestedDMPParameters()
    grid, P = W.sigma_process(p, n_sigma=11)
    assert grid.shape == (11,) and P.shape == (11, 11)
    assert np.all(grid >= 0.0)                       # uncertainty is a level
    np.testing.assert_allclose(P.sum(axis=1), np.ones(11))


def test_regime_markov_symmetric_stationary():
    p = NestedDMPParameters()
    Pt = W.regime_transition(p)                       # 2x2, order [dove, hawk]
    np.testing.assert_allclose(Pt.sum(axis=1), [1, 1])
    w, V = np.linalg.eig(Pt.T)
    stat = np.real(V[:, int(np.argmin(np.abs(w - 1)))]); stat /= stat.sum()
    np.testing.assert_allclose(np.sort(stat), [0.5, 0.5], atol=1e-9)


def test_belief_transition_rows_sum_and_learning_direction():
    p = NestedDMPParameters()
    pi_grid = np.linspace(0.02, 0.98, 25)
    Ppi = W.belief_transition(p, pi_grid, sigma=1.0)  # {"dove":(25,25), "hawk":(25,25)}
    for tau in ("dove", "hawk"):
        np.testing.assert_allclose(Ppi[tau].sum(axis=1), np.ones(25))
    # Under a TRUE dove beliefs drift UP on average; under a TRUE hawk, DOWN.
    assert np.mean(Ppi["dove"] @ pi_grid - pi_grid) > 0.0
    assert np.mean(Ppi["hawk"] @ pi_grid - pi_grid) < 0.0


def test_recursive_reduces_to_static_ss_under_degenerate_sigma():
    # Degenerate sigma (innovations ~ 0) pins sigma==0 forever, so the recursive
    # J(.,pi,~0) collapses to the STATIC steady-state solve at sigma=0.
    from dataclasses import replace
    from puremacro.models.nested_dmp import dynamics as D
    p = replace(NestedDMPParameters(), sigma_innov_sd=1e-6, sigma_bar=0.0)
    sol = W.solve_recursive(p, n_sigma=5, n_pi=11)
    grid, erg = D._fixed_grid(p, sigma_ref=1.0)
    ss = D._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
    js = int(np.argmin(np.abs(sol.sigma_grid - 0.0)))
    jp = int(np.argmin(np.abs(sol.pi_grid - p.prior_pi0)))
    assert sol.theta[jp, js] == pytest.approx(ss["theta"], rel=0.05)

def test_theta_monotone_in_belief_at_high_sigma():
    p = NestedDMPParameters()
    sol = W.solve_recursive(p, n_sigma=7, n_pi=15)
    js = int(np.argmax(sol.sigma_grid))      # highest-uncertainty node
    th = sol.theta[:, js]                     # pi_grid ascending (hawkish -> dovish)
    assert th[-1] > th[0]

def test_phi_sigma_raises_tightness_at_high_sigma():
    from dataclasses import replace
    p0 = NestedDMPParameters()
    s0 = W.solve_recursive(p0, n_sigma=7, n_pi=15)
    s1 = W.solve_recursive(replace(p0, phi_sigma=0.01), n_sigma=7, n_pi=15)
    js = int(np.argmax(s0.sigma_grid))
    assert np.all(s1.theta[:, js] >= s0.theta[:, js] - 1e-9)


@pytest.mark.slow
def test_ergodic_welfare_finite_and_employment_sensible():
    p = E.default_baseline()
    sol = W.solve_recursive(p, n_sigma=7, n_pi=11)
    res = W.ergodic_welfare(p, sol, T=2500, seed=0, burn=400)
    assert np.isfinite(res.welfare) and res.welfare > 0.0
    assert 0.0 < res.mean_urate < 0.5


def test_ergodic_welfare_rejects_uncalibrated_unstable_params():
    # Raw defaults give f(theta)>1 (unstable forward LoM); welfare must REFUSE to
    # run on it rather than silently clip -- the caller must calibrate mu first.
    p = NestedDMPParameters()
    sol = W.solve_recursive(p, n_sigma=5, n_pi=9)
    with pytest.raises(ValueError):
        W.ergodic_welfare(p, sol, T=500, seed=0, burn=50)


@pytest.mark.slow
def test_optimal_phi_sigma_beats_status_quo_and_cev_nonneg():
    p = E.default_baseline()
    rep = W.optimal_phi_sigma(p, grid=np.linspace(0.0, 0.01, 4),
                              n_sigma=7, n_pi=11, T=2500, seed=0, burn=400)
    assert rep.welfare_curve.shape == (4,)
    assert rep.phi_sigma_star >= 0.0
    assert rep.welfare_star >= rep.welfare_zero - 1e-12         # optimum >= status quo
    assert rep.cev_pct == pytest.approx(100.0 * (rep.welfare_star / rep.welfare_zero - 1.0), rel=1e-6)
    assert rep.cev_pct >= 0.0

@pytest.mark.slow
def test_optimal_phi_sigma_lucas_robust_curve_per_grid_point():
    p = E.default_baseline()
    grid = np.linspace(0.0, 0.01, 4)
    rep = W.optimal_phi_sigma(p, grid=grid, n_sigma=7, n_pi=11, T=2000, seed=0, burn=300)
    assert rep.phi_sigma_grid.shape == (4,)
    np.testing.assert_allclose(rep.phi_sigma_grid, grid)
    assert np.isfinite(rep.welfare_zero)        # status-quo phi_sigma=0 is always feasible
