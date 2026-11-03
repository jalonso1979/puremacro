"""Independent dense comparisons and omitted-equation checks for HANK reduction."""
import numpy as np
import pytest

from puremacro.models.one_asset_hank import calibrate_one_asset_hank
from puremacro.models.hank_two_asset_equilibrium import calibrate_two_asset_equilibrium, _TwoAssetKernel
from puremacro.models._hank_one_asset_recursive import _OneAssetKernel
from puremacro.models.hank_recursive import solve_hank_stochastic_policy
from puremacro.models.hank_reduction import (HANKMarginalBasis, solve_hank_reduced_policy,
                                            solve_hank_reduced_kernel)


@pytest.fixture(scope="module")
def steady_states():
    return {"one": calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.),
            "two": calibrate_two_asset_equilibrium(n_a=10, n_b=10)}


@pytest.fixture(scope="module")
def policies(steady_states):
    return {name: solve_hank_reduced_policy(ss, distribution_rank=12, marginal_rank=40)
            for name, ss in steady_states.items()}


@pytest.mark.parametrize("name", ["one", "two"])
def test_control_refinement_matches_independent_dense_solver(steady_states, policies, name):
    ss = steady_states[name]
    reference = solve_hank_stochastic_policy(ss, distribution_rank=12)
    coarse = solve_hank_reduced_policy(ss, distribution_rank=12, marginal_rank=6)
    fine = policies[name]
    innovations = np.random.default_rng(99).normal(0, 1e-5, 30)
    expected = reference.simulate(innovations).paths.to_numpy()
    coarse_gap = np.max(abs(coarse.simulate(innovations).paths.to_numpy()-expected))
    fine_gap = np.max(abs(fine.simulate(innovations).paths.to_numpy()-expected))
    assert fine_gap < 5e-9
    assert fine_gap < coarse_gap*.01
    assert fine.diagnostics["system_dimension"] < reference.diagnostics["system_dimension"]
    assert fine.diagnostics["full_marginal_stationary_relative_rms_max"] < coarse.diagnostics["full_marginal_stationary_relative_rms_max"]*.01


@pytest.mark.parametrize("name", ["one", "two"])
def test_full_control_limit_matches_dense_policy(steady_states, name):
    ss = steady_states[name]
    expected = solve_hank_stochastic_policy(ss, distribution_rank=12)
    actual = solve_hank_reduced_policy(ss, distribution_rank=12, marginal_rank=None)
    np.testing.assert_allclose(actual.aggregate_feedback, expected.aggregate_feedback, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual.transition, expected.transition, atol=2e-6, rtol=1e-6)
    assert actual.diagnostics["full_marginal_relative_residual_max_abs"] < (1e-9 if name == "one" else 2e-6)


@pytest.mark.parametrize("name", ["one", "two"])
@pytest.mark.pyodide_smoke
def test_projected_equations_do_not_hide_omitted_controls(policies, name):
    p = policies[name]
    ns, rank, nm = len(p.transition), p.marginal_basis.rank, len(p._kernel.m)
    stack = np.vstack([np.eye(ns), p.reduced_feedback])
    projected = p._F0@stack+p._F1@stack@p.transition
    np.testing.assert_allclose(projected, 0., atol=1e-9)
    assert p._F0.shape == (ns+rank+len(p._kernel.aggregate_names),)*2
    assert p.feedback.shape == (nm+len(p._kernel.aggregate_names), ns)
    np.testing.assert_allclose(p.feedback[:nm], p.marginal_basis.vectors@p.reduced_feedback[:rank], atol=1e-12)
    assert p.diagnostics["full_marginal_residual_max_abs"] > p.diagnostics["projected_equilibrium_residual_max_abs"]
    assert "projected" in p.diagnostics["convergence_scope"]
    simulation = p.simulate(np.array([1e-4, -1e-4, 0.]))
    assert simulation.diagnostics["full_marginal_relative_error"] > 0
    assert simulation.diagnostics["linear_equilibrium_error"] < 1e-10


def test_two_asset_never_calls_or_allocates_legacy_cache(steady_states, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy dense derivative cache was called")
    monkeypatch.setattr(_TwoAssetKernel, "_tangent", forbidden)
    p = solve_hank_reduced_policy(steady_states["two"], distribution_rank=12, marginal_rank=24)
    assert p._kernel._derivatives is None
    assert not p.diagnostics["full_household_derivative_cache"]
    assert not p.diagnostics["full_grid_schur_allocated"]
    assert p.diagnostics["reduced_jacobian_bytes"] == p._F0.nbytes+p._F1.nbytes
    with pytest.raises(ValueError, match="directional kernel"):
        solve_hank_reduced_kernel(_TwoAssetKernel(steady_states["two"]))


@pytest.mark.parametrize("name", ["one", "two"])
@pytest.mark.pyodide_smoke
def test_full_grid_audit_checks_binding_households_and_marginal_errors(policies, name):
    p = policies[name]
    audit = p.audit(np.random.default_rng(99).normal(0, 1e-6, 10))
    assert audit.converged
    assert audit.diagnostics["mass_error"] < 1e-12
    assert audit.diagnostics["minimum_mass"] >= -1e-12
    assert audit.diagnostics["household_budget_error"] < 1e-9
    assert audit.diagnostics["policy_grid_exit_mass"] == 0.
    assert audit.diagnostics["full_marginal_relative_error"] > 0
    assert audit.diagnostics["euler_kkt_max"] > 0
    kernel = p._kernel
    zero = kernel.prices(np.zeros(len(kernel.aggregate_names)), 0., 0., distribution=kernel.d)
    household = kernel.backward(kernel.m, zero)
    if name == "one":
        borrowing = household["a"] == kernel.ss.asset_grid[0]
    else:
        borrowing = household["b"] == kernel.hh.b_grid[0]
    assert borrowing.any()
    assert kernel.d[borrowing].sum() > 0
    strict = p.audit(np.array([1e-4, -1e-4]), tolerance=1e-12)
    assert not strict.converged


def test_marginal_only_failure_is_not_accepted(steady_states):
    p = solve_hank_reduced_policy(steady_states["one"], marginal_rank=6)
    audit = p.audit(np.random.default_rng(99).normal(0, 1e-5, 10), tolerance=1e-7)
    assert audit.diagnostics["market_residual_max_abs"] < 1e-7
    assert audit.diagnostics["full_marginal_relative_error"] > 1e-7
    assert not audit.converged


def test_positive_marginal_domain_and_immutable_basis(policies):
    p = policies["one"]
    basis = p.marginal_basis
    np.testing.assert_allclose(basis.vectors.T@basis.vectors, np.eye(basis.rank), atol=1e-12)
    np.testing.assert_allclose(basis.decode(np.zeros(basis.rank)), basis.center)
    with pytest.raises(ValueError, match="positive domain"):
        basis.decode(-basis.project(basis.center)*1e5)
    with pytest.raises(ValueError, match="orthonormal"):
        HANKMarginalBasis(basis.center, 2*basis.vectors)
    with pytest.raises(ValueError):
        basis.vectors[0, 0] = 1.
    with pytest.raises(ValueError, match="positive domain"):
        p._forecast(-p.feedback[0]*1e7, p._kernel.r)


def test_snapshot_controls_have_independent_price_responses(steady_states, policies):
    reference = policies["one"]
    train = reference.simulate(np.random.default_rng(30).normal(0, 1e-6, 100))
    m = reference._kernel.m+train.states@reference.feedback[:len(reference._kernel.m)].T
    fitted = solve_hank_reduced_policy(steady_states["one"], distribution_rank=12,
        marginal_rank=24, marginal_snapshots=m)
    held_out = np.random.default_rng(91).normal(0, 1e-6, 30)
    np.testing.assert_allclose(fitted.simulate(held_out).paths, reference.simulate(held_out).paths, atol=2e-9, rtol=0)
    assert fitted.marginal_basis.diagnostics["method"] == "marginal_snapshot_svd"
    assert fitted.marginal_basis.diagnostics["price_response_directions"] >= 2


def test_generic_extra_state_and_shock_index(steady_states):
    class ExtendedKernel(_OneAssetKernel):
        aggregate_state_names = ("lag_nominal", "monetary_shock", "auxiliary")

        def prices_from_state(self, u, aggregate, distribution, *, lag_q=None):
            return self.prices(u, *aggregate[:2], distribution=distribution, lag_q=lag_q)

        def next_aggregate_state(self, prices, aggregate, rho):
            return np.r_[prices["nominal"]-self.r, rho*aggregate[1], .8*aggregate[2]]

        def linear_residual(self, cur, nxt, basis, rho):
            j = basis.rank+2
            base = super().linear_residual(np.delete(cur, j), np.delete(nxt, j), basis, rho)
            return np.insert(base, j, nxt[j]-.8*cur[j])

    p = solve_hank_reduced_kernel(ExtendedKernel(steady_states["one"]), marginal_rank=24)
    state = p.state(auxiliary=.01)
    assert state[-1] == .01 and p.shock_index == len(state)-2
    simulation = p.simulate(np.array([1e-6, 0.]), initial_state=state)
    assert simulation.states[0, p.shock_index] == 1e-6
    assert simulation.states[1, -1] == pytest.approx(.008)
    Q = np.zeros_like(p.transition)
    Q[p.shock_index, p.shock_index] = p.shock_std**2
    covariance = p.covariance()
    np.testing.assert_allclose(covariance, p.transition@covariance@p.transition.T+Q, atol=1e-14)
    assert p.audit(np.array([1e-6, 0.])).converged
    with pytest.raises(ValueError, match="unknown aggregate"):
        p.state(misspelled=.1)


@pytest.mark.parametrize("kwargs", [dict(marginal_rank=0), dict(marginal_rank=True),
    dict(marginal_rank=1), dict(marginal_rank=81), dict(krylov_steps=0),
    dict(rho=1.), dict(shock_std=-1.), dict(max_system_size=10)])
def test_invalid_controls_fail_before_returning_policy(steady_states, kwargs):
    with pytest.raises(ValueError):
        solve_hank_reduced_policy(steady_states["one"], **kwargs)


@pytest.mark.slow
def test_default_household_grid_uses_small_projected_system():
    ss = calibrate_one_asset_hank()
    p = solve_hank_reduced_policy(ss, distribution_rank=48, marginal_rank=96, krylov_steps=80)
    assert ss.distribution.shape == (500, 7)
    assert p.feedback.shape == (3503, 50)
    assert p._F0.shape == (149, 149)
    assert p.diagnostics["full_control_system_dimension"] == 3553
    assert p.audit(np.array([-1e-6, 0., 0.])).converged
