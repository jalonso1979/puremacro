"""Economic and numerical checks for solved stochastic HANK continuation."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from puremacro.models import (calibrate_one_asset_hank, calibrate_two_asset_equilibrium,
    solve_hank_stochastic_policy, solve_hank_stochastic_tree, monetary_scenario_tree,
    HANKDistributionBasis, solve_hank_equilibrium, solve_one_asset_hank)


@pytest.fixture(scope="module")
def one():
    return calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.)


@pytest.fixture(scope="module")
def two():
    return calibrate_two_asset_equilibrium(n_a=10, n_b=10, a_max=80.)


@pytest.fixture(scope="module")
def policies(one, two):
    return {"one": solve_hank_stochastic_policy(one, distribution_rank=12),
            "two": solve_hank_stochastic_policy(two, distribution_rank=12)}


@pytest.fixture(scope="module")
def full(one):
    return solve_hank_stochastic_policy(one, distribution_rank=None, shock_std=0.)


def chain(horizon, magnitude=-1e-4, rho=.7):
    return pd.DataFrame(dict(parent=[None]+[f"n{i}" for i in range(horizon-1)],
        conditional_probability=np.ones(horizon), monetary_shock=magnitude*rho**np.arange(horizon)),
        index=[f"n{i}" for i in range(horizon)])


def nonlinear_period(policy, cur, nxt):
    """Evaluate equations in levels, independently of the linearization."""
    kernel, basis = policy._kernel, policy.basis
    k, nm, nu = basis.rank, len(kernel.m), len(kernel.aggregate_names)
    ns = k+2
    d = basis.decode(cur[:k], check_positive=False)
    v = kernel.prices(cur[-nu:], cur[k], cur[k+1], distribution=d)
    h = kernel.backward(kernel.m+nxt[ns:ns+nm], v)
    dp = kernel.forward(d, h)
    future = nxt[-nu:]
    future_Y, future_pi = kernel.Y*np.exp(future[2]), np.expm1(future[0])
    future_gross = (1+v["nominal"])/(1+future_pi)
    continuation = future_Y/v["Y"]*future[0]/future_gross
    markets = kernel.markets(d, h, v, continuation)[0]
    return np.r_[nxt[:k]-basis.project(dp-basis.center), nxt[k]-(v["nominal"]-kernel.r),
        nxt[k+1]-policy.rho*cur[k+1], kernel.m+cur[ns:ns+nm]-h["m"], markets]


@pytest.mark.parametrize("name", ["one", "two"])
def test_joint_derivatives_against_nonlinear_period(policies, name):
    p = policies[name]
    n = len(p._F0)
    rng = np.random.default_rng(327)
    cur, nxt = rng.normal(size=(2, n))*.01
    epsilon = 2e-6
    difference = (nonlinear_period(p, epsilon*cur, epsilon*nxt)-
                  nonlinear_period(p, -epsilon*cur, -epsilon*nxt))/(2*epsilon)
    tangent = p._F0@cur+p._F1@nxt
    np.testing.assert_allclose(tangent, difference, atol=2e-7 if name == "one" else 2e-5, rtol=2e-4)


@pytest.mark.parametrize("name", ["one", "two"])
@pytest.mark.pyodide_smoke
def test_schur_policy_expectations_and_covariance(policies, name):
    p = policies[name]
    assert p.converged and p.diagnostics["spectral_radius"] < 1
    rng = np.random.default_rng(823)
    for _ in range(3):
        x = rng.normal(size=len(p.transition))*1e-4
        current = np.r_[x, p.feedback@x]
        xp = p.transition@x
        future = []
        for innovation in (-p.shock_std, p.shock_std):
            state = xp.copy()
            state[-1] += innovation
            future.append(np.r_[state, p.feedback@state])
        np.testing.assert_allclose(p._F0@current+p._F1@np.mean(future, axis=0), 0., atol=1e-10)
    covariance = p.covariance()
    Q = np.zeros_like(covariance)
    Q[-1, -1] = p.shock_std**2
    np.testing.assert_allclose(covariance, p.transition@covariance@p.transition.T+Q, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > -1e-13


def test_full_distribution_policy_matches_long_published_transition(one, full):
    innovations = np.r_[-1e-6, np.zeros(159)]
    simulation = full.simulate(innovations)
    reference = solve_one_asset_hank(one, horizon=160, magnitude=-1e-6, rho=.7, nonlinear=False)
    assert reference.converged and simulation.converged
    # Recursive controls are log deviations; the reference reports level changes.
    observed = simulation.paths.iloc[:20][["pi", "w", "Y"]].to_numpy(copy=True)
    observed[:, 1] *= one.calibration["w"]
    np.testing.assert_allclose(observed, reference.paths.iloc[:20][["pi", "w", "Y"]], atol=2e-9, rtol=3e-3)


def test_constraint_preserving_basis_and_no_silent_clipping(policies):
    b = policies["one"].basis
    coeff = np.arange(b.rank)*1e-8
    d = b.decode(coeff, check_positive=False)
    np.testing.assert_allclose(b.constraints.T@(d-b.center).ravel(), 0., atol=1e-13)
    assert d.sum() == pytest.approx(1., abs=1e-14)
    bad = b.decode(np.ones(b.rank), check_positive=False)
    assert bad.min() < 0
    with pytest.raises(ValueError, match="negative"):
        b.decode(np.ones(b.rank))
    with pytest.raises(ValueError, match="probability"):
        b.encode(bad)
    changed = b.center.copy()
    changed[0, 0] -= .001
    changed[1, 0] += .001
    with pytest.raises(ValueError, match="restrictions"):
        b.encode(changed)
    with pytest.raises(ValueError, match="orthonormal"):
        HANKDistributionBasis(b.center, 2*b.vectors, b.constraints)


@pytest.mark.parametrize("name", ["one", "two"])
def test_compression_preserves_market_observables(policies, name):
    p = policies[name]
    b = p.basis
    delta = np.random.default_rng(41).normal(size=b.center.size)*1e-6
    delta -= b.constraints@(b.constraints.T@delta)
    reconstructed = b.vectors@(b.vectors.T@delta)
    moments = p._kernel.moment_directions()
    np.testing.assert_allclose(moments.T@reconstructed, moments.T@delta, atol=2e-13)
    assert b.diagnostics["market_observable_directions"] == (2 if name == "one" else 5)


def test_snapshot_basis_is_evaluated_on_held_out_histories(one, full):
    training = full.simulate(np.random.default_rng(30).normal(0, 1e-7, 120))
    fitted = solve_hank_stochastic_policy(one, distribution_rank=12,
        snapshots=training.distributions, shock_std=0.)
    shocks = np.random.default_rng(91).normal(0, 1e-6, 50)
    reference, reduced = full.simulate(shocks), fitted.simulate(shocks)
    assert fitted.basis.diagnostics["method"] == "snapshot_svd"
    error = np.max(abs(reference.paths.to_numpy()-reduced.paths.to_numpy()))
    assert error < 1e-8
    assert fitted.diagnostics["system_dimension"] < full.diagnostics["system_dimension"]


@pytest.mark.parametrize("name", ["one", "two"])
@pytest.mark.pyodide_smoke
def test_held_out_audit_transports_actual_nonnegative_distribution(policies, name):
    p = policies[name]
    audit = p.audit(np.random.default_rng(99).normal(0, 1e-6, 10), tolerance=1e-5)
    assert audit.converged
    assert audit.distributions.shape[0] == 11
    assert audit.diagnostics["minimum_mass"] >= -1e-12
    assert audit.diagnostics["mass_error"] < 1e-12
    assert audit.diagnostics["household_budget_error"] < 1e-9
    assert audit.diagnostics["compression_l1_max"] > 0
    assert audit.diagnostics["euler_kkt_max"] > 0
    strict = p.audit(np.array([1e-4, -1e-4]), tolerance=1e-12)
    assert not strict.converged
    assert strict.diagnostics["market_residual_max_abs"] > 1e-12


@pytest.mark.parametrize("name", ["one", "two"])
@pytest.mark.pyodide_smoke
def test_nonlinear_terminal_tree_matches_independent_dense_solve(policies, name):
    p = policies[name]
    tree = monetary_scenario_tree(horizon=6, innovations=(-.0003, .0003), probabilities=(.3, .7))
    optimized = solve_hank_stochastic_tree(p, tree)
    dense = solve_hank_stochastic_tree(p, tree, method="hybr")
    assert optimized.converged and dense.converged
    np.testing.assert_allclose(optimized.paths, dense.paths, atol=3e-8, rtol=0)
    np.testing.assert_allclose(optimized.distributions, dense.distributions, atol=3e-8, rtol=0)
    assert optimized.diagnostics["terminal_consistency_error"] < 1e-8
    assert optimized.diagnostics["market_residual_max_abs"] < 1e-8
    assert not optimized.diagnostics["dense_jacobian_assembled"]
    assert dense.diagnostics["dense_jacobian_assembled"]
    assert optimized.metadata["terminal_policy_order"] == 1
    assert optimized.metadata["stochastic_tail_solved"]
    assert set(optimized.nodes.index) == set(optimized.to_dataframe().index)


def test_solved_tail_reduces_terminal_horizon_error(one, full):
    short = solve_hank_stochastic_tree(full, chain(6, -1e-5))
    long = solve_hank_stochastic_tree(full, chain(48, -1e-5))
    truncated = solve_hank_equilibrium(one, chain(6, -1e-5))
    assert short.converged and long.converged and truncated.converged
    columns = ["pi", "w", "Y"]
    gap = np.max(abs(short.paths[columns].iloc[0]-long.paths[columns].iloc[0]))
    old_gap = np.max(abs(truncated.paths[columns].iloc[0]-long.paths[columns].iloc[0]))
    assert gap < 2e-9
    assert gap < old_gap*.02


def test_consol_returns_fiscal_accounts_and_both_asset_markets(policies):
    policy = policies["two"]
    ss = policy.steady_state
    tree = chain(8, -.001)
    result = solve_hank_stochastic_tree(policy, tree)
    assert result.converged
    p = ss.calibration
    q_previous, nominal_previous = 1., p["r"]
    hh = ss._household
    for i, (_, row) in enumerate(result.paths.iterrows()):
        q, Y = 1+row.q, p["Y"]+row.Y
        ra, rb = p["r_a"]+row.r_a, p["r"]+row.r
        assert 1+ra == pytest.approx((q+p["coupon"])/q_previous, abs=1e-12)
        assert 1+rb == pytest.approx((1+nominal_previous)/(1+row.pi), abs=1e-12)
        tax = ss.aggregates["Tax"]+row.Tax
        assert tax == pytest.approx(rb*p["B"]+p["coupon"]*p["consol_units"], abs=1e-12)
        d = result.distributions[i]
        a, b, c, chi = [result.policies[k][i] for k in ("a", "b", "c", "chi")]
        assert np.sum(d*a) == pytest.approx(q*p["consol_units"], abs=1e-8)
        assert np.sum(d*b) == pytest.approx(p["B"], abs=1e-8)
        adj = p["mu"]/(p["mu"]-1)/(2*p["kappa"])*np.log1p(row.pi)**2*Y
        np.testing.assert_allclose(c+a+b+chi, (1+rb)*hh.b_grid[None, :, None]+
            (1+ra)*hh.a_grid[:, None, None]+(Y-adj-tax)*hh.s_grid, atol=1e-9)
        assert Y-np.sum(d*(c+chi))-adj == pytest.approx(0., abs=1e-8)
        q_previous, nominal_previous = q, p["r"]+tree.monetary_shock.iloc[i]+p["phi"]*row.pi


def test_audit_uses_previous_quoted_consol_price_despite_market_errors(policies):
    policy = policies["two"]
    audit = policy.audit(np.random.default_rng(9917).normal(0, 1e-4, 12))
    p = policy.steady_state.calibration
    previous_q = 1.
    for _, row in audit.paths.iterrows():
        q = 1.+row.q
        assert 1+p["r_a"]+row.r_a == pytest.approx((q+p["coupon"])/previous_q, abs=1e-12)
        previous_q = q
    assert abs(audit.paths.illiquid_mkt).max() > 1e-10


def test_input_ownership_and_shock_variance_scope(one):
    source = deepcopy(one)
    p = solve_hank_stochastic_policy(source, distribution_rank=8, shock_std=1e-5)
    p2 = solve_hank_stochastic_policy(source, distribution_rank=8, shock_std=2e-5)
    np.testing.assert_array_equal(p.feedback, p2.feedback)
    np.testing.assert_allclose(p2.covariance(), 4*p.covariance(), atol=1e-14)
    original = p.simulate([1e-6, 0.]).paths
    source.distribution[:] = 0
    source.calibration["B"] = 1000
    np.testing.assert_array_equal(p.simulate([1e-6, 0.]).paths, original)


@pytest.mark.parametrize("options", [dict(rho=1), dict(rho=np.nan), dict(shock_std=-1),
    dict(distribution_rank=0), dict(distribution_rank=True), dict(distribution_rank=1000),
    dict(max_system_size=2), dict(snapshots=np.zeros((2, 3)))])
def test_policy_validation(one, options):
    with pytest.raises(ValueError):
        solve_hank_stochastic_policy(one, **options)


@pytest.mark.parametrize("options", [dict(tolerance=0), dict(tolerance=np.nan), dict(max_iterations=0),
    dict(method="silent_fallback"), dict(preconditioner="invalid"), dict(initial=np.zeros(2)), dict(max_workspace_bytes=2)])
def test_terminal_tree_validation(policies, options):
    with pytest.raises(ValueError):
        solve_hank_stochastic_tree(policies["one"], chain(3), **options)


def test_failed_budget_retains_residuals(policies):
    out = solve_hank_stochastic_tree(policies["one"], chain(8, -.005), max_iterations=1, tolerance=1e-10)
    assert not out.converged
    assert out.diagnostics["optimizer_message"] != "converged"
    assert out.diagnostics["market_residual_max_abs"] > 1e-10
    assert np.isfinite(out.paths.to_numpy()).all()


def test_passive_monetary_rule_does_not_return_an_arbitrary_policy(one):
    bad = deepcopy(one)
    bad.calibration["phi"] = .5
    with pytest.raises(ValueError, match="determinacy"):
        solve_hank_stochastic_policy(bad, distribution_rank=8)


def test_inconsistent_steady_state_and_local_domain_fail_explicitly(one, policies):
    bad = deepcopy(one)
    bad.calibration["B"] *= 2
    with pytest.raises(ValueError, match="steady-state"):
        solve_hank_stochastic_policy(bad, distribution_rank=8)
    policy = deepcopy(policies["one"])
    policy.shock_std = 100.
    with pytest.raises(ValueError, match="terminal"):
        policy.audit(np.zeros(2))


def test_unpreconditioned_terminal_correction(policies):
    out = solve_hank_stochastic_tree(policies["one"], chain(4), preconditioner="none")
    assert out.converged
    assert out.diagnostics["preconditioner"] == "none"
