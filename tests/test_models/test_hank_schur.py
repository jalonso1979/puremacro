"""Analytic generalized pencils and live economic closure comparisons."""
import numpy as np
import pytest
from scipy.linalg import block_diag

from puremacro.models._hank_schur import _solve_schur_policy
from puremacro.models.hank_reduction import solve_hank_reduced_policy
from puremacro.models.one_asset_hank import calibrate_one_asset_hank
from puremacro.models.hank_two_asset_equilibrium import calibrate_two_asset_equilibrium
from puremacro.models.hank_capital import calibrate_capital_hank


def rotating_pencil():
    angle = .6
    H = .8*np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    G = np.array([[.3, -.2], [1.1, .4]])
    coordinates = np.block([[np.eye(2), np.zeros((2, 2))], [-G, np.eye(2)]])
    # One unstable dynamic control and one static constraint (infinite root).
    F0 = block_diag(-H, -1.4, -1.)@coordinates
    F1 = np.diag([1., 1., 1., 0.])@coordinates
    rows, _ = np.linalg.qr(np.random.default_rng(18).normal(size=(4, 4)))
    return rows@F0, rows@F1, H, G


@pytest.mark.parametrize("output", ["real", "complex"])
def test_rotating_stable_block_and_infinite_root_recover_analytic_policy(output):
    F0, F1, expected_H, expected_G = rotating_pencil()
    originals = F0.copy(), F1.copy()
    H, G, report = _solve_schur_policy(F0, F1, 2, output=output)
    np.testing.assert_allclose(H, expected_H, atol=1e-10, rtol=0)
    np.testing.assert_allclose(G, expected_G, atol=1e-10, rtol=0)
    assert np.isrealobj(H) and np.isrealobj(G)
    assert report["schur_output"] == output
    assert report["spectral_radius"] == pytest.approx(.8)
    assert report["stable_roots"] == 2
    assert report["projected_equilibrium_residual_max_abs"] < 1e-12
    # Evaluate the original equations along a path, independently of Schur.
    x = np.array([.7, -.3])
    for _ in range(12):
        nxt = H@x
        np.testing.assert_allclose(F0@np.r_[x, G@x]+F1@np.r_[nxt, G@nxt], 0., atol=1e-12)
        x = nxt
    np.testing.assert_array_equal(F0, originals[0])
    np.testing.assert_array_equal(F1, originals[1])


@pytest.mark.parametrize("output", ["real", "complex"])
@pytest.mark.parametrize("root", [1., -1., 1.-5e-9, 1.+5e-9, .5])
def test_unit_circle_and_excess_stable_roots_are_rejected(output, root):
    with pytest.raises(ValueError, match="determinacy"):
        _solve_schur_policy(-np.diag([.7, root]), np.eye(2), 1, output=output)


@pytest.mark.parametrize("output", ["real", "complex"])
def test_too_few_stable_roots_and_singular_pencil_are_rejected(output):
    with pytest.raises(ValueError, match="determinacy"):
        _solve_schur_policy(-np.diag([1.2, 1.4]), np.eye(2), 1, output=output)
    with pytest.raises(ValueError, match="determinacy"):
        _solve_schur_policy(-np.diag([.7, 0.]), np.diag([1., 0.]), 1, output=output)


@pytest.mark.parametrize("output", ["real", "complex"])
def test_stable_subspace_must_identify_predetermined_states(output):
    # Two stable roots, but one is a control; the second state cannot be set.
    with pytest.raises(ValueError, match="well-conditioned state policy"):
        _solve_schur_policy(-np.diag([.7, 1.4, .6]), np.eye(3), 2, output=output)


@pytest.mark.parametrize("output", ["real", "complex"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_equations_cannot_produce_a_policy(output, bad):
    F0 = -np.diag([.7, 1.4])
    F0[0, 1] = bad
    with pytest.raises(ValueError):
        _solve_schur_policy(F0, np.eye(2), 1, output=output)


@pytest.mark.parametrize("output", [None, "auto", "REAL", ""])
def test_invalid_schur_output_fails_before_accessing_a_kernel(output):
    from puremacro.models.hank_reduction import solve_hank_reduced_kernel

    with pytest.raises(ValueError, match="schur_output"):
        solve_hank_reduced_kernel(object(), schur_output=output)
    with pytest.raises(ValueError, match="schur_output"):
        solve_hank_reduced_policy(object(), schur_output=output)


@pytest.fixture(scope="module", params=["one", "two", "capital"])
def comparison(request):
    if request.param == "one":
        ss = calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.)
    elif request.param == "two":
        ss = calibrate_two_asset_equilibrium(n_a=10, n_b=10)
    else:
        ss = calibrate_capital_hank(n_a=10, n_b=10, a_max=100., b_max=15.)
    kwargs = dict(distribution_rank=12, marginal_rank=24)
    real = solve_hank_reduced_policy(ss, **kwargs)
    complex_ = solve_hank_reduced_policy(ss, schur_output="complex", **kwargs)
    return real, complex_


def test_default_real_backend_preserves_live_equations_and_policies(comparison):
    real, complex_ = comparison
    assert real.diagnostics["schur_output"] == "real"
    assert complex_.diagnostics["schur_output"] == "complex"
    for field in ("_F0", "_F1"):
        np.testing.assert_array_equal(getattr(real, field), getattr(complex_, field))
    np.testing.assert_array_equal(real.basis.vectors, complex_.basis.vectors)
    np.testing.assert_array_equal(real.marginal_basis.vectors, complex_.marginal_basis.vectors)
    for field in ("transition", "feedback", "reduced_feedback", "_full_residual"):
        np.testing.assert_allclose(getattr(real, field), getattr(complex_, field), atol=1e-6, rtol=1e-8)
        assert not getattr(real, field).flags.writeable
    np.testing.assert_allclose(real.covariance(), complex_.covariance(), atol=1e-12, rtol=1e-8)
    innovations = np.random.default_rng(9917).normal(0., 1e-5, 30)
    np.testing.assert_allclose(real.simulate(innovations).paths, complex_.simulate(innovations).paths,
                               atol=1e-10, rtol=0)


def test_real_backend_keeps_full_grid_audit_and_failure_decisions(comparison):
    real, complex_ = comparison
    innovations = np.random.default_rng(9917).normal(0., 1e-6, 10)
    for tolerance in (1e-5, 1e-12):
        a, b = (p.audit(innovations, tolerance=tolerance) for p in comparison)
        assert a.converged == b.converged
        if tolerance == 1e-12:
            assert not a.converged
        np.testing.assert_allclose(a.paths, b.paths, atol=1e-8, rtol=0)
        assert a.diagnostics["mass_error"] < 1e-10
        assert a.diagnostics["household_budget_error"] < 1e-8
        assert a.diagnostics["full_marginal_relative_error"] > 0
        assert a.diagnostics["euler_kkt_max"] > 0
    assert not real.diagnostics["full_household_derivative_cache"]


@pytest.mark.slow
def test_sensitive_two_asset_rank_keeps_nonlinear_failures():
    ss = calibrate_two_asset_equilibrium(n_a=40, n_b=40, a_max=80., b_max=10.)
    coarse = solve_hank_reduced_policy(ss, distribution_rank=24, marginal_rank=48, krylov_steps=96)
    sensitive = solve_hank_reduced_policy(ss, distribution_rank=24, marginal_rank=80, krylov_steps=96)
    assert sensitive.diagnostics["projected_equilibrium_residual_max_abs"] < 1e-8
    assert sensitive.diagnostics["state_projection_condition"] < 1e12
    eps = np.random.default_rng(9917).normal(0., 1e-4, 48)
    np.testing.assert_allclose(sensitive.simulate(eps).paths, coarse.simulate(eps).paths, atol=1e-8, rtol=0)
    assert not sensitive.audit(eps, tolerance=1e-5).converged
    assert sensitive.audit(.25*eps, tolerance=1e-5).converged
