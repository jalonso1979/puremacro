"""The mode-search menu: Dynare's ``mode_compute`` for puremacro.

The 2.5.0 audit found the single bounded L-BFGS-B run returning
``initial_params`` bit-identically while reporting convergence. The fix then was
a finite penalty so the optimiser could see across the infeasible region; the
fix here is alternatives — a derivative-free simplex, Sims's csminwel, and
CMA-ES for a posterior that is not unimodal.

``find_mode`` minimises, matching ``scipy.optimize`` and ``estimate_dsge``'s
negative log posterior.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize as _scipy_minimize

from puremacro.dsge.mode import cmaes, csminwel, find_mode, mode_check


_METHODS = ["lbfgs", "simplex", "csminwel", "cmaes"]


def _quadratic(v):
    """Minimum at (3, -1)."""
    return (v[0] - 3.0) ** 2 + 2.0 * (v[1] + 1.0) ** 2


def _rosenbrock(v):
    return 100.0 * (v[1] - v[0] ** 2) ** 2 + (1.0 - v[0]) ** 2


def _rastrigin(v):
    v = np.asarray(v, dtype=float)
    return 10.0 * len(v) + np.sum(v ** 2 - 10.0 * np.cos(2.0 * np.pi * v))


@pytest.mark.parametrize("method", _METHODS)
def test_each_method_finds_a_known_maximum(method):
    res = find_mode(_quadratic, np.array([0.0, 0.0]), method=method, seed=0)
    np.testing.assert_allclose(res.x, [3.0, -1.0], atol=1e-3)
    assert res.fun == pytest.approx(0.0, abs=1e-5)
    assert res.method == method


def test_csminwel_on_rosenbrock():
    res = csminwel(_rosenbrock, np.array([-1.2, 1.0]))
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-3)


def test_cmaes_escapes_a_local_optimum_that_lbfgs_does_not():
    """2-D Rastrigin from (2.5, 2.5): a gradient method stops at the nearest
    well, a population method does not have to."""
    x0 = np.array([2.5, 2.5])
    grad = find_mode(_rastrigin, x0, method="lbfgs")
    glob = find_mode(_rastrigin, x0, method="cmaes", sigma0=2.0, seed=0)
    assert np.max(np.abs(grad.x)) > 0.5          # stuck in a side well
    np.testing.assert_allclose(glob.x, [0.0, 0.0], atol=1e-2)
    assert glob.fun < grad.fun


@pytest.mark.parametrize("method", _METHODS)
def test_bounds_are_respected(method):
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    res = find_mode(_quadratic, np.array([0.0, 0.0]), method=method,
                    bounds=bounds, seed=0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(res.x >= lo - 1e-9) and np.all(res.x <= hi + 1e-9)
    np.testing.assert_allclose(res.x, [1.0, -1.0], atol=1e-2)


def test_lbfgs_route_is_bit_identical_to_calling_scipy_directly():
    """The guard that protects the frozen SW07 parity fixture: routing the
    default path through find_mode must not perturb it by a single bit."""
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    options = {"maxiter": 100, "maxfun": 500 * 2}
    ref = _scipy_minimize(_quadratic, np.array([0.4, 0.7]), method="L-BFGS-B",
                          bounds=bounds, options=options)
    got = find_mode(_quadratic, np.array([0.4, 0.7]), method="lbfgs",
                    bounds=bounds, options=options)
    np.testing.assert_array_equal(got.x, ref.x)
    assert got.fun == ref.fun
    assert got.success == ref.success


def test_a_flat_objective_returns_the_starting_point_without_claiming_progress():
    res = find_mode(lambda v: 1.0, np.array([0.3, -0.2]), method="csminwel")
    np.testing.assert_allclose(res.x, [0.3, -0.2])
    assert res.fun == 1.0
    assert res.nit == 0


def test_csminwel_reports_its_perturbed_hessian_retries():
    res = csminwel(_rosenbrock, np.array([-1.2, 1.0]))
    assert hasattr(res, "n_hessian_resets")
    assert res.n_hessian_resets >= 0


def test_unknown_method_raises_listing_the_menu():
    with pytest.raises(ValueError, match="csminwel"):
        find_mode(_quadratic, np.array([0.0, 0.0]), method="newton_raphson")


def test_mode_check_slices_bottom_out_at_the_mode():
    mode = np.array([3.0, -1.0])
    res = mode_check(_quadratic, mode, ["a", "b"], n_points=21, width=1.0,
                     cov=np.eye(2))
    assert res.peaks_at_mode == {"a": True, "b": True}
    for name in ("a", "b"):
        sl = res.slices[name]
        assert len(sl) == 21
        assert sl["objective"].idxmin() == 10          # the centre point


def test_mode_check_flags_a_point_that_is_not_the_mode():
    """The single most common sign that a reported 'mode' is not one."""
    res = mode_check(_quadratic, np.array([3.0, 1.5]), ["a", "b"],
                     n_points=21, width=2.0, cov=np.eye(2))
    assert res.peaks_at_mode["a"] is True
    assert res.peaks_at_mode["b"] is False
    assert "b" in res.summary()


def test_mode_check_presentation_contract():
    res = mode_check(_quadratic, np.array([3.0, -1.0]), ["a", "b"], cov=np.eye(2))
    assert isinstance(res.summary(), str) and res.summary()
    for fn in (res.to_markdown, res.to_latex, res.to_typst):
        assert isinstance(fn(), str) and fn()
    import matplotlib
    matplotlib.use("Agg")
    assert res.plot() is not None


def test_estimate_dsge_mode_compute_defaults_to_lbfgs():
    """Flipping this default changes every existing posterior, so it is pinned
    until a release that refreshes the goldens with it."""
    import inspect
    from puremacro.dsge.estimate import estimate_dsge

    assert inspect.signature(estimate_dsge).parameters["mode_compute"].default == "lbfgs"
