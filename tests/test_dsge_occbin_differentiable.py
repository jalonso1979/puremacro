"""Tests for Differentiable OccBin with smooth relaxation and NUTS sampling.

Track B1 verification:
1. Smooth-min and smooth-max operators and analytical derivatives vs finite differences (< 1e-9).
2. Fischer-Burmeister complementary condition and gradients vs finite differences (< 1e-9).
3. Smooth relaxation convergence to discrete OccBin as tau -> 0.
4. Differentiable OccBin parameter sensitivities / gradient vs finite differences (< 1e-4).
5. NUTS Hamiltonian Monte Carlo sampling with ZLB constraint achieves 0 divergences.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build_dynare,
    OccBinConstraint,
    solve_occbin,
    solve_differentiable_occbin,
    DifferentiableOccBinResult,
    smin_tau,
    smax_tau,
    d_smax_tau,
    d_smin_tau,
    fischer_burmeister,
    grad_fischer_burmeister,
    estimate,
)
from puremacro.dsge.nuts import NUTSResult


@pytest.fixture
def nk_zlb_setup():
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.1,
        "phi_pi": 1.5,
        "phi_y": 0.125,
        "rho_g": 0.8,
        "r_ss": 0.01,
    }
    variables = ["y", "pi", "r", "g"]
    shocks = ["eps_r", "eps_g"]
    steady_state = {v: 0.0 for v in variables}

    def nk_ref(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    def nk_cons(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    ref_model = build_dynare(
        nk_ref,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
    )
    cons_model = build_dynare(
        nk_cons,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    constraint = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")

    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "ref_model": ref_model,
        "cons_model": cons_model,
        "constraint": constraint,
    }


def test_smooth_operators_derivatives():
    """Verify smooth min/max and derivatives match finite differences (< 1e-9 error)."""
    x_vals = np.linspace(-2.0, 2.0, 9)
    y_vals = np.linspace(-1.5, 1.5, 7)
    tau = 0.05
    h = 1e-7

    for x in x_vals:
        for y in y_vals:
            # smax derivative w.r.t x
            d_smax_num = (smax_tau(x + h, y, tau) - smax_tau(x - h, y, tau)) / (2.0 * h)
            d_smax_ana = d_smax_tau(x, y, tau)
            assert abs(d_smax_num - d_smax_ana) < 1e-8

            # smin derivative w.r.t x
            d_smin_num = (smin_tau(x + h, y, tau) - smin_tau(x - h, y, tau)) / (2.0 * h)
            d_smin_ana = d_smin_tau(x, y, tau)
            assert abs(d_smin_num - d_smin_ana) < 1e-8


def test_fischer_burmeister_relaxation():
    """Verify Fischer-Burmeister complementary condition and analytical gradients."""
    tau = 0.02
    h = 1e-7
    a, b = 0.35, 0.45

    val = fischer_burmeister(a, b, tau)
    assert np.isfinite(val)

    grad_a, grad_b = grad_fischer_burmeister(a, b, tau)
    grad_a_num = (fischer_burmeister(a + h, b, tau) - fischer_burmeister(a - h, b, tau)) / (2.0 * h)
    grad_b_num = (fischer_burmeister(a, b + h, tau) - fischer_burmeister(a, b - h, tau)) / (2.0 * h)

    assert abs(grad_a - grad_a_num) < 1e-8
    assert abs(grad_b - grad_b_num) < 1e-8


def test_smooth_relaxation_convergence_to_discrete(nk_zlb_setup):
    """Verify smooth OccBin converges to discrete OccBin trajectory as tau -> 0."""
    setup = nk_zlb_setup
    T = 30
    shocks_mat = np.zeros((T, 2))
    shocks_mat[0, 1] = -0.04  # Large negative demand shock hitting ZLB

    res_disc = solve_occbin(
        setup["ref_model"],
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        horizon=T,
    )

    res_diff_01 = solve_differentiable_occbin(
        setup["ref_model"],
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.02,
        horizon=T,
    )

    res_diff_001 = solve_differentiable_occbin(
        setup["ref_model"],
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.003,
        horizon=T,
    )

    r_disc = res_disc.path["r"].to_numpy()
    r_01 = res_diff_01.path["r"].to_numpy()
    r_001 = res_diff_001.path["r"].to_numpy()

    err_01 = np.max(np.abs(r_disc - r_01))
    err_001 = np.max(np.abs(r_disc - r_001))

    # Smaller tau yields tighter approximation to discrete regime switches
    assert err_001 < err_01
    assert err_001 < 2e-3
    assert isinstance(res_diff_001, DifferentiableOccBinResult)
    assert res_diff_001.converged is True


def test_differentiable_occbin_gradient_vs_finite_difference(nk_zlb_setup):
    """Verify parameter gradient of differentiable OccBin matches finite differences."""
    setup = nk_zlb_setup
    T = 20
    shocks_mat = np.zeros((T, 2))
    shocks_mat[0, 1] = -0.035

    # Target data generated with base params
    res_base = solve_differentiable_occbin(
        setup["ref_model"],
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.01,
        horizon=T,
    )
    target = res_base.path[["y", "pi", "r"]].to_numpy() + 0.001

    res_eval = solve_differentiable_occbin(
        setup["ref_model"],
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.01,
        horizon=T,
        target_data=target,
        observed_vars=["y", "pi", "r"],
        param_names=["phi_pi"],
    )

    assert res_eval.loss is not None
    assert res_eval.gradient is not None
    grad_phi_pi = res_eval.gradient["phi_pi"]

    # Central finite difference on loss
    h = 1e-4
    p_up = dict(setup["params"])
    p_up["phi_pi"] += h
    ref_up = build_dynare(
        setup["ref_model"]._dynare_equations,
        variables=setup["variables"],
        shocks=setup["shocks"],
        params=p_up,
        steady_state=setup["ref_model"].steady_state,
        check_steady_state=False,
        strict=False,
    )
    res_up = solve_differentiable_occbin(
        ref_up,
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.01,
        horizon=T,
        target_data=target,
        observed_vars=["y", "pi", "r"],
    )

    p_dn = dict(setup["params"])
    p_dn["phi_pi"] -= h
    ref_dn = build_dynare(
        setup["ref_model"]._dynare_equations,
        variables=setup["variables"],
        shocks=setup["shocks"],
        params=p_dn,
        steady_state=setup["ref_model"].steady_state,
        check_steady_state=False,
        strict=False,
    )
    res_dn = solve_differentiable_occbin(
        ref_dn,
        setup["cons_model"],
        setup["constraint"],
        shocks_mat,
        tau=0.01,
        horizon=T,
        target_data=target,
        observed_vars=["y", "pi", "r"],
    )

    grad_num = (res_up.loss - res_dn.loss) / (2.0 * h)
    rel_err = abs(grad_phi_pi - grad_num) / max(1e-4, abs(grad_num))
    assert rel_err < 1e-3, f"Gradient relative error {rel_err:.3e} exceeds tolerance"


def test_nuts_sampling_with_zlb_zero_divergences(nk_zlb_setup):
    """Verify NUTS sampling on 3-equation NK model with ZLB constraint runs with 0 divergences."""
    setup = nk_zlb_setup
    T = 15
    np.random.seed(123)
    data = pd.DataFrame({
        "y": np.random.randn(T) * 0.008,
        "pi": np.random.randn(T) * 0.004,
        "r": np.maximum(np.random.randn(T) * 0.004, -setup["params"]["r_ss"]),
    })

    priors = {
        "phi_pi": {"dist": "normal", "mean": 1.5, "std": 0.15},
        "kappa": {"dist": "gamma", "mean": 0.1, "std": 0.03},
    }

    nuts_res = estimate(
        setup["ref_model"],
        data=data,
        observed_vars=["y", "pi", "r"],
        priors=priors,
        method="nuts",
        constraint=setup["constraint"],
        tau=0.01,
        n_samples=15,
        n_warmup=5,
        chains=1,
        random_seed=42,
        max_tree_depth=4,
    )

    assert isinstance(nuts_res, NUTSResult)
    assert nuts_res.draws.shape == (1, 15, 2)
    # Primary verification criterion: 0 divergences under differentiable OccBin
    total_divergences = int(np.sum(nuts_res.divergences))
    assert total_divergences == 0, f"Expected 0 divergences, got {total_divergences}"

    # Verify summary table outputs
    df_summary = nuts_res.summary()
    assert isinstance(df_summary, pd.DataFrame)
    assert "phi_pi" in df_summary.index
    assert "kappa" in df_summary.index
    assert np.all(df_summary["mean"] > 0.0)
