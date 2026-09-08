"""Tests for Optimal Simple Rules (OSR) and Policy Regimes (Discretion & Commitment).

Covers:
- Validation 10: 3-equation NK model OSR optimal Taylor coefficient matches
  closed-form analytical optimum within 1e-4 tolerance.
- Validation 11: LQ commitment IRFs and timeless perspective (lambda_{-1} = 0),
  verifying multiplier dynamics, inflation undershoot (price-level targeting),
  and commitment vs discretion loss comparison.
- Dennis (2007) policy iteration convergence and properties under discretion.
- Continuous Blanchard-Kahn penalty surface near and beyond indeterminacy boundaries.
- LinearModel public properties (A_plus, A_0, A_minus, B_u) and .osr() method.
- Complete 6-method presentation contract (.to_frame(), .summary(), .plot(),
  .to_markdown(), .to_latex(), .to_typst()) on OSRResult and PolicyResult.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.dsge import (
    build_dynare,
    osr,
    discretionary_policy,
    lq_commitment,
    OSRResult,
    PolicyResult,
    LinearModel,
)


# ==============================================================================
# Model Fixtures
# ==============================================================================

def _nk_equations(lead, curr, lag, shocks, p):
    """Textbook 3-equation New Keynesian model with AR(1) cost-push shock.

    1. IS curve: y_t = E_t y_{t+1} - (1/sigma) * (r_t - E_t pi_{t+1})
    2. NKPC: pi_t = beta * E_t pi_{t+1} + kappa * y_t + u_t
    3. Taylor rule: r_t = phi_pi * pi_t + phi_y * y_t
    4. Cost-push shock AR(1): u_t = rho_u * u_{t-1} + eps_u
    """
    eq_is = lead.y - curr.y - (curr.r - lead.pi) / p.sigma
    eq_pc = p.beta * lead.pi + p.kappa * curr.y + curr.u - curr.pi
    eq_tr = p.phi_pi * curr.pi + p.phi_y * curr.y - curr.r
    eq_u = p.rho_u * lag.u + shocks.eps_u - curr.u
    return [eq_is, eq_pc, eq_tr, eq_u]


def _build_nk_model(
    *,
    beta: float = 0.99,
    sigma: float = 1.0,
    kappa: float = 0.5,
    phi_pi: float = 1.5,
    phi_y: float = 0.0,
    rho_u: float = 0.0,
) -> LinearModel:
    return build_dynare(
        _nk_equations,
        variables=["y", "pi", "r", "u"],
        states=["u"],
        shocks=["eps_u"],
        params={
            "beta": beta,
            "sigma": sigma,
            "kappa": kappa,
            "phi_pi": phi_pi,
            "phi_y": phi_y,
            "rho_u": rho_u,
        },
        guess={"y": 0.0, "pi": 0.0, "r": 0.0, "u": 0.0},
        strict=False,
    )


# ==============================================================================
# 1. Validation 10: OSR Analytical Optimum
# ==============================================================================

def test_validation_10_osr_analytical_optimum():
    """Validation 10: OSR optimal Taylor coefficient matches closed-form optimum within 1e-4.

    Analytical derivation:
    In the 3-equation NK model with i.i.d. cost-push shock (rho_u = 0), r_t = phi_pi * pi_t,
    and loss L = w_pi * Var(pi) + w_y * Var(y), the optimal coefficient satisfies:
        phi_pi^* = kappa * w_pi / w_y.
    With kappa = 0.5, w_pi = 1.0, w_y = 0.25:
        phi_pi^* = 0.5 * 1.0 / 0.25 = 2.0.
    """
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.0)
    res = model.osr(rule_params=["phi_pi"], weights={"pi": 1.0, "y": 0.25})

    assert isinstance(res, OSRResult)
    assert res.converged is True
    assert res.loss_opt < res.loss_initial

    opt_phi = res.optimal_params["phi_pi"]
    assert abs(opt_phi - 2.0) < 1e-4, f"Expected 2.0, got {opt_phi:.6f}"


def test_validation_10_osr_alternative_analytical_weights():
    """Verify analytical optimum under another calibration (kappa=0.8, w_pi=1.0, w_y=0.5 -> phi_pi^* = 1.6)."""
    model = _build_nk_model(kappa=0.8, phi_pi=2.5, rho_u=0.0)
    res = model.osr(rule_params=["phi_pi"], weights={"pi": 1.0, "y": 0.5})

    assert res.converged is True
    opt_phi = res.optimal_params["phi_pi"]
    assert abs(opt_phi - 1.6) < 1e-4, f"Expected 1.6, got {opt_phi:.6f}"


def test_osr_multi_parameter_with_bounds():
    """OSR optimization over multiple parameters with parameter bounds."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, phi_y=0.1, rho_u=0.5)
    bounds = {"phi_pi": (1.1, 4.0), "phi_y": (0.0, 1.5)}
    res = model.osr(
        rule_params=["phi_pi", "phi_y"],
        weights={"pi": 1.0, "y": 0.5},
        bounds=bounds,
    )

    assert res.converged is True
    assert res.loss_opt < res.loss_initial
    assert 1.1 <= res.optimal_params["phi_pi"] <= 4.0
    assert 0.0 <= res.optimal_params["phi_y"] <= 1.5
    assert len(res.rule_params) == 2


def test_osr_continuous_bk_penalty_surface():
    """Continuous BK penalty allows Nelder-Mead to contract away from indeterminacy."""
    # Start close to indeterminacy threshold phi_pi = 1.02
    model = _build_nk_model(kappa=0.5, phi_pi=1.02, rho_u=0.0)
    res = model.osr(rule_params=["phi_pi"], weights={"pi": 1.0, "y": 0.25})

    assert res.converged is True
    assert abs(res.optimal_params["phi_pi"] - 2.0) < 1e-4


# ==============================================================================
# 2. Validation 11: LQ Commitment & Timeless Perspective
# ==============================================================================

def test_validation_11_lq_commitment_irf_and_timeless_perspective():
    """Validation 11: LQ commitment reproduces published IRF sign pattern and timeless perspective.

    Under commitment following a cost-push shock:
    - Period 0: inflation rises (pi_0 > 0), output contracts (y_0 < 0).
    - Period 1+: inflation undershoots (pi_1 < 0) as part of price-level targeting commitment.
    - Timeless perspective sets initial lagged multipliers lambda_{-1} = 0.
    - Policy multipliers (mult_*) are accessible as model variables for IRFs and moments.
    """
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.5)
    res = lq_commitment(
        model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
    )

    assert isinstance(res, PolicyResult)
    assert res.regime == "commitment"
    assert len(res.multipliers) > 0
    assert any("mult" in m for m in res.multipliers)

    # Multipliers exposed on the augmented LinearModel
    aug_model = res.linear_model
    assert isinstance(aug_model, LinearModel)
    for mult in res.multipliers:
        assert mult in aug_model.variables

    # Impulse responses on cost-push shock
    irf = aug_model.irf("eps_u", horizon=12)

    # 1. Period 0: inflation positive, output negative
    assert irf["pi"].iloc[0] > 0.0
    assert irf["y"].iloc[0] < 0.0

    # 2. Period 1+: inflation undershoots below steady state (deflation)
    assert irf["pi"].iloc[1] < 0.0, "Commitment requires inflation undershoot in period 1"
    assert irf["pi"].iloc[2] < 0.0

    # 3. Output recovers towards steady state
    assert irf["y"].iloc[1] > irf["y"].iloc[0]

    # 4. Multiplier dynamics
    assert "mult_pi" in irf.columns
    assert irf["mult_pi"].iloc[0] > 0.0


# ==============================================================================
# 3. Discretion vs Commitment Comparison
# ==============================================================================

def test_discretionary_policy_convergence_and_properties():
    """Dennis (2007) policy iteration converges and yields time-consistent equilibrium."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.5)
    res_disc = discretionary_policy(
        model,
        target_vars=["pi", "y"],
        weights={"pi": 1.0, "y": 0.25},
        instruments="r",
    )

    assert isinstance(res_disc, PolicyResult)
    assert res_disc.regime == "discretion"
    assert res_disc.multipliers == ()
    assert res_disc.policy_rules is not None
    assert "r" in res_disc.policy_rules.index

    # In discretion, inflation does not undershoot after cost-push shock
    irf_disc = res_disc.linear_model.irf("eps_u", horizon=8)
    assert irf_disc["pi"].iloc[0] > 0.0
    assert irf_disc["pi"].iloc[1] >= 0.0, "Under discretion, inflation does not undershoot"


def test_commitment_achieves_lower_loss_than_discretion():
    """Theoretical property: Loss(Commitment) < Loss(Discretion)."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.5)
    weights = {"pi": 1.0, "y": 0.25}

    res_disc = discretionary_policy(model, target_vars=["pi", "y"], weights=weights, instruments="r")
    res_comm = lq_commitment(model, target_vars=["pi", "y"], weights=weights, instruments="r")

    assert res_comm.loss < res_disc.loss, (
        f"Commitment loss ({res_comm.loss:.4f}) should be strictly less than "
        f"discretion loss ({res_disc.loss:.4f})"
    )


# ==============================================================================
# 4. LinearModel Accessors & Direct Methods
# ==============================================================================

def test_linearmodel_first_order_properties():
    """LinearModel exposes public A_plus, A_0, A_minus, B_u properties."""
    model = _build_nk_model()
    assert model.A_plus is not None
    assert model.A_0 is not None
    assert model.A_minus is not None
    assert model.B_u is not None

    n = len(model.variables)
    assert model.A_plus.shape == (n, n)
    assert model.A_0.shape == (n, n)
    assert model.A_minus.shape == (n, n)
    assert model.B_u.shape[0] == n


def test_linearmodel_osr_method():
    """model.osr(...) convenience method dispatches cleanly."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.0)
    res = model.osr(["phi_pi"], {"pi": 1.0, "y": 0.25})
    assert isinstance(res, OSRResult)
    assert abs(res.optimal_params["phi_pi"] - 2.0) < 1e-4


# ==============================================================================
# 5. Presentation Contract Tests
# ==============================================================================

def test_osr_result_presentation_contract():
    """OSRResult satisfies all 6 presentation methods."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.0)
    res = model.osr(["phi_pi"], {"pi": 1.0, "y": 0.25})

    # 1. to_frame
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "var_initial" in df.columns
    assert "var_optimal" in df.columns

    # 2. summary
    s = res.summary()
    assert isinstance(s, str)
    assert "OPTIMAL SIMPLE RULES" in s
    assert "phi_pi" in s

    # 3. plot
    fig, ax = plt.subplots()
    ret_ax = res.plot(ax=ax)
    assert ret_ax is ax
    plt.close(fig)

    # 4. to_markdown
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    # 5. to_latex
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert r"\begin{tabular}" in ltx

    # 6. to_typst
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ

    # Properties
    assert res.loss_init == res.loss_initial
    assert res.loss_calib == res.loss_initial

    # Immutability
    with pytest.raises(Exception):
        res.loss_opt = 0.0


def test_policy_result_presentation_contract():
    """PolicyResult satisfies all 6 presentation methods for both regimes."""
    model = _build_nk_model(kappa=0.5, phi_pi=1.5, rho_u=0.5)

    for regime_fn, regime_name in [(discretionary_policy, "discretion"), (lq_commitment, "commitment")]:
        res = regime_fn(model, target_vars=["pi", "y"], weights={"pi": 1.0, "y": 0.25}, instruments="r")

        # 1. to_frame
        df = res.to_frame()
        assert isinstance(df, pd.DataFrame)

        # 2. summary
        s = res.summary()
        assert isinstance(s, str)
        assert f"OPTIMAL POLICY REGIME: {regime_name.upper()}" in s

        # 3. plot
        fig, ax = plt.subplots()
        ret_ax = res.plot(ax=ax, periods=10)
        assert ret_ax is ax
        plt.close(fig)

        # 4. to_markdown
        md = res.to_markdown()
        assert isinstance(md, str)
        assert "|" in md

        # 5. to_latex
        ltx = res.to_latex()
        assert isinstance(ltx, str)
        assert r"\begin{tabular}" in ltx

        # 6. to_typst
        typ = res.to_typst()
        assert isinstance(typ, str)
        assert "#table(" in typ

        # Properties
        assert res.augmented_model is res.linear_model

        # Immutability
        with pytest.raises(Exception):
            res.loss = 0.0


# ==============================================================================
# 6. Error & Boundary Handling
# ==============================================================================

def test_osr_unknown_parameter_raises_value_error():
    """model.osr raises ValueError for parameters not in model._params."""
    model = _build_nk_model()
    with pytest.raises(ValueError, match="Rule parameters.*not found"):
        model.osr(["nonexistent_param"], {"pi": 1.0})


def test_policy_unknown_instrument_raises_value_error():
    """lq_commitment and discretionary_policy raise ValueError for unknown instruments."""
    model = _build_nk_model()
    with pytest.raises(ValueError, match="Instrument.*not found"):
        lq_commitment(model, target_vars=["pi"], weights={"pi": 1.0}, instruments="unknown_inst")
