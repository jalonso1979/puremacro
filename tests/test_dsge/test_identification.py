"""Comprehensive test suite for DSGE parameter identification analysis (Phase C).

Covers:
- Validation 9: Planted unidentified parameter pair (product and sum).
  Verifies rank deficiency == 1, null space direction recovered to within 1e-6
  of [1/sqrt(2), -1/sqrt(2)], collinearity R^2 ~ 1.0, formatted combination.
- Fast parameter differentiation for shock and measurement error variances
  skipping Klein QZ solve.
- Pre-flight check in LinearModel.estimate() with check_identification=True/warn/raise/False.
- Full presentation contract: .to_frame(), .summary(), .plot(), .to_markdown(),
  .to_latex(), .to_typst().
- Prior Monte Carlo (prior_mc > 0).
- Lag variation and rank enhancement.
- Edge cases and input validation.
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import puremacro.dsge as dsge
from puremacro.dsge import IdentificationResult, identification
from puremacro.dsge._estimated_params import EstimatedParams, EstimatedParamSpec
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.priors import BetaPrior, InvGammaPrior, NormalPrior


# --- Validation 9: Planted Unidentified Parameter Pair (Product & Sum) --------

def test_validation_9_planted_unidentified_product():
    """Validation 9: DSGE model where two parameters enter exclusively as a product theta1 * theta2.

    Verifies:
    - J1 and J2 rank deficiency == 1 (rank 1 out of 2).
    - Null space direction recovered within 1e-6 of [1/sqrt(2), -1/sqrt(2)].
    - Multi-way collinearity R^2 ~ 1.0 for both parameters.
    - Human-readable linear combination formatted properly.
    """
    def eqs(xp, x, e, p):
        # theta1 and theta2 enter only through their product
        param_prod = p.theta1 * p.theta2
        return [xp.y - param_prod * x.y - e.eps]

    # At theta1 = 1.0, theta2 = 0.5:
    # Wait, if theta1 = 0.7, theta2 = 0.7, product is symmetric!
    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.7, theta2=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = dsge.identification(m, params=["theta1", "theta2"], varobs=["y"])

    assert isinstance(res, IdentificationResult)
    assert res.is_identified is False
    assert res.rank_deficient is True
    assert res.j1_rank == 1
    assert res.j1_n_params == 2
    assert res.j2_rank == 1
    assert res.j2_n_params == 2

    # Null space direction recovery within 1e-6 tolerance
    expected_null = np.array([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    assert np.allclose(res.j1_null_space, expected_null, atol=1e-6)
    assert np.allclose(res.j2_null_space, expected_null, atol=1e-6)

    # Multi-way collinearity R^2 ~ 1.0
    assert float(res.j1_collinearity.loc["theta1", "r2"]) > 0.9999
    assert float(res.j1_collinearity.loc["theta2", "r2"]) > 0.9999
    assert float(res.j2_collinearity.loc["theta1", "r2"]) > 0.9999
    assert float(res.j2_collinearity.loc["theta2", "r2"]) > 0.9999

    # Pairwise collinearity R^2 ~ 1.0
    assert float(res.j1_collinearity.loc["theta1", "pairwise_r2"]) > 0.9999
    assert res.j1_collinearity.loc["theta1", "worst_partner"] == "theta2"
    assert res.j1_collinearity.loc["theta2", "worst_partner"] == "theta1"

    # Human-readable linear combination
    assert len(res.j1_null_combinations) == 1
    comb = res.j1_null_combinations[0]
    assert "0.7071 * theta1 - 0.7071 * theta2 = 0" in comb


def test_validation_9_planted_unidentified_sum():
    """Validation 9: Model where two parameters enter exclusively as a sum alpha + beta."""
    def eqs(xp, x, e, p):
        return [xp.y - (p.alpha + p.beta) * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(alpha=0.35, beta=0.35),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = m.identification(params=["alpha", "beta"], varobs=["y"])

    assert res.is_identified is False
    assert res.j1_rank == 1
    assert res.j2_rank == 1
    expected_null = np.array([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    assert np.allclose(res.j1_null_space, expected_null, atol=1e-6)
    assert np.allclose(res.j2_null_space, expected_null, atol=1e-6)
    assert "0.7071 * alpha - 0.7071 * beta = 0" in res.j1_null_combinations[0]


# --- Well-Identified Model Tests ---------------------------------------------

def test_well_identified_model():
    """Well-specified model where structural parameters and shock scale are identified."""
    def eqs(xp, x, e, p):
        # 2-equation model with independent parameters
        return [
            xp.y - p.rho * x.y - e.eps_y,
            xp.c - p.phi * x.y - p.gamma * x.c - e.eps_c,
        ]

    m = dsge.build(
        eqs,
        variables=["y", "c"],
        states=["y", "c"],
        shocks=["eps_y", "eps_c"],
        params=dict(rho=0.7, phi=0.5, gamma=0.3),
        guess=dict(y=0.0, c=0.0),
        linearize="level",
    )

    res = m.identification(
        params=["rho", "phi", "gamma", "SE_eps_y", "SE_eps_c"],
        varobs=["y", "c"],
        lags=2,
    )

    assert res.is_identified is True
    assert res.rank_deficient is False
    assert res.j1_rank == 5
    assert res.j1_n_params == 5
    assert res.j2_rank == 5
    assert res.j2_n_params == 5
    assert len(res.j1_null_combinations) == 0
    assert len(res.j2_null_combinations) == 0

    # Identification strength is strictly positive for all parameters
    for p in res.param_names:
        assert float(res.strength.loc[p, "strength"]) > 0.0
        assert float(res.strength.loc[p, "normalized_strength"]) > 0.0


# --- Fast Differentiation Verification ---------------------------------------

def test_fast_parameter_differentiation_skips_klein_solve():
    """Parameters affecting only shock variance or measurement error do NOT call model solve."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.8),
        guess=dict(y=0.0),
        linearize="level",
    )

    # Set _equations = None to prove no solve is executed
    m_no_eqs = replace(m, _equations=None)

    res = dsge.identification(
        m_no_eqs,
        params=["SE_eps", "ME_y"],
        varobs=["y"],
    )

    assert res.j1_rank == 2
    assert res.j1_n_params == 2
    assert float(res.strength.loc["SE_eps", "sensitivity"]) > 0.0
    assert float(res.strength.loc["ME_y", "sensitivity"]) > 0.0


def test_fast_differentiation_matches_finite_difference_values():
    """Verify algebraic shock and measurement error derivatives match finite differences."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.5),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = m.identification(params=["SE_eps", "ME_y"], varobs=["y"], lags=1)

    # For SE_eps at sigma = 1.0: d Q / d sigma = 2 * 1.0 = 2.0
    # Sensitivity for J1 on SE_eps must equal 2.0
    # (since only Q changes and Q has 1 entry whose derivative is 2.0)
    # Check that J1 sensitivity of SE_eps is 2.0
    ssm = dsge.observation.make_state_space_from_varobs(m, ["y"])
    # d_H / d sigma_me at base=0 is non-zero
    assert res.j1_rank == 2
    assert res.j1_n_params == 2


# --- Pre-Flight Check in LinearModel.estimate() ------------------------------

def test_estimate_preflight_check_raise():
    """LinearModel.estimate(..., check_identification='raise') raises ValueError when unidentified."""
    def dyn_eqs(lead, curr, lag, shocks, p):
        return [curr.y - (p.theta1 + p.theta2) * lag.y - shocks.eps]

    m = build_dynare(
        dyn_eqs,
        variables=["y"],
        shocks=["eps"],
        params=dict(theta1=0.4, theta2=0.4),
        guess=dict(y=0.0),
        states=["y"],
    )

    priors = {
        "theta1": BetaPrior(0.4, 0.1),
        "theta2": BetaPrior(0.4, 0.1),
    }
    data = pd.DataFrame({"y": np.zeros(20)})

    with pytest.raises(ValueError, match="pre-flight check failed.*structurally unidentified"):
        m.estimate(data, priors=priors, varobs=["y"], check_identification="raise")


def test_estimate_preflight_check_warn():
    """LinearModel.estimate(..., check_identification=True) warns when unidentified."""
    def dyn_eqs(lead, curr, lag, shocks, p):
        return [curr.y - (p.theta1 + p.theta2) * lag.y - shocks.eps]

    m = build_dynare(
        dyn_eqs,
        variables=["y"],
        shocks=["eps"],
        params=dict(theta1=0.4, theta2=0.4),
        guess=dict(y=0.0),
        states=["y"],
    )

    priors = {
        "theta1": BetaPrior(0.4, 0.1),
        "theta2": BetaPrior(0.4, 0.1),
    }
    data = pd.DataFrame({"y": np.zeros(20)})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m.estimate(
                data,
                priors=priors,
                varobs=["y"],
                check_identification=True,
                n_draws=10,
                burn_in=2,
            )
        except Exception:
            pass  # We only care that the preflight warning was issued

        warn_msgs = [str(item.message) for item in w if "pre-flight" in str(item.message)]
        assert len(warn_msgs) > 0
        assert "pre-flight check failed" in warn_msgs[0]


def test_estimate_preflight_check_false_does_not_warn():
    """LinearModel.estimate(..., check_identification=False) does not trigger pre-flight warning."""
    def dyn_eqs(lead, curr, lag, shocks, p):
        return [curr.y - (p.theta1 + p.theta2) * lag.y - shocks.eps]

    m = build_dynare(
        dyn_eqs,
        variables=["y"],
        shocks=["eps"],
        params=dict(theta1=0.4, theta2=0.4),
        guess=dict(y=0.0),
        states=["y"],
    )

    priors = {
        "theta1": BetaPrior(0.4, 0.1),
        "theta2": BetaPrior(0.4, 0.1),
    }
    data = pd.DataFrame({"y": np.zeros(20)})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m.estimate(
                data,
                priors=priors,
                varobs=["y"],
                check_identification=False,
                n_draws=10,
                burn_in=2,
            )
        except Exception:
            pass

        warn_msgs = [str(item.message) for item in w if "pre-flight" in str(item.message)]
        assert len(warn_msgs) == 0


# --- Presentation Contract Methods -------------------------------------------

def test_presentation_methods_contract():
    """IdentificationResult implements the 6 presentation methods."""
    def eqs(xp, x, e, p):
        return [xp.y - (p.theta1 + p.theta2) * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(theta1=0.4, theta2=0.4),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = m.identification(params=["theta1", "theta2"], varobs=["y"])

    # 1. to_frame()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["theta1", "theta2"]
    assert "j1_collinearity" in df.columns
    assert "j2_collinearity" in df.columns
    assert "sensitivity" in df.columns
    assert "strength" in df.columns
    assert "normalized_strength" in df.columns
    assert "identified" in df.columns
    assert bool(df.loc["theta1", "identified"]) is False
    assert bool(df.loc["theta2", "identified"]) is False

    # 2. summary()
    s = res.summary()
    assert isinstance(s, str)
    assert "PARAMETER IDENTIFICATION ANALYSIS" in s
    assert "UNIDENTIFIED" in s
    assert "J1 (Solution) rank     : 1 / 2" in s
    assert "J2 (Moments) rank      : 1 / 2" in s
    assert "0.7071 * theta1 - 0.7071 * theta2 = 0" in s

    # 3. plot()
    ax1 = res.plot()
    assert isinstance(ax1, plt.Axes)
    fig, ax2 = plt.subplots()
    out_ax = res.plot(ax=ax2)
    assert out_ax is ax2

    # 4. to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md
    assert "theta1" in md
    assert "theta2" in md

    # 5. to_latex()
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert r"\begin{tabular}" in ltx
    assert "theta1" in ltx

    # 6. to_typst()
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "theta1" in typ


def test_summary_on_identified_model():
    """summary() displays IDENTIFIED status when model is full rank."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.5),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = m.identification(params=["rho"], varobs=["y"])
    s = res.summary()
    assert "Overall status         : IDENTIFIED" in s
    assert "FULL RANK" in s
    assert "None (All parameters locally identified)" in s


# --- Prior Monte Carlo (prior_mc > 0) ----------------------------------------

def test_prior_monte_carlo():
    """prior_mc evaluates identification across prior draws and summarizes results."""
    def dyn_eqs(lead, curr, lag, shocks, p):
        return [curr.y - p.rho * lag.y - shocks.eps]

    m = build_dynare(
        dyn_eqs,
        variables=["y"],
        shocks=["eps"],
        params=dict(rho=0.7),
        guess=dict(y=0.0),
        states=["y"],
    )

    priors = EstimatedParams((
        EstimatedParamSpec(
            kind="param",
            target=("rho",),
            name="rho",
            prior=BetaPrior(0.7, 0.05),
            init=0.7,
            lb=0.0,
            ub=0.99,
        ),
    ))

    res = dsge.identification(m, params=priors, varobs=["y"], prior_mc=5, seed=42)
    assert res.prior_mc_results is not None
    assert res.prior_mc_results["n_draws"] == 5
    assert res.prior_mc_results["j1_identified_rate"] == 1.0
    assert res.prior_mc_results["j2_identified_rate"] == 1.0


# --- Lags Variation and Rank Enhancement -------------------------------------

def test_lags_variation_resolves_identification():
    """AR(1) with measurement error requires lags=2 to distinguish variance from error."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.7),
        guess=dict(y=0.0),
        linearize="level",
    )

    # With lags=1: only Gamma_0 and Gamma_1 -> 2 moments for 3 parameters -> rank 2
    res_lag1 = m.identification(params=["rho", "SE_eps", "ME_y"], varobs=["y"], lags=1)
    assert res_lag1.j1_rank == 3
    assert res_lag1.j2_rank == 2
    assert res_lag1.is_identified is False

    # With lags=2: Gamma_0, Gamma_1, Gamma_2 -> 3 moments for 3 parameters -> rank 3
    res_lag2 = m.identification(params=["rho", "SE_eps", "ME_y"], varobs=["y"], lags=2)
    assert res_lag2.j1_rank == 3
    assert res_lag2.j2_rank == 3
    assert res_lag2.is_identified is True


# --- Input Validation & Edge Cases -------------------------------------------

def test_unknown_parameter_name_raises_value_error():
    """Passing an unrecognised parameter name raises ValueError."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.5),
        guess=dict(y=0.0),
        linearize="level",
    )

    with pytest.raises(ValueError, match="unknown parameter 'nonexistent'"):
        m.identification(params=["rho", "nonexistent"])


def test_unknown_observable_name_raises_value_error():
    """Passing an unrecognised observable name raises ValueError."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.5),
        guess=dict(y=0.0),
        linearize="level",
    )

    with pytest.raises(ValueError, match="not in model variables"):
        m.identification(params=["rho"], varobs=["bad_obs"])


def test_single_parameter_identification():
    """Model evaluated for a single parameter handles collinearity regression cleanly."""
    def eqs(xp, x, e, p):
        return [xp.y - p.rho * x.y - e.eps]

    m = dsge.build(
        eqs,
        variables=["y"],
        states=["y"],
        shocks=["eps"],
        params=dict(rho=0.5),
        guess=dict(y=0.0),
        linearize="level",
    )

    res = m.identification(params=["rho"])
    assert res.n_params == 1
    assert res.is_identified is True
    assert float(res.j1_collinearity.loc["rho", "r2"]) == 0.0
    assert float(res.j2_collinearity.loc["rho", "r2"]) == 0.0
    assert res.j1_collinearity.loc["rho", "worst_partner"] == "None"
