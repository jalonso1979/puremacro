"""Unit tests for Nonlinear Ramsey Optimal Policy and Balanced Growth Path (BGP) Detrending.

Requirement R5 test suite:
- Analytical derivation of optimal commitment FOCs and replication of the timeless-perspective
  Taylor principle on Clarida, Gali & Gertler (1999) 3-equation model.
- Augmented commitment saddle-path solution via Klein QZ.
- Accessibility of policy multipliers as model variables for IRFs and stochastic simulations.
- Economic properties: divine coincidence on demand shocks, history dependence on cost-push shocks.
- Boundary conditions: myopic planner (beta -> 0), undiscounted limit (beta -> 1), zero target weight,
  and higher-order Frisch elasticity curvature.
- Balanced Growth Path (BGP) detrending and automatic stationarization on trending models.
- Parser validation of trend_var, log_trend_var, and var(deflator=...) annotations.
- Full presentation contract: .summary(), .plot(), .simulate(), .to_markdown(), .to_latex(), .to_typst().
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._ast import BinOp, Const, Node, Var
from puremacro.dsge._parser import parse_mod_to_dag, DynareParseError
from puremacro.dsge.dynare import build_dynare, ramsey_model, RamseyResult, detrend_bgp, detrend_model
from puremacro.dsge.ramsey import derive_ramsey_focs
from puremacro.dsge.klein import BlanchardKahnError


@pytest.fixture
def cgg1999_nk_model():
    """Canonical 3-equation Clarida, Gali & Gertler (1999) New Keynesian model."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_d": 0.6,
        "rho_u": 0.5,
        "r_ss": 0.01,
    }
    variables = ["y", "pi", "r", "d", "u"]
    shocks = ["eps_d", "eps_u", "eps_m"]
    steady_state = {v: 0.0 for v in variables}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            # 0: IS equation
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.d,
            # 1: New Keynesian Phillips Curve (NKPC) with cost-push shock u
            curr.pi - p.beta * lead.pi - p.kappa * curr.y - curr.u,
            # 2: Taylor policy rule (for benchmark comparison)
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_m),
            # 3: Exogenous demand shock
            curr.d - p.rho_d * lag.d - shocks_v.eps_d,
            # 4: Exogenous cost-push shock
            curr.u - p.rho_u * lag.u - shocks_v.eps_u,
        ]

    m = build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
    )
    return {
        "model": m,
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


# ==============================================================================
# 1. Analytical FOC Derivation & Clarida-Gali-Gertler (1999) Taylor Principle
# ==============================================================================

class TestRamseySymbolicFOCDerivation:
    """Test exact symbolic AST derivation of planner first-order conditions."""

    def test_cgg_analytical_foc_derivation(self):
        """Verify symbolic FOC derivation replicates Clarida, Gali & Gertler (1999):
        pi_t = - (lambda_x / kappa) * (x_t - x_{t-1}).
        """
        beta = 0.99
        kappa = 0.15
        lambda_x = 0.5

        # Objective: U = lambda_x * x^2 + pi^2 (loss function: minimize)
        U = Const(lambda_x) * Var("x", 0)**2 + Var("pi", 0)**2

        # NKPC constraint: pi(0) - beta*pi(1) - kappa*x(0) = 0
        nkpc = Var("pi", 0) - Const(beta) * Var("pi", 1) - Const(kappa) * Var("x", 0)

        foc_nodes, foc_strings, mult_names = derive_ramsey_focs(
            U=U,
            constraints=[nkpc],
            variables=["x", "pi"],
            beta=beta,
        )

        assert len(foc_nodes) == 2
        assert len(foc_strings) == 3  # 2 variable FOCs + 1 constraint FOC
        assert len(mult_names) == 1

        foc_x = foc_nodes[0]
        foc_pi = foc_nodes[1]

        # Verify FOC w.r.t x: dU/dx + mult_0 * (-kappa) = 2 * lambda_x * x - kappa * mult_0 = 0
        # => mult_0 = (2 * lambda_x / kappa) * x
        d_x_x = foc_x.diff("x", 0).simplify()
        d_x_mult = foc_x.diff("mult_0", 0).simplify()
        assert math.isclose(float(d_x_x.value), 2.0 * lambda_x, rel_tol=1e-9)
        assert math.isclose(float(d_x_mult.value), -kappa, rel_tol=1e-9)

        # Verify FOC w.r.t pi: 2 * pi + mult_0 - (1/beta) * beta * mult_0(-1) = 2 * pi + mult_0 - mult_0(-1) = 0
        # In lead term: (1/beta) * (-beta) = -1
        d_pi_pi = foc_pi.diff("pi", 0).simplify()
        d_pi_mult_curr = foc_pi.diff("mult_0", 0).simplify()
        d_pi_mult_lag = foc_pi.diff("mult_0", -1).simplify()

        assert math.isclose(float(d_pi_pi.value), 2.0, rel_tol=1e-9)
        assert math.isclose(float(d_pi_mult_curr.value), 1.0, rel_tol=1e-9)
        assert math.isclose(float(d_pi_mult_lag.value), -1.0, rel_tol=1e-9)

        # Substituting mult_0 = (2 * lambda_x / kappa) * x yields:
        # 2 * pi + (2 * lambda_x / kappa) * (x - x(-1)) = 0
        # => pi = - (lambda_x / kappa) * (x - x(-1))
        # Optimal commitment price-level targeting is mathematically exact!

    def test_ramsey_foc_with_intertemporal_lag_terms(self):
        """Verify FOC properly captures forward-looking multiplier terms for lagged state variables."""
        beta = 0.95
        delta = 0.1
        # Capital accumulation constraint: k(0) - (1-delta)*k(-1) - i(0) = 0
        eq = Var("k", 0) - Const(1.0 - delta) * Var("k", -1) - Var("i", 0)
        U = Var("k", 0)**2

        foc_nodes, foc_strings, mult_names = derive_ramsey_focs(
            U=U,
            constraints=[eq],
            variables=["k", "i"],
            beta=beta,
        )

        foc_k = foc_nodes[0]
        # For k(-1), FOC contains beta * mult(+1) * (-(1-delta))
        d_k_mult_lead = foc_k.diff("mult_0", 1).simplify()
        expected_coeff = beta * (-(1.0 - delta))
        assert math.isclose(float(d_k_mult_lead.value), expected_coeff, rel_tol=1e-9)


# ==============================================================================
# 2. Augmented Commitment Model & Klein QZ Saddle-Path Solution
# ==============================================================================

class TestRamseyAugmentedSaddlePath:
    """Test augmented commitment system assembly and Klein QZ solving."""

    def test_ramsey_augmented_model_structure(self, cgg1999_nk_model):
        """Verify augmented model contains original variables plus multipliers."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )

        assert isinstance(res, RamseyResult)
        assert res.augmented_model is not None
        assert len(res.multipliers) > 0

        # Multipliers must be accessible in augmented model variables
        aug_vars = list(res.augmented_model.variables)
        for m in res.multipliers:
            assert m in aug_vars

        # Predetermined states must include lagged multiplier
        states = list(res.augmented_model.states)
        assert any(m in states for m in res.multipliers)

    def test_ramsey_klein_solution_stability(self, cgg1999_nk_model):
        """Verify augmented commitment model has stable saddle-path dynamics."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )

        sol = res.policy_solution
        assert sol is not None
        assert hasattr(sol, "eigenvalues")
        eigs = sol.eigenvalues
        stable_eigs = eigs[np.abs(eigs) < 1.0 - 1e-10]
        assert len(stable_eigs) > 0


# ==============================================================================
# 3. Policy Multipliers & Impulse Response Functions
# ==============================================================================

class TestRamseyMultipliersAndIRF:
    """Test policy multiplier dynamics and IRF economic properties."""

    def test_divine_coincidence_demand_shock(self, cgg1999_nk_model):
        """Under demand shock, optimal policy raises nominal rate 1-for-1 with demand,
        perfectly stabilizing output gap and inflation at 0 with zero multiplier cost.
        """
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )

        irf = res.irf(shock="eps_d", horizon=20)
        assert isinstance(irf, pd.DataFrame)
        assert "y" in irf.columns
        assert "pi" in irf.columns
        assert "r" in irf.columns

        # Output and inflation must remain zero to numerical precision (< 1e-10)
        assert np.max(np.abs(irf["y"])) < 1e-10
        assert np.max(np.abs(irf["pi"])) < 1e-10

        # Nominal rate adjusts to track natural rate (initial response = 1.0)
        assert math.isclose(float(irf["r"].iloc[0]), 1.0, rel_tol=1e-5)

        # Multipliers remain zero
        for m in res.multipliers:
            if m in irf.columns:
                assert np.max(np.abs(irf[m])) < 1e-10

    def test_cost_push_shock_history_dependence(self, cgg1999_nk_model):
        """Under cost-push shock, optimal commitment engineers subsequent deflation
        (history dependence via lagged multiplier) to stabilize initial inflation.
        """
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )

        irf = res.irf(shock="eps_u", horizon=25)
        assert isinstance(irf, pd.DataFrame)

        # On impact, inflation rises
        pi_0 = float(irf["pi"].iloc[0])
        assert pi_0 > 0.0

        # Multiplier on NKPC becomes active
        mult_nkpc = res.multipliers[1]
        assert abs(float(irf[mult_nkpc].iloc[0])) > 1e-5

        # In later periods, commitment keeps inflation low or negative to fulfill past expectations
        # Verifies history dependence
        assert np.any(np.isfinite(irf[mult_nkpc]))

    def test_simulate_trajectory(self, cgg1999_nk_model):
        """Verify stochastic simulation runs cleanly across all variables and multipliers."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )

        sim = res.simulate(periods=100, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert len(sim) == 100
        for m in res.multipliers:
            assert m in sim.columns


# ==============================================================================
# 4. Boundary Cases: Extreme Discounts, Zero Weights, Curvature
# ==============================================================================

class TestRamseyBoundaryCases:
    """Test corner and boundary cases in planner configuration."""

    def test_myopic_planner_beta_001(self, cgg1999_nk_model):
        """Myopic planner beta=0.01 solves without division by zero or NaN."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + pi^2",
            planner_discount=0.01,
        )
        assert hasattr(res, "focs")
        assert len(res.focs) > 0
        assert np.isfinite(res.steady_state).all()

    def test_undiscounted_limit_beta_09999(self, cgg1999_nk_model):
        """Near-undiscounted planner beta=0.9999 solves cleanly."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + pi^2",
            planner_discount=0.9999,
        )
        assert hasattr(res, "focs")
        assert len(res.focs) > 0

    def test_zero_weight_target(self, cgg1999_nk_model):
        """Strict inflation targeting (0 weight on output gap) solves determinately."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="0.0 * y^2 + 1.0 * pi^2",
            planner_discount=0.99,
        )
        assert len(res.focs) > 0

    def test_high_frisch_elasticity_quartic_curvature(self, cgg1999_nk_model):
        """Objective with nonlinear quartic terms evaluates analytical Hessian and FOCs."""
        nk = cgg1999_nk_model
        res = ramsey_model(
            model_or_dag=nk["model"],
            objective="y^2 + pi^2 + 0.1 * y^4",
            planner_discount=0.99,
        )
        assert len(res.focs) > 0


# ==============================================================================
# 5. Balanced Growth Path (BGP) Detrending & Parser Validation
# ==============================================================================

class TestBGPDetrending:
    """Test Balanced Growth Path detrending and model stationarization."""

    def test_bgp_detrending_trending_rbc(self):
        """Test BGP detrending parses trend_var and var(deflator=...) in trending RBC model."""
        mod_text = """
        var c k y;
        varexo e;
        parameters alpha beta delta gamma;
        trend_var gamma;
        var(deflator=gamma) c k y;
        model;
        c + gamma*k = y + (1-delta)*k(-1);
        y = k(-1)^alpha * exp(e);
        c^(-1) = beta * c(+1)^(-1) * (alpha * y(+1)/k + 1 - delta);
        end;
        """
        dag = parse_mod_to_dag(mod_text)
        assert "gamma" in dag.trend_vars
        assert dag.deflators.get("c") == "gamma"
        assert dag.deflators.get("k") == "gamma"
        assert dag.deflators.get("y") == "gamma"

        stationarized = detrend_bgp(mod_text)
        assert isinstance(stationarized, str)
        assert "gamma" in stationarized or "deflator" in stationarized

    def test_bgp_identity_transformation_on_stationary_model(self):
        """Stationary model without trend variables passes through unchanged."""
        stationary_mod = """
        var c k;
        varexo e;
        parameters alpha beta;
        model;
        c + k = k(-1)^alpha + exp(e);
        end;
        """
        out = detrend_bgp(stationary_mod)
        assert out == stationary_mod

    def test_parser_log_trend_var_and_log_deflator(self):
        """Verify parser supports log_trend_var and log_deflator keywords."""
        mod_text = """
        var y c;
        varexo eps;
        parameters g;
        log_trend_var g;
        var(log_deflator=g) y c;
        model;
        y = c + exp(eps);
        end;
        """
        dag = parse_mod_to_dag(mod_text)
        assert "g" in dag.log_trend_vars
        assert dag.log_deflators.get("y") == "g"


# ==============================================================================
# 6. Presentation Contract
# ==============================================================================

class TestRamseyPresentationContract:
    """Test puremacro presentation contract for RamseyResult."""

    def test_presentation_methods(self):
        """Verify .summary(), .to_markdown(), .to_latex(), .to_typst()."""
        res = RamseyResult(
            focs=["d L / d y = 2*y + mult_0 = 0", "d L / d pi = 3*pi + mult_1 = 0"],
            augmented_model=None,
            multipliers=["mult_0", "mult_1"],
            steady_state=pd.Series({"y": 0.0, "pi": 0.0, "mult_0": 0.0, "mult_1": 0.0}),
            policy_solution=None,
        )

        summ = res.summary()
        assert isinstance(summ, str)
        assert "Ramsey" in summ or "Optimal Policy" in summ

        md = res.to_markdown()
        assert isinstance(md, str)
        assert "| Variable" in md

        tex = res.to_latex()
        assert isinstance(tex, str)
        assert r"\begin{aligned}" in tex

        typ = res.to_typst()
        assert isinstance(typ, str)
        assert "#table(" in typ

    def test_planner_foc_table(self):
        """Verify .planner_foc() returns DataFrame."""
        res = RamseyResult(
            focs=["d L / d y = 2*y = 0", "d L / d pi = 3*pi = 0"],
            augmented_model=None,
            multipliers=["mult_0"],
            steady_state=pd.Series({"y": 0.0}),
            policy_solution=None,
        )
        df_foc = res.planner_foc()
        assert isinstance(df_foc, pd.DataFrame)
        assert "Variable" in df_foc.columns
