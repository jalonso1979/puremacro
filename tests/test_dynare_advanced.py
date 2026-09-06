"""Advanced tests for puremacro Dynare-compatible engine.

Tests:
1. Full Smets-Wouters (2007) canonical .mod file parsing and solving (40 vars, 7 shocks).
2. Hansen (1985) indivisible labor RBC model at 1st and 2nd order.
3. Explicit predetermined_variables block.
4. Analytical steady_state_model block.
5. Multi-period leads and lags auxiliary variable generation (|offset| >= 2).
6. StochSimulResult reporting and multi-format exports.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build_dynare,
    parse_mod,
    load_mod,
    LinearModel,
    PrunedDSGESolution,
    DynareDR,
    Dynare2ndDR,
    TheoreticalMomentsResult,
    StochSimulResult,
)


def test_sw07_pfeifer_full_solve():
    """Verify full end-to-end parse and solve of Pfeifer's Smets-Wouters 2007 .mod file."""
    ref_file = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    assert ref_file.is_file(), "Reference file sw07_pfeifer.mod not found"

    m = load_mod(ref_file)
    assert isinstance(m, LinearModel)

    # 40 endogenous variables and 7 exogenous innovations
    assert len(m.variables) == 40
    assert len(m.shocks) == 7
    assert m.n_states == 15
    assert m.n_controls == 25

    # Decision rules (oo_.dr parity)
    dr = m.decision_rules()
    assert isinstance(dr, DynareDR)
    assert dr.ghx.shape == (40, 15)
    assert dr.ghu.shape == (40, 7)

    # Consolidated stoch_simul routine
    res = m.stoch_simul(irf=20)
    assert isinstance(res, StochSimulResult)
    assert res.order == 1
    assert len(res.irfs) == 40 * 7  # 280 IRF series

    # Subscript access
    assert "labobs_ea" in res.irfs
    irf_lab_ea = res["labobs_ea"]
    assert len(irf_lab_ea) == 21
    assert isinstance(irf_lab_ea, pd.Series)

    # DataFrame retrieval for specific shock
    df_ea = res.to_frame(shock="ea")
    assert df_ea.shape == (21, 40)
    assert "labobs" in df_ea.columns
    assert "robs" in df_ea.columns

    # The model is determinate: a unique stable solution with nonzero,
    # equation-satisfying decision rules (before 2.3.1 the engine dropped the
    # lead-of-state Jacobian block, the BK check failed and every rule was 0).
    assert m.solution.eu == (1, 1)
    assert np.abs(dr.ghu.to_numpy()).max() > 1.0
    assert np.abs(res.irfs["robs_em"].to_numpy()).max() > 0.05
    states = list(m.states)
    F_full = dr.ghx.to_numpy()
    L_full = dr.ghu.to_numpy()
    s_idx = [list(m.variables).index(v) for v in states]
    resid_x = m._A_plus @ F_full @ m.solution.G + m._A_0 @ F_full + m._A_minus[:, s_idx]
    resid_u = m._A_plus @ F_full @ m.solution.N + m._A_0 @ L_full + m._B_u
    assert np.abs(resid_x).max() < 1e-9
    assert np.abs(resid_u).max() < 1e-9

    # Formatted reports
    summ = res.summary()
    assert "DYNARE STOCH_SIMUL REPORT (Order 1)" in summ
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table" in res.to_typst()
    md = res.to_markdown()
    # column padding depends on the numbers' width, so check the headers only
    assert "Mean" in md and "Std.Dev." in md and md.startswith("|")


def test_hansen_1985_rbc_1st_and_2nd_order():
    """Test Hansen (1985) indivisible labor model at 1st and 2nd order with stoch_simul."""
    mod_text = """
    var c k h y z;
    varexo eps;
    parameters alpha beta delta gamma rho A;

    alpha = 0.36;
    beta = 0.99;
    delta = 0.025;
    gamma = 1.0;
    rho = 0.95;
    A = 1.72;

    model;
      c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(z(+1)) * k^(alpha - 1.0) * h(+1)^(1.0 - alpha) + 1.0 - delta);
      A * c^gamma = (1.0 - alpha) * exp(z) * k(-1)^alpha * h^(-alpha);
      y = exp(z) * k(-1)^alpha * h^(1.0 - alpha);
      k = y - c + (1.0 - delta) * k(-1);
      z = rho * z(-1) + eps;
    end;

    steady_state_model;
      z = 0.0;
      h = 0.5003959968320251;
      k = 19.009670393128232;
      c = 1.378254441521983;
      y = 1.8534962013501888;
    end;

    shocks;
      var eps; stderr 0.007;
    end;
    """
    # 1st order
    m1 = load_mod(mod_text, order=1)
    assert isinstance(m1, LinearModel)
    assert m1.states == ("k", "z")

    res1 = m1.stoch_simul(irf=15, periods=120, seed=42)
    assert isinstance(res1, StochSimulResult)
    assert res1.order == 1
    assert res1.simulated_moments is not None
    assert list(res1.simulated_moments.columns) == ["Mean", "Std.Dev.", "Variance", "Skewness", "Kurtosis"]
    assert len(res1.simulated_moments) == 5

    # Theoretical moments
    theo1 = res1.theoretical_moments
    assert isinstance(theo1, TheoreticalMomentsResult)
    assert theo1.moments.loc["z", "Mean"] == pytest.approx(0.0, abs=1e-6)

    # 2nd order
    m2 = load_mod(mod_text, order=2)
    assert isinstance(m2, PrunedDSGESolution)
    assert m2.state_names == ("k", "z")
    assert m2.is_stable

    res2 = m2.stoch_simul(irf=15, periods=120, seed=42)
    assert isinstance(res2, StochSimulResult)
    assert res2.order == 2
    assert isinstance(res2.dr, Dynare2ndDR)
    assert res2.simulated_moments is not None

    # Risk correction on capital is positive (precautionary capital accumulation)
    assert m2.H_sigmasigma[0] > 0.0

    # Check 2nd-order IRF
    irf_y_eps = res2["y_eps"]
    assert len(irf_y_eps) == 16
    assert irf_y_eps.iloc[1] > 0.0


def test_predetermined_variables_block():
    """Verify predetermined_variables declaration overrides automatic detection."""
    mod_text = """
    var y c k a;
    varexo e;
    parameters alpha beta delta rho;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.90;

    predetermined_variables k a;

    model;
      c^(-1) = beta * c(+1)^(-1) * (alpha * exp(a(+1)) * k^(alpha-1) + 1 - delta);
      y = exp(a) * k(-1)^alpha;
      k = y - c + (1-delta)*k(-1);
      a = rho * a(-1) + e;
    end;

    initval;
      k = 25.0;
      c = 2.0;
      y = 2.6;
      a = 0.0;
    end;
    """
    parsed = parse_mod(mod_text)
    assert parsed["predetermined_variables"] == ["k", "a"]

    m = load_mod(mod_text)
    assert isinstance(m, LinearModel)
    assert m.states == ("k", "a")
    assert set(m.controls) == {"y", "c"}


def test_steady_state_model_block():
    """Verify analytical steady_state_model block is parsed, evaluated, and verified."""
    mod_text = """
    var c k y a;
    varexo e;
    parameters alpha beta delta rho;
    alpha = 0.30;
    beta = 0.99;
    delta = 0.025;
    rho = 0.85;

    model;
      c^(-1) = beta * c(+1)^(-1) * (alpha * exp(a(+1)) * k^(alpha-1) + 1 - delta);
      y = exp(a) * k(-1)^alpha;
      k = y - c + (1-delta)*k(-1);
      a = rho * a(-1) + e;
    end;

    steady_state_model;
      a = 0.0;
      k = (alpha / (1/beta - (1-delta)))^(1/(1-alpha));
      y = k^alpha;
      c = y - delta * k;
    end;
    """
    parsed = parse_mod(mod_text)
    assert parsed["steady_state"] is not None
    assert parsed["steady_state"]["a"] == 0.0
    assert parsed["steady_state"]["k"] > 0.0

    m = load_mod(mod_text)
    assert isinstance(m, LinearModel)
    assert m.residual_norm < 1e-12
    # k_ss verified
    expected_k = (0.30 / (1.0 / 0.99 - (1.0 - 0.025))) ** (1.0 / 0.70)
    assert m.steady_state["k"] == pytest.approx(expected_k, rel=1e-5)


def test_multi_period_lags_and_leads():
    """Verify automatic auxiliary variable generation for |offset| >= 2."""
    mod_text = """
    var c k a;
    varexo eps;
    parameters alpha beta delta rho;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.90;

    model;
      c^(-1) = beta * c(+1)^(-1) * (alpha * exp(a(+1)) * k^(alpha-1) + 1 - delta);
      // Equation with k(-2) lag
      k = exp(a) * k(-1)^alpha - c + (1-delta)*k(-2);
      a = rho * a(-1) + eps;
    end;

    initval;
      k = 25.0;
      c = 2.0;
      a = 0.0;
    end;
    """
    parsed = parse_mod(mod_text)
    # Auxiliary variable AUX_LAG_k_1 must be automatically introduced
    assert "AUX_LAG_k_1" in parsed["variables"]
    assert len(parsed["variables"]) == 4

    m = load_mod(mod_text)
    assert isinstance(m, LinearModel)
    assert "AUX_LAG_k_1" in m.variables
    # Model solves without error
    irf = m.irf("eps", horizon=5)
    assert len(irf) == 6
    assert "k" in irf.columns
    assert "AUX_LAG_k_1" in irf.columns


def test_stoch_simul_result_api():
    """Verify StochSimulResult dictionary access, to_frame(), and formatted outputs."""
    mod_text = """
    var c k z;
    varexo eps;
    parameters alpha beta delta rho;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.90;

    model;
      c^(-1) = beta * c(+1)^(-1) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
      c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
      log(z) = rho * log(z(-1)) + eps;
    end;

    initval;
      k = 25.0;
      c = 2.0;
      z = 1.0;
    end;
    """
    m = load_mod(mod_text)
    res = m.stoch_simul(irf=10, periods=50, seed=123)

    # Subscripting by attribute
    assert res["dr"] is res.dr
    assert res["theoretical_moments"] is res.theoretical_moments
    assert res["simulated_moments"] is res.simulated_moments

    # Subscripting by IRF name
    assert isinstance(res["c_eps"], pd.Series)
    assert len(res["c_eps"]) == 11

    with pytest.raises(KeyError, match="has no attribute or IRF series 'nonexistent'"):
        _ = res["nonexistent"]

    # to_frame()
    wide_df = res.to_frame()
    assert "c_eps" in wide_df.columns
    assert len(wide_df) == 11

    shock_df = res.to_frame(shock="eps")
    assert list(shock_df.columns) == list(m.variables)
    assert len(shock_df) == 11

    # Format exports
    latex_out = res.to_latex()
    assert "\\begin{tabular}" in latex_out
    typst_out = res.to_typst()
    assert "#table" in typst_out
    md_out = res.to_markdown()
    assert "|" in md_out
