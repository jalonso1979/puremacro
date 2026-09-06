"""Unit tests for Dynare-parity Forecast Error Variance Decomposition (FEVD)
and Historical Shock Decomposition.

Tests:
1. FEVD on canonical Hansen (1985) indivisible labor RBC model.
2. FEVD on Smets & Wouters (2007) benchmark 44-variable 7-shock model.
3. Adversarial property: row sums strictly equal 1.0 across all variables and horizons.
4. Historical Shock Decomposition on simulated data: assert data reconstruction
   identity error is < 1e-11 everywhere.
5. Invariant checking: steady_state + initial_condition + sum(shocks) == actual data.
6. Export methods: .to_latex(), .to_typst(), .to_markdown(), and .plot().
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from puremacro.dsge import (
    FEVDResult,
    ShockDecompResult,
    build,
    compute_fevd,
    compute_shock_decomposition,
    load_mod,
)
from puremacro.dsge.smets_wouters import solve_sw07, SW07_SHOCK_STDS


@pytest.fixture
def hansen_rbc_model():
    """Hansen (1985) RBC model with indivisible labor."""
    mod_text = """
    var c k h y z;
    varexo eps;
    parameters alpha beta delta gamma rho A;
    alpha = 0.36; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.95; A = 1.72;
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
    return load_mod(mod_text)


@pytest.fixture
def multi_shock_rbc_model():
    """RBC model with technology (ez) and preference (eb) shocks."""
    def eqs_2shock(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta) - e.eb,
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.ez,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    return build(
        eqs_2shock,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["ez", "eb"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )


# ===========================================================================
# 1. FEVD Tests
# ===========================================================================

def test_fevd_hansen_rbc(hansen_rbc_model):
    """Test FEVD on Hansen RBC: single shock must explain 100% (1.0) of variance everywhere."""
    horizons = [1, 4, 8, 16, 32, None]
    fevd_res = compute_fevd(hansen_rbc_model, horizons=horizons)

    assert isinstance(fevd_res, FEVDResult)
    assert fevd_res.variable_names == ["c", "k", "h", "y", "z"]
    assert fevd_res.shock_names == ["eps"]
    assert fevd_res.horizons == horizons

    table = fevd_res.table
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["eps"]
    assert table.index.names == ["Variable", "Horizon"]

    # ADVERSARIAL PROPERTY: row sums strictly equal 1.0 within machine precision
    row_sums = table.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)
    assert np.all(table["eps"].to_numpy() == 1.0)

    # Formatted exports
    frame = fevd_res.to_frame()
    assert frame.shape == table.shape
    summ = fevd_res.summary()
    assert "FORECAST ERROR VARIANCE DECOMPOSITION" in summ
    assert "Variable" in fevd_res.to_markdown()
    assert "Horizon" in fevd_res.to_markdown()
    assert "\\begin{tabular}" in fevd_res.to_latex()
    assert "#table" in fevd_res.to_typst()

    # Plot
    fig = fevd_res.plot(variables=["y", "c", "k"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_fevd_smets_wouters():
    """Test FEVD on Smets-Wouters (2007) canonical 44-variable 7-shock system."""
    sol = solve_sw07()
    horizons = [1, 4, 8, 16, 32, 40, None]
    with pytest.warns(UserWarning, match="forecast-error variance is zero"):
        fevd_res = compute_fevd(sol, horizons=horizons)

    assert isinstance(fevd_res, FEVDResult)
    assert len(fevd_res.variable_names) == 44
    assert len(fevd_res.shock_names) == 7
    assert list(fevd_res.shock_names) == ["ea", "eb", "eg", "eqs", "em", "epinf", "ew"]

    table = fevd_res.table
    assert list(table.columns) == list(fevd_res.shock_names)

    # The lagged-copy auxiliaries (``c_lag = c(-1)`` ...) are known one period
    # ahead, so their 1-step forecast error is exactly zero and their shares
    # are undefined: NaN, never a padded 1/7. Every other row sums to 1.
    lag_rows = [(v, 1) for v in fevd_res.variable_names if v.endswith("_lag")]
    assert len(lag_rows) > 0
    for key in lag_rows:
        assert table.loc[key].isna().all(), key
    # The hand-coded SW07 state space also carries the two ARMA markup
    # processes (spinf, sw) whose innovation enters through a lagged
    # auxiliary, so their 1-step forecast error is zero as well. Undefined
    # rows are allowed only at horizon 1 and only when the whole row is NaN.
    nan_rows = table[table.isna().any(axis=1)]
    assert nan_rows.isna().all(axis=1).all()
    assert set(nan_rows.index.get_level_values(1)) == {1}
    assert set(lag_rows) <= set(nan_rows.index)
    defined = table.dropna(how="any")
    assert len(defined) == len(table) - len(nan_rows)
    table = defined

    # ADVERSARIAL PROPERTY: row sums strictly equal 1.0 for all 44 variables and all defined horizons
    row_sums = table.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)

    # Every variance share must be in [0, 1]
    arr = table.to_numpy()
    assert np.all(arr >= -1e-12)
    assert np.all(arr <= 1.0 + 1e-12)

    # Macroeconomic property: technology shock 'ea' and risk premium 'eb' drive output 'y'
    y_h40 = table.loc[("y", 40)]
    assert y_h40["ea"] > 0.05
    assert y_h40["eb"] > 0.50

    # Formatted exports
    assert "\\begin{tabular}" in fevd_res.to_latex()
    assert "Variable" in fevd_res.to_markdown()
    assert "Horizon" in fevd_res.to_markdown()

    # Plotting subset of variables
    fig = fevd_res.plot(variables=["y", "c", "inve", "pinf", "w", "r"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_fevd_multi_shock_rbc(multi_shock_rbc_model):
    """Test FEVD on 2-shock RBC model with custom shock standard deviations."""
    sigma = {"ez": 0.01, "eb": 0.005}
    fevd_res = compute_fevd(multi_shock_rbc_model, horizons=[1, 4, 12, None], sigma=sigma)

    assert isinstance(fevd_res, FEVDResult)
    assert fevd_res.shock_names == ["ez", "eb"]
    assert fevd_res.variable_names == ["c", "k", "z"]

    # ADVERSARIAL PROPERTY: row sums strictly equal 1.0
    row_sums = fevd_res.table.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)

    # Both shocks contribute positive variance to consumption
    c_fevd = fevd_res.table.loc["c"]
    assert np.all(c_fevd["ez"].to_numpy() > 0.0)
    assert np.all(c_fevd["eb"].to_numpy() > 0.0)

    # Technology shock completely explains technology variable z
    z_fevd = fevd_res.table.loc["z"]
    np.testing.assert_allclose(z_fevd["ez"].to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(z_fevd["eb"].to_numpy(), 0.0, atol=1e-12)


# ===========================================================================
# 2. Historical Shock Decomposition Tests
# ===========================================================================

def test_shock_decomposition_hansen_simulated(hansen_rbc_model):
    """Test Historical Shock Decomposition on simulated Hansen RBC data.

    Asserts:
    1. Data reconstruction identity error is < 1e-11 everywhere.
    2. ShockDecompResult invariant: steady_state + initial_condition + sum(shocks) == actual.
    3. Smoothed shocks match true simulation innovations.
    """
    dr = hansen_rbc_model.decision_rules()
    states = list(dr.state_variables)
    variables = list(dr.variable_names)
    shocks = list(dr.shock_names)

    A = dr.ghx.loc[states, states].to_numpy()
    B = dr.ghu.loc[states, shocks].to_numpy()
    C = dr.ghx.loc[variables, states].to_numpy()
    D = dr.ghu.loc[variables, shocks].to_numpy()
    ys = dr.ys.loc[variables].to_numpy()

    T = 60
    rng = np.random.default_rng(2026)
    u_true = rng.standard_normal((T, len(shocks))) * 0.007
    s_0 = np.array([0.03, -0.01])

    # Simulate true ground-truth path
    s_path = np.zeros((T + 1, len(states)))
    y_path = np.zeros((T, len(variables)))
    s_path[0] = s_0
    for t in range(T):
        y_path[t] = ys + C @ s_path[t] + D @ u_true[t]
        s_path[t + 1] = A @ s_path[t] + B @ u_true[t]

    df_sim = pd.DataFrame(y_path, columns=variables)

    # Compute decomposition with supplied initial state
    decomp = compute_shock_decomposition(hansen_rbc_model, data=df_sim, initial_state=s_0)
    assert isinstance(decomp, ShockDecompResult)
    assert decomp.variable_names == variables
    assert decomp.shock_names == shocks
    assert decomp.smoothed_shocks.shape == (T, 1)

    # 1. Recovered smoothed shocks match true shocks to machine precision (< 1e-11)
    u_diff = np.max(np.abs(decomp.smoothed_shocks.to_numpy() - u_true))
    assert u_diff < 1e-11, f"Smoothed shocks differ from true innovations: {u_diff:.3e}"

    # 2. DATA RECONSTRUCTION IDENTITY: error is < 1e-11 everywhere
    for var in variables:
        df_comp = decomp.to_frame(var)
        assert list(df_comp.columns) == ["eps", "initial_condition", "steady_state", "actual"]
        reconstructed = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp["eps"]
        )
        recon_err = np.max(np.abs(reconstructed - df_sim[var]))
        assert recon_err < 1e-11, f"Reconstruction error for {var} exceeds 1e-11: {recon_err:.3e}"
        # Invariant within component frame
        inv_err = np.max(np.abs(reconstructed - df_comp["actual"]))
        assert inv_err < 1e-10

    # Exports and formatting
    summ = decomp.summary("y")
    assert "HISTORICAL SHOCK DECOMPOSITION: y" in summ
    assert "\\begin{tabular}" in decomp.to_latex("y")
    assert "#table" in decomp.to_typst("y")
    assert "eps" in decomp.to_markdown("y")
    assert "initial_condition" in decomp.to_markdown("y")

    # Plotting
    fig = decomp.plot("y", style="publication")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_shock_decomposition_unknown_initial_state(hansen_rbc_model):
    """Test Historical Shock Decomposition when initial_state is None (inferred via Kalman smoother)."""
    dr = hansen_rbc_model.decision_rules()
    states = list(dr.state_variables)
    variables = list(dr.variable_names)
    shocks = list(dr.shock_names)

    A = dr.ghx.loc[states, states].to_numpy()
    B = dr.ghu.loc[states, shocks].to_numpy()
    C = dr.ghx.loc[variables, states].to_numpy()
    D = dr.ghu.loc[variables, shocks].to_numpy()
    ys = dr.ys.loc[variables].to_numpy()

    T = 50
    rng = np.random.default_rng(888)
    u_true = rng.standard_normal((T, len(shocks))) * 0.007
    s_0 = np.array([0.015, -0.005])

    s_path = np.zeros((T + 1, len(states)))
    y_path = np.zeros((T, len(variables)))
    s_path[0] = s_0
    for t in range(T):
        y_path[t] = ys + C @ s_path[t] + D @ u_true[t]
        s_path[t + 1] = A @ s_path[t] + B @ u_true[t]

    df_sim = pd.DataFrame(y_path, columns=variables)

    # Initial state is None: inferred by Kalman smoother
    decomp = compute_shock_decomposition(hansen_rbc_model, data=df_sim, initial_state=None)
    assert isinstance(decomp, ShockDecompResult)

    # DATA RECONSTRUCTION IDENTITY: error is < 1e-11 everywhere
    for var in variables:
        df_comp = decomp.to_frame(var)
        recon = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp[shocks].sum(axis=1)
        )
        recon_err = np.max(np.abs(recon - df_sim[var]))
        assert recon_err < 1e-11, f"Reconstruction error for {var} exceeds 1e-11: {recon_err:.3e}"


def test_shock_decomposition_multi_shock(multi_shock_rbc_model):
    """Test Historical Shock Decomposition on multi-shock system (ez and eb)."""
    dr = multi_shock_rbc_model.decision_rules()
    states = list(dr.state_variables)
    variables = list(dr.variable_names)
    shocks = list(dr.shock_names)

    A = dr.ghx.loc[states, states].to_numpy()
    B = dr.ghu.loc[states, shocks].to_numpy()
    C = dr.ghx.loc[variables, states].to_numpy()
    D = dr.ghu.loc[variables, shocks].to_numpy()
    ys = dr.ys.loc[variables].to_numpy()

    T = 45
    rng = np.random.default_rng(777)
    sigmas = np.array([0.01, 0.005])
    u_true = rng.standard_normal((T, len(shocks))) * sigmas
    s_0 = np.array([0.02, 0.01])

    s_path = np.zeros((T + 1, len(states)))
    y_path = np.zeros((T, len(variables)))
    s_path[0] = s_0
    for t in range(T):
        y_path[t] = ys + C @ s_path[t] + D @ u_true[t]
        s_path[t + 1] = A @ s_path[t] + B @ u_true[t]

    df_sim = pd.DataFrame(y_path, columns=variables)

    decomp = compute_shock_decomposition(
        multi_shock_rbc_model,
        data=df_sim,
        initial_state=s_0,
        sigma=sigmas,
    )
    assert isinstance(decomp, ShockDecompResult)
    assert decomp.shock_names == ["ez", "eb"]

    # Assert reconstruction identity < 1e-11 everywhere
    for var in variables:
        df_comp = decomp.to_frame(var)
        recon = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp["ez"]
            + df_comp["eb"]
        )
        recon_err = np.max(np.abs(recon - df_sim[var]))
        assert recon_err < 1e-11, f"Reconstruction error for {var} exceeds 1e-11: {recon_err:.3e}"

    # Verify plotting
    fig = decomp.plot("c", style="publication")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_linear_model_methods(hansen_rbc_model):
    """Test convenience methods attached directly to LinearModel."""
    fevd_res = hansen_rbc_model.fevd_result(horizons=[1, 4, 8])
    assert isinstance(fevd_res, FEVDResult)
    np.testing.assert_allclose(fevd_res.table.sum(axis=1), 1.0, atol=1e-12)

    # Test shock_decomposition method
    sim_data = hansen_rbc_model.simulate(periods=30, seed=42)
    # simulate returns deviations from steady state; convert to levels matching model
    for v in hansen_rbc_model.variables:
        sim_data[v] += hansen_rbc_model.steady_state[v]

    s_0 = np.zeros(hansen_rbc_model.n_states)
    decomp = hansen_rbc_model.shock_decomposition(sim_data, initial_state=s_0)
    assert isinstance(decomp, ShockDecompResult)

    for v in hansen_rbc_model.variables:
        df_comp = decomp.to_frame(v)
        recon = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp[list(hansen_rbc_model.shocks)].sum(axis=1)
        )
        assert np.max(np.abs(recon - df_comp["actual"])) < 1e-10
