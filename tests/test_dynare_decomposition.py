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
import matplotlib.dates as mdates
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
from puremacro.dsge._moments import ZeroVarianceWarning
from puremacro.dsge.smets_wouters import solve_sw07, SW07_SHOCK_STDS


def _reported_companion(model):
    """(A, B, C, D) of ``model`` in the timing it reports its variables in."""
    A, B = model.solution.G, model.solution.N
    M_x, M_u = model._reported_loadings()
    order = list(model.states) + list(model.controls)
    idx = [order.index(v) for v in model.variables]
    return A, B, M_x[idx], M_u[idx]


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
    """FEVD of a Klein-timed ``build()`` model is dated like irf() / simulate().

    ``build()`` reports the state at ``t`` (``v_t = [x_t; y_t]``), so a state is
    known one period ahead: its 1-step forecast error is identically zero and
    its shares are undefined. Feeding the Dynare-timed ``ghx``/``ghu`` loadings
    instead would date every state row one period later than the control rows.
    """
    sigma = {"ez": 0.01, "eb": 0.005}
    with pytest.warns(ZeroVarianceWarning, match=r"k@h=1"):
        fevd_res = compute_fevd(
            multi_shock_rbc_model, horizons=[1, 3, 4, 12, None], sigma=sigma
        )

    assert isinstance(fevd_res, FEVDResult)
    assert fevd_res.shock_names == ["ez", "eb"]
    assert fevd_res.variable_names == ["c", "k", "z"]
    table = fevd_res.table

    # The predetermined states have no 1-step forecast error at all.
    assert table.loc[("k", 1)].isna().all()
    assert table.loc[("z", 1)].isna().all()
    nan_rows = table[table.isna().any(axis=1)]
    assert set(nan_rows.index) == {("k", 1), ("z", 1)}

    # ADVERSARIAL PROPERTY: every defined row sums to exactly 1.0
    defined = table.dropna(how="any")
    np.testing.assert_allclose(defined.sum(axis=1).to_numpy(), 1.0, atol=1e-12)

    # Both shocks contribute positive variance to consumption
    c_fevd = table.loc["c"]
    assert np.all(c_fevd["ez"].to_numpy() > 0.0)
    assert np.all(c_fevd["eb"].to_numpy() > 0.0)

    # Technology shock completely explains technology variable z
    z_fevd = table.loc["z"].dropna()
    np.testing.assert_allclose(z_fevd["ez"].to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(z_fevd["eb"].to_numpy(), 0.0, atol=1e-12)

    # Independent re-derivation of the h-step forecast error in the *reported*
    # timing: fe = M_u u_{t+h} + M_x sum_{k=1}^{h-1} G^{h-1-k} N u_{t+k}.
    A, B, M_x, M_u = _reported_companion(multi_shock_rbc_model)
    sd = np.array([sigma[s] for s in multi_shock_rbc_model.shocks])
    for h in (3, 4, 12):
        psi = [M_u] + [M_x @ np.linalg.matrix_power(A, j - 1) @ B for j in range(1, h)]
        v = sum((p ** 2) * sd ** 2 for p in psi)
        expected = v / v.sum(axis=1, keepdims=True)
        got = table.xs(h, level="Horizon").loc[list(multi_shock_rbc_model.variables)]
        np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-12)

    # The old Dynare-timed loadings put k's h=4 answer on the h=3 row.
    assert table.loc[("k", 3), "ez"] == pytest.approx(0.203684, abs=1e-5)
    assert table.loc[("k", 4), "ez"] == pytest.approx(0.491553, abs=1e-5)


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
        assert list(df_comp.columns) == [
            "eps", "initial_condition", "steady_state", "residual", "actual"
        ]
        # 'actual' is the caller's own series, bit for bit -- never the
        # model's reconstruction dressed up as data.
        np.testing.assert_array_equal(
            df_comp["actual"].to_numpy(), df_sim[var].to_numpy()
        )
        reconstructed = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp["eps"]
        )
        recon_err = np.max(np.abs(reconstructed - df_sim[var]))
        assert recon_err < 1e-11, f"Reconstruction error for {var} exceeds 1e-11: {recon_err:.3e}"
        # This model fits the data exactly, so nothing is left unexplained.
        assert np.max(np.abs(df_comp["residual"].to_numpy())) < 1e-11
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
    """Test Historical Shock Decomposition on multi-shock system (ez and eb).

    The data is generated in the timing this Klein-timed model *reports* its
    variables in (the timing ``simulate()`` produces), not in Dynare's
    end-of-period state timing.
    """
    dr = multi_shock_rbc_model.decision_rules()
    states = list(dr.state_variables)
    variables = list(dr.variable_names)
    shocks = list(dr.shock_names)

    A, B, C, D = _reported_companion(multi_shock_rbc_model)
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

    # The recovered shocks are the true innovations, not a min-norm split.
    # Under Klein timing the only contemporaneous loading on the shocks is the
    # control block L (1 x 2 here), so the *final* period's split between ez
    # and eb is not identified by data ending at T -- every earlier one is,
    # through the state observations that follow it.
    np.testing.assert_allclose(
        decomp.smoothed_shocks.to_numpy()[:-1], u_true[:-1], atol=1e-10
    )

    # Assert reconstruction identity < 1e-11 everywhere, against the DATA
    for var in variables:
        df_comp = decomp.to_frame(var)
        np.testing.assert_array_equal(
            df_comp["actual"].to_numpy(), df_sim[var].to_numpy()
        )
        recon = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp["ez"]
            + df_comp["eb"]
        )
        recon_err = np.max(np.abs(recon - df_sim[var]))
        assert recon_err < 1e-11, f"Reconstruction error for {var} exceeds 1e-11: {recon_err:.3e}"
        assert np.max(np.abs(df_comp["residual"].to_numpy())) < 1e-11

    # Verify plotting
    fig = decomp.plot("c", style="publication")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_linear_model_methods(hansen_rbc_model):
    """Test convenience methods attached directly to LinearModel."""
    fevd_res = hansen_rbc_model.fevd_result(horizons=[1, 4, 8])
    assert isinstance(fevd_res, FEVDResult)
    np.testing.assert_allclose(fevd_res.table.sum(axis=1), 1.0, atol=1e-12)

    # Test shock_decomposition method. ``simulate`` burns in 100 periods, so
    # the state at the start of the returned sample is NOT zero: the initial
    # condition has to be inferred by the smoother.
    sim_data = hansen_rbc_model.simulate(periods=30, seed=42)
    # simulate returns deviations from steady state; convert to levels matching model
    for v in hansen_rbc_model.variables:
        sim_data[v] += hansen_rbc_model.steady_state[v]

    decomp = hansen_rbc_model.shock_decomposition(sim_data)
    assert isinstance(decomp, ShockDecompResult)

    for v in hansen_rbc_model.variables:
        df_comp = decomp.to_frame(v)
        np.testing.assert_array_equal(
            df_comp["actual"].to_numpy(), sim_data[v].to_numpy()
        )
        recon = (
            df_comp["steady_state"]
            + df_comp["initial_condition"]
            + df_comp[list(hansen_rbc_model.shocks)].sum(axis=1)
        )
        assert np.max(np.abs(recon - df_comp["actual"])) < 1e-9

    # Asserting a *wrong* initial condition is not silently absorbed: the
    # gap shows up in 'residual' and is announced, instead of being written
    # into the column labelled 'actual'.
    with pytest.warns(UserWarning, match="does not reproduce the observed data"):
        bad = hansen_rbc_model.shock_decomposition(
            sim_data, initial_state=np.zeros(hansen_rbc_model.n_states)
        )
    df_bad = bad.to_frame("k")
    np.testing.assert_array_equal(
        df_bad["actual"].to_numpy(), sim_data["k"].to_numpy()
    )
    assert np.max(np.abs(df_bad["residual"].to_numpy())) > 1e-3


# ===========================================================================
# 3. The smoother is the estimator, and 'actual' is the caller's own data
# ===========================================================================

@pytest.fixture
def partially_observed_model():
    """Two AR(1) states of very different volatility, one observable."""
    return load_mod(
        """
        var z1 z2 x; varexo e1 e2;
        model;
          z1 = 0.9*z1(-1) + e1;
          z2 = 0.5*z2(-1) + e2;
          x = z1 + z2;
        end;
        shocks; var e1; stderr 0.01; var e2; stderr 0.0001; end;
        """
    )


def _simulate(model, u, x0=None):
    """Model-consistent data for ``u``, in the model's reported timing."""
    dr = model.decision_rules()
    V, S, K = list(dr.variable_names), list(dr.state_variables), list(dr.shock_names)
    A = dr.ghx.loc[S, S].to_numpy()
    B = dr.ghu.loc[S, K].to_numpy()
    C = dr.ghx.loc[V, S].to_numpy()
    D = dr.ghu.loc[V, K].to_numpy()
    ys = dr.ys.loc[V].to_numpy()
    x = np.zeros(len(S)) if x0 is None else np.asarray(x0, dtype=float)
    rows = []
    for t in range(len(u)):
        rows.append(ys + C @ x + D @ u[t])
        x = A @ x + B @ u[t]
    return pd.DataFrame(rows, columns=V)


def test_shock_decomposition_uses_the_smoother_when_shocks_outnumber_observables(
    partially_observed_model,
):
    """Fewer observables than shocks: the Kalman smoother, not a min-norm solve.

    ``lstsq`` on an underdetermined ``D_obs u = dev`` returns the minimum-norm
    solution, whose residual is zero *by construction*. The old accept-if-exact
    test therefore fired at every period, discarding the smoother and splitting
    each observed movement evenly across shock directions -- ignoring that e2
    is a hundred times smaller than e1. The two recovered series came out
    bit-identical.
    """
    m = partially_observed_model
    T = 80
    rng = np.random.default_rng(11)
    sd = np.array([0.01, 0.0001])
    u_true = rng.standard_normal((T, 2)) * sd
    data = _simulate(m, u_true)[["x"]]

    decomp = compute_shock_decomposition(m, data=data)
    u_hat = decomp.smoothed_shocks.to_numpy()

    # The two shocks are not the same series.
    assert np.max(np.abs(u_hat[:, 0] - u_hat[:, 1])) > 1e-4
    # The large, identified shock is recovered; the tiny one is shrunk toward
    # zero as its prior demands, not inflated to the size of the large one.
    assert np.corrcoef(u_hat[:, 0], u_true[:, 0])[0, 1] > 0.99
    assert u_hat[:, 0].std() == pytest.approx(u_true[:, 0].std(), rel=0.05)
    assert u_hat[:, 1].std() < 10.0 * sd[1]

    # The observable is still reproduced exactly.
    df = decomp.to_frame("x")
    np.testing.assert_array_equal(df["actual"].to_numpy(), data["x"].to_numpy())
    recon = df["steady_state"] + df["initial_condition"] + df[["e1", "e2"]].sum(axis=1)
    assert np.max(np.abs(recon - data["x"])) < 1e-10


def test_actual_column_is_the_users_data_and_the_gap_is_reported(
    partially_observed_model,
):
    """A model that cannot fit the data says so; it does not rewrite 'actual'.

    The column used to be overwritten with the model's own reconstruction
    whenever the two disagreed by more than 1e-10, which made the adding-up
    invariant an algebraic identity that could never fail.
    """
    m = partially_observed_model
    T = 40
    rng = np.random.default_rng(5)
    u_true = rng.standard_normal((T, 2)) * np.array([0.01, 0.0001])
    data = _simulate(m, u_true)
    # Break the cross-equation restriction x = z1 + z2 that the model imposes.
    data = data.copy()
    data["x"] = data["x"] + 0.02 * rng.standard_normal(T)

    with pytest.warns(UserWarning, match="does not reproduce the observed data"):
        decomp = compute_shock_decomposition(m, data=data)

    for var in ["z1", "z2", "x"]:
        df = decomp.to_frame(var)
        np.testing.assert_array_equal(df["actual"].to_numpy(), data[var].to_numpy())
        recon = (
            df["steady_state"] + df["initial_condition"] + df[["e1", "e2"]].sum(axis=1)
        )
        # The adding-up identity holds only once the residual is counted.
        np.testing.assert_allclose(
            (recon + df["residual"]).to_numpy(), df["actual"].to_numpy(), atol=1e-12
        )
    assert np.max(np.abs(decomp.to_frame("x")["residual"].to_numpy())) > 1e-3
    assert "unexplained residual" in decomp.summary("x")

    # The residual is drawn, so the stacked bars still add up to the black line.
    fig = decomp.plot("x")
    assert isinstance(fig, Figure)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "residual" in labels
    plt.close(fig)


def test_shock_decomposition_reports_periods_excluded_by_missing_data(
    hansen_rbc_model,
):
    """A NaN observation is interpolated, and the summary says it was not checked."""
    dr = hansen_rbc_model.decision_rules()
    variables = list(dr.variable_names)
    T = 40
    rng = np.random.default_rng(3)
    u_true = rng.standard_normal((T, 1)) * 0.007
    data = _simulate(hansen_rbc_model, u_true)
    data.loc[10, "y"] = np.nan

    decomp = compute_shock_decomposition(hansen_rbc_model, data=data)
    df = decomp.to_frame("y")
    assert np.isnan(df["actual"].to_numpy()[10])
    assert "periods excluded from the adding-up check (missing 'actual'): 1" in (
        decomp.summary("y")
    )
    # Every other variable is fully observed and fully checked.
    assert "periods excluded from the adding-up check (missing 'actual'): 0" in (
        decomp.summary("c")
    )


def test_shock_decomposition_plot_bars_span_a_quarter_on_a_date_axis(
    hansen_rbc_model,
):
    """Bar width is in data units; on a date axis one unit is a DAY, not a period."""
    T = 24
    rng = np.random.default_rng(7)
    u_true = rng.standard_normal((T, 1)) * 0.007
    data = _simulate(hansen_rbc_model, u_true)
    data.index = pd.date_range("1990-01-01", periods=T, freq="QE")

    decomp = compute_shock_decomposition(hansen_rbc_model, data=data)
    fig = decomp.plot("y")
    widths = [p.get_width() for p in fig.axes[0].patches]
    assert widths, "no bars drawn"
    expected = 0.8 * np.median(np.diff(mdates.date2num(data.index.to_pydatetime())))
    assert expected > 60.0  # ~0.8 * 91 days
    np.testing.assert_allclose(max(widths), expected, rtol=1e-9)
    plt.close(fig)

    # A RangeIndex keeps the matplotlib default of 0.8 (one period wide).
    data2 = data.reset_index(drop=True)
    fig2 = compute_shock_decomposition(hansen_rbc_model, data=data2).plot("y")
    assert max(p.get_width() for p in fig2.axes[0].patches) == pytest.approx(0.8)
    plt.close(fig2)
