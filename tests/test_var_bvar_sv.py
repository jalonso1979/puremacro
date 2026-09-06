"""Unit tests for Bayesian VAR with Stochastic Volatility (BVAR-SV).

Verifies:
1. Synthetic SV recovery with known stochastic volatility jumps:
   - Gelman-Rubin split-Rhat convergence diagnostics (assert Rhat < 1.1).
   - Recovery of true volatility paths within 95% credible intervals.
2. Volatility-conditioned IRFs:
   - Returns BVAR_SV_IRF (ndarray subclass with .lower, .upper, .median).
   - Volatility conditioning: crisis/high-volatility date has wider bands than calm date.
3. Predictive density log-scores:
   - In-sample lppd (finite scalar total and pointwise series of length T_eff).
   - Out-of-sample log_score(holdout) on a hold-out sample.
4. Presentation suite:
   - .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot().
5. Multi-chain and single-chain execution; n_draws is per chain.
6. Input validation and error handling.
7. Regression tests for the audit findings (phi MH sign, n_draws semantics,
   stability-rejection accounting, result aliases, to_frame filters,
   intercept prior, forecasts / fan charts).
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from puremacro.var import bvar_sv, BVAR_SVResult
from puremacro.var.bvar_sv import BVAR_SV_IRF, BVAR_SVForecast, _sample_phi_mh


@pytest.fixture
def synthetic_sv_var_data():
    """Generate synthetic VAR(1) data with known stochastic volatility jumps."""
    rng = np.random.default_rng(42)
    T = 150
    n = 2
    lags = 1

    # True parameters
    c_true = np.array([0.05, -0.02])
    A1_true = np.array([[0.45, 0.10],
                        [0.05, 0.40]])
    A_mat_true = np.array([[1.0, 0.0],
                           [0.3, 1.0]])
    A_inv_true = np.linalg.inv(A_mat_true)

    # Known stochastic volatility jumps: regime 1 (calm) vs regime 2 (volatile)
    true_h = np.empty((T, n))
    half = T // 2
    true_h[:half, 0] = -1.5
    true_h[half:, 0] = 0.5
    true_h[:half, 1] = -1.0
    true_h[half:, 1] = 0.8

    # Simulate VAR-SV
    Y = np.zeros((T, n))
    for t in range(1, T):
        eps_t = rng.standard_normal(n)
        nu_t = np.exp(true_h[t] / 2.0) * eps_t
        u_t = A_inv_true @ nu_t
        Y[t] = c_true + A1_true @ Y[t - 1] + u_t

    df = pd.DataFrame(Y, columns=["y1", "y2"])
    return {
        "df": df,
        "Y": Y,
        "true_h": true_h,
        "c_true": c_true,
        "A1_true": A1_true,
        "lags": lags,
        "n": n,
        "T": T,
    }


def _simulate_univariate_sv(T: int, phi: float, sigma_h: float, mu: float, h0_dev: float, seed: int) -> np.ndarray:
    """AR(1)-SV path y_t = 0.5 y_{t-1} + exp(h_t/2) eps_t with a far-from-mean h_1."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, 1))
    h = np.zeros(T)
    h[0] = mu + h0_dev
    for t in range(1, T):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_h * rng.standard_normal()
        Y[t, 0] = 0.5 * Y[t - 1, 0] + np.exp(h[t] / 2.0) * rng.standard_normal()
    return Y


def test_bvar_sv_synthetic_recovery(synthetic_sv_var_data):
    """Test synthetic SV recovery: MCMC draws pass Rhat < 1.1 and recover true h within 95% CI."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=1000,
        n_burn=1000,
        minnesota_prior=True,
        n_chains=2,
        seed=42,
    )

    assert isinstance(res, BVAR_SVResult)
    assert res.n == 2
    assert res.lags == 1
    assert res.T_eff == data["T"] - 1
    assert res.n_draws == 1000
    assert res.n_chains == 2
    assert res.n_total_draws == 2000

    # 1. Gelman-Rubin convergence diagnostics: assert Rhat < 1.1 on all parameters
    r_hat = res.gelman_rubin()
    assert isinstance(r_hat, dict)
    assert "max" in r_hat
    assert r_hat["max"] < 1.1, f"Expected R_hat < 1.1, got {r_hat['max']:.4f}"
    if "beta_max" in r_hat:
        assert r_hat["beta_max"] < 1.1
    if "h_max" in r_hat:
        assert r_hat["h_max"] < 1.1

    # 2. True volatility recovery within 95% credible intervals
    h_draws = res.h_draws  # (D, T_eff, n)
    h_lower = np.percentile(h_draws, 2.5, axis=0)
    h_upper = np.percentile(h_draws, 97.5, axis=0)
    true_h_eff = data["true_h"][data["lags"]:]  # (T_eff, n)

    # Assert coverage rate exceeds 90% (nominal 95%)
    cov_rate = np.mean((true_h_eff >= h_lower) & (true_h_eff <= h_upper))
    assert cov_rate >= 0.90, f"Expected coverage >= 90%, got {cov_rate * 100:.1f}%"

    # 3. Estimated VAR coefficients are close to true coefficients
    beta_mean = np.mean(res.beta_draws, axis=0)  # (k, n)
    A1_est = beta_mean[1:, :].T  # (n, n)
    np.testing.assert_allclose(A1_est, data["A1_true"], atol=0.20)


def test_bvar_sv_volatility_conditioned_irf(synthetic_sv_var_data):
    """Test volatility-conditioned IRFs and verify wider bands during high-volatility regime."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=400,
        n_burn=400,
        seed=42,
    )

    # Condition on low-volatility period (date 20) vs high-volatility period (date 120)
    irf_low = res.irf(horizon=10, t_idx=20, ci=0.9)
    irf_high = res.irf(horizon=10, t_idx=120, ci=0.9)

    # Type contract: must be ndarray subclass BVAR_SV_IRF
    assert isinstance(irf_low, np.ndarray)
    assert isinstance(irf_low, BVAR_SV_IRF)
    assert irf_low.shape == (11, 2, 2)
    assert irf_low.median.shape == (11, 2, 2)
    assert irf_low.lower.shape == (11, 2, 2)
    assert irf_low.upper.shape == (11, 2, 2)
    assert irf_low.draws.shape == (res.n_total_draws, 11, 2, 2)

    # Lower band <= median <= upper band
    assert np.all(irf_low.lower <= irf_low.median + 1e-10)
    assert np.all(irf_low.median <= irf_low.upper + 1e-10)

    # Volatility conditioning: high-volatility regime has strictly wider credible bands
    width_low = np.mean(irf_low.upper - irf_low.lower)
    width_high = np.mean(irf_high.upper - irf_high.lower)
    assert width_high > width_low, (
        f"High volatility width ({width_high:.4f}) should exceed low volatility width ({width_low:.4f})"
    )

    # Test to_frame on IRF
    df_irf = irf_low.to_frame(target_idx=0, shock_idx=0, names=res.names)
    assert isinstance(df_irf, pd.DataFrame)
    assert len(df_irf) == 11
    assert list(df_irf.columns) == ["horizon", "target", "shock", "median", "lower", "upper"]


def test_bvar_sv_predictive_log_score(synthetic_sv_var_data):
    """Test in-sample lppd computation and its aliases."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=250,
        n_burn=200,
        seed=10,
    )

    total_ls = res.predictive_log_score()
    assert isinstance(total_ls, float)
    assert np.isfinite(total_ls)

    pointwise_ls = res.predictive_log_score(point_by_point=True)
    assert isinstance(pointwise_ls, np.ndarray)
    assert pointwise_ls.shape == (res.T_eff,)
    assert np.all(np.isfinite(pointwise_ls))
    np.testing.assert_allclose(np.sum(pointwise_ls), total_ls, rtol=1e-10)

    # log_score() without a hold-out sample is the same in-sample quantity
    assert res.log_score() == total_ls
    np.testing.assert_allclose(res.log_score(point_by_point=True), pointwise_ls)


def test_bvar_sv_presentation_suite(synthetic_sv_var_data):
    """Test .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot()."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=200,
        n_burn=200,
        seed=7,
    )

    # 1. Summary: settings, convergence, honest fit label, per-parameter CI and R-hat
    summ = res.summary()
    assert isinstance(summ, str)
    assert "Bayesian VAR with Stochastic Volatility" in summ
    assert "Variables (n)" in summ
    assert "Lag order (p)" in summ
    assert "In-sample lppd" in summ
    assert "not out-of-sample" in summ
    assert "2 chain(s) x 200" in summ
    for name in res.names:
        for label in ("μ", "φ", "σ_h"):
            assert any(line.strip().startswith(name) and label in line and "R̂ =" in line
                       for line in summ.splitlines()), f"missing {label} line for {name}"
    assert "[90% CI]" in summ
    assert "[50% CI]" in res.summary(ci=0.5)

    # 2. DataFrame
    df_tab = res.to_frame()
    assert isinstance(df_tab, pd.DataFrame)
    assert len(df_tab) == res.n
    for col in ["variable", "mu_mean", "phi_mean", "sigma_h_mean", "R_hat"]:
        assert col in df_tab.columns

    # 3. Markdown
    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert "| variable |" in md_str

    # 4. LaTeX
    tex_str = res.to_latex()
    assert isinstance(tex_str, str)
    assert "\\begin{tabular}" in tex_str
    assert "\\end{tabular}" in tex_str

    # 5. Typst
    typ_str = res.to_typst()
    assert isinstance(typ_str, str)
    assert "#table(" in typ_str

    # 6. Plot: all variables in the volatility panels by default
    fig = res.plot(t_idx=-1, horizon=10, ci=0.9)
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("y1" in lab for lab in labels) and any("y2" in lab for lab in labels)
    plt.close("all")

    # target_idx restricts the volatility panels to one variable
    fig = res.plot(horizon=5, target_idx=1)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert all("y2" in lab for lab in labels) and len(labels) == 1
    plt.close("all")

    # Plot into existing axes
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    out_axes = res.plot(ax=axes, horizon=10)
    assert out_axes is axes
    plt.close("all")


def test_bvar_sv_single_chain_and_alias(synthetic_sv_var_data):
    """Test bvar_sv with n_chains=1, keyword alias p, and diffuse prior."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["Y"],
        p=1,
        n_draws=400,
        n_burn=200,
        n_chains=1,
        minnesota_prior=False,
        seed=99,
    )
    assert res.lags == 1
    assert res.p == 1
    assert res.n_chains == 1
    assert res.n_total_draws == 400
    assert res.beta_draws.shape == (400, 1 + 2 * 1, 2)
    assert res.A_draws.shape == (400, 1, 2, 2)
    assert res.intercept_draws.shape == (400, 2)
    assert res.a_draws.shape == (400, 2, 2)
    assert res.h_draws.shape == (400, data["T"] - 1, 2)


def test_bvar_sv_custom_names_and_dataframe(synthetic_sv_var_data):
    """Test custom column names propagation to summary and properties."""
    Y = synthetic_sv_var_data["Y"]
    df = pd.DataFrame(Y, columns=["GDP_Growth", "Inflation"])
    res = bvar_sv(df, lags=1, n_draws=100, n_burn=100, seed=12)

    assert res.names == ["GDP_Growth", "Inflation"]
    summ = res.summary()
    assert "GDP_Growth" in summ
    assert "Inflation" in summ


def test_bvar_sv_input_validation():
    """Test error handling for invalid input parameters."""
    Y = np.random.default_rng(0).standard_normal((50, 2))

    # Invalid lags
    with pytest.raises(ValueError, match="lags must be a positive integer"):
        bvar_sv(Y, lags=0)

    with pytest.raises(ValueError, match="lags must be a positive integer"):
        bvar_sv(Y, lags=-1)

    # Invalid n_draws / n_burn
    with pytest.raises(ValueError, match="n_draws must be > 0"):
        bvar_sv(Y, n_draws=0)

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient observations"):
        bvar_sv(Y[:5], lags=4)

    # 1D input
    with pytest.raises(ValueError, match="data must be a 2D array"):
        bvar_sv(np.random.randn(50))

    # Out of bounds t_idx in irf()
    res = bvar_sv(Y, lags=1, n_draws=25, n_burn=20, seed=1)
    with pytest.raises(ValueError, match="out of bounds"):
        res.irf(t_idx=1000)

    # Invalid ax in plot()
    with pytest.raises(ValueError, match="must contain 3 subplots"):
        _, single_ax = plt.subplots(1, 1)
        res.plot(ax=single_ax)
    plt.close("all")


def test_bvar_sv_precision_symmetry_and_stability():
    """Verify that Sigma_inv and prec maintain strict mathematical symmetry (||prec - prec.T||_inf < 1e-14)."""
    # 1. Direct algebraic verification with non-diagonal A and asymmetric volatility
    rng = np.random.default_rng(1234)
    T = 100
    n = 3
    k = 1 + n * 1  # constant + 1 lag = 4

    # Lower triangular contemporaneous matrix with non-zero off-diagonals
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.6, 1.0, 0.0],
        [-0.4, 0.5, 1.0],
    ])
    # Highly unequal log-volatilities across equations
    exp_neg_h = np.exp(-np.column_stack([
        np.linspace(-1.0, 2.0, T),
        np.linspace(1.5, -0.5, T),
        np.linspace(-2.0, 0.5, T),
    ]))

    # Compute Sigma_inv via corrected formula: Sigma_t^{-1} = A' diag(exp(-h_t)) A
    Sigma_inv = np.einsum("ji,tj,jk->tik", A, exp_neg_h, A)
    asym_raw = np.max(np.abs(Sigma_inv - np.swapaxes(Sigma_inv, 1, 2)))
    assert asym_raw < 1e-14, f"Raw Sigma_inv is mathematically asymmetric: {asym_raw}"

    Sigma_inv = 0.5 * (Sigma_inv + np.swapaxes(Sigma_inv, 1, 2))

    # Form outer products of regressors X
    X = rng.standard_normal((T, k))
    X_outer = X[:, :, None] * X[:, None, :]
    V0_inv = np.diag(rng.uniform(0.1, 10.0, n * k))

    # Joint precision matrix: V_post^{-1} = V0^{-1} + sum_t Sigma_t^{-1} (x_t x_t')
    prec = V0_inv + np.einsum("tij,tkl->ikjl", Sigma_inv, X_outer).reshape(n * k, n * k)
    asym_prec_raw = np.max(np.abs(prec - prec.T))
    assert asym_prec_raw < 1e-14, f"Raw prec is mathematically asymmetric: {asym_prec_raw}"

    prec = 0.5 * (prec + prec.T)
    np.testing.assert_allclose(prec, prec.T, atol=1e-15)

    # 2. End-to-end MCMC run on Challenger 1 DGP with persistent volatility shifts
    Y = np.zeros((T, 2))
    for t in range(1, T):
        vol = 0.5 if t < 50 else 2.5
        Y[t, 0] = 0.4 * Y[t - 1, 0] + vol * rng.standard_normal()
        Y[t, 1] = 0.2 * Y[t - 1, 1] + 0.3 * Y[t, 0] + vol * rng.standard_normal()
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=100, n_burn=50, n_chains=2, seed=42)
    # n_draws is per chain: two chains pool 200 draws (the old test enshrined a pooled total of 100)
    assert res.n_draws == 100
    assert res.n_total_draws == 200

    # Check precision and covariance symmetry across sampled draws
    for m in range(min(20, res.n_total_draws)):
        A_m = res.a_draws[m]
        h_m = res.h_draws[m]
        exp_neg_h_m = np.exp(-h_m)
        Sigma_inv_m = np.einsum("ji,tj,jk->tik", A_m, exp_neg_h_m, A_m)
        asym_draw = np.max(np.abs(Sigma_inv_m - np.swapaxes(Sigma_inv_m, 1, 2)))
        assert asym_draw < 1e-14, f"Draw {m} produced asymmetric Sigma_inv: {asym_draw}"


# ---------------------------------------------------------------------------
# Regression tests for the audit findings
# ---------------------------------------------------------------------------

def test_phi_mh_kernel_targets_exact_conditional_posterior():
    """The phi Metropolis-Hastings step must target the exact conditional posterior.

    The old acceptance ratio carried the h_1 stationary-density term with the
    wrong sign (-0.5 (phi'^2 - phi^2) h_1^2 / sigma^2 instead of +0.5 ...). On
    this T=12 path with a large initial deviation the old kernel converged to a
    chain mean of 0.454 while the exact conditional mean is 0.793.
    """
    rng = np.random.default_rng(1)
    T = 12
    phi_true = 0.5
    sig2 = 0.3
    h_dev = np.empty(T)
    h_dev[0] = 2.0
    for t in range(1, T):
        h_dev[t] = phi_true * h_dev[t - 1] + np.sqrt(sig2) * rng.standard_normal()
    x_phi, y_phi = h_dev[:-1], h_dev[1:]

    # Exact conditional posterior on a fine grid: prior N(0.85, 0.1) truncated,
    # stationary initial state N(0, sig2 / (1 - phi^2)), AR(1) transitions.
    grid = np.linspace(-0.998, 0.998, 20001)
    logp = (
        norm.logpdf(grid, 0.85, np.sqrt(0.1))
        + 0.5 * np.log(1.0 - grid ** 2)
        - (1.0 - grid ** 2) * h_dev[0] ** 2 / (2.0 * sig2)
        - np.sum((y_phi[None, :] - grid[:, None] * x_phi[None, :]) ** 2, axis=1) / (2.0 * sig2)
    )
    w = np.exp(logp - logsumexp(logp))
    exact_mean = float(np.sum(w * grid))
    exact_sd = float(np.sqrt(np.sum(w * (grid - exact_mean) ** 2)))

    kernel_rng = np.random.default_rng(7)
    N = 40_000
    phi = 0.5
    out = np.empty(N)
    for i in range(N):
        phi = _sample_phi_mh(h_dev, phi, sig2, kernel_rng)
        out[i] = phi
    chain = out[N // 5:]
    assert abs(chain.mean() - exact_mean) < 0.02, f"kernel mean {chain.mean():.4f} vs exact {exact_mean:.4f}"
    assert abs(chain.std() - exact_sd) < 0.02


def test_phi_persistence_recovered_end_to_end():
    """bvar_sv must recover a persistent volatility process (phi = 0.95).

    With the sign error in the phi acceptance ratio the sampler on this
    univariate AR(1)-SV path (T=400, sigma_h=0.2, h_1 two units above its
    mean) returned a phi posterior mean of 0.245 (sd 0.18); the corrected
    kernel gives 0.914 (sd 0.06).
    """
    Y = _simulate_univariate_sv(T=400, phi=0.95, sigma_h=0.2, mu=0.0, h0_dev=2.0, seed=3)
    res = bvar_sv(Y, lags=1, n_draws=800, n_burn=400, n_chains=1, seed=11)
    phi_mean = float(res.phi_draws.mean())
    assert phi_mean > 0.85, f"phi posterior mean {phi_mean:.3f} far below true 0.95"
    assert abs(phi_mean - 0.95) < 0.1
    assert float(res.sigma_h_draws.mean()) < 0.35


def test_n_draws_is_per_chain_without_truncation():
    """n_draws means draws per chain; nothing is silently dropped.

    Old behaviour: n_draws=101, n_chains=3 retained 99 pooled draws (33 per
    chain) although the docstring promised 101 per chain.
    """
    Y = np.random.default_rng(0).standard_normal((70, 2))
    res = bvar_sv(Y, lags=1, n_draws=101, n_burn=5, n_chains=3, seed=1)
    assert res.n_draws == 101
    assert res.n_chains == 3
    assert res.n_total_draws == 303
    for arr in (res.beta_draws, res.h_draws, res.a_draws, res.mu_draws, res.phi_draws, res.sigma_h_draws):
        assert arr.shape[0] == 303
    # thin keeps exactly n_draws per chain
    res_thin = bvar_sv(Y, lags=1, n_draws=10, n_burn=2, n_chains=2, thin=3, seed=1)
    assert res_thin.n_total_draws == 20


def test_tiny_n_draws_does_not_crash_split_rhat():
    """n_draws=2 with two chains used to crash in the split-R-hat reshape (size 0)."""
    Y = np.random.default_rng(0).standard_normal((70, 2))
    res = bvar_sv(Y, lags=1, n_draws=2, n_burn=2, n_chains=2, seed=1)
    assert res.n_total_draws == 4
    # With one draw per half-chain no split R-hat exists: the result says so
    # (NaN, "UNDIAGNOSED") instead of reporting a reassuring 1.0.
    assert np.isnan(res.max_rhat)
    assert "UNDIAGNOSED" in res.summary()


def test_argument_validation_errors():
    """thin <= 0, n_chains <= 0, n_draws < 2, n_burn < 0, NaN data and lags too
    large for the sample must raise ValueError (old code: IndexError, an
    internal 'chains must be 2-D' error, a reshape crash, a raw LinAlgError
    from lstsq, and a silent run with k > T_eff respectively)."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((70, 2))
    with pytest.raises(ValueError, match="thin must be >= 1"):
        bvar_sv(Y, lags=1, n_draws=10, n_burn=2, thin=0)
    with pytest.raises(ValueError, match="n_chains must be >= 1"):
        bvar_sv(Y, lags=1, n_draws=10, n_burn=2, n_chains=0)
    with pytest.raises(ValueError, match="n_draws must be >= 2"):
        bvar_sv(Y, lags=1, n_draws=1, n_burn=2)
    with pytest.raises(ValueError, match="n_burn must be >= 0"):
        bvar_sv(Y, lags=1, n_draws=10, n_burn=-1)
    Y_nan = Y.copy()
    Y_nan[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        bvar_sv(Y_nan, lags=1, n_draws=10, n_burn=2)
    Y_big = rng.standard_normal((120, 2))
    with pytest.raises(ValueError, match="regressors per equation"):
        bvar_sv(Y_big, lags=70, n_draws=10, n_burn=2)


def test_stability_rejections_are_counted_and_warned():
    """On a mildly explosive VAR(1) the stability rejection loop exhausts its
    budget; the sampler must warn and expose the counts instead of silently
    keeping the previous B (old code: bare `pass`, no warning, no counter)."""
    rng = np.random.default_rng(0)
    T = 80
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = 1.05 * Y[t - 1] + rng.standard_normal(2)
    with pytest.warns(UserWarning, match="no stable VAR coefficient draw"):
        res = bvar_sv(Y, lags=1, n_draws=50, n_burn=20, n_chains=1, seed=1)
    assert res.n_stuck_iterations > 0
    assert res.n_unstable_rejections >= 50 * res.n_stuck_iterations
    assert "Stability rejections" in res.summary()

    # A well-behaved system emits no warning and reports zero stuck sweeps
    Y_ok = rng.standard_normal((80, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res_ok = bvar_sv(Y_ok, lags=1, n_draws=20, n_burn=5, n_chains=1, seed=1)
    assert res_ok.n_stuck_iterations == 0
    assert "Stability rejections" not in res_ok.summary()


def test_result_exposes_documented_aliases():
    """rhat (dict), max_rhat (float) and log_score() must exist (documented usage
    crashed with AttributeError on the old result class)."""
    Y = np.random.default_rng(0).standard_normal((70, 2))
    res = bvar_sv(Y, lags=1, n_draws=20, n_burn=5, seed=1)
    assert isinstance(res.rhat, dict)
    assert res.rhat is res.r_hat
    assert isinstance(res.max_rhat, float)
    assert res.max_rhat == res.r_hat["max"]
    assert res.log_score() == res.predictive_log_score()


def test_irf_to_frame_honours_single_index_filter():
    """to_frame(target_idx=i) / to_frame(shock_idx=j) filter on the given index
    alone (old code only filtered when both were passed and otherwise
    returned every pair)."""
    Y = np.random.default_rng(0).standard_normal((70, 2))
    res = bvar_sv(Y, lags=1, n_draws=20, n_burn=5, n_chains=1, seed=1)
    irf = res.irf(horizon=3)
    df_t = irf.to_frame(target_idx=0, names=res.names)
    assert len(df_t) == 4 * 2
    assert set(df_t["target"]) == {"y1"} and set(df_t["shock"]) == {"y1", "y2"}
    df_s = irf.to_frame(shock_idx=1, names=res.names)
    assert len(df_s) == 4 * 2
    assert set(df_s["shock"]) == {"y2"} and set(df_s["target"]) == {"y1", "y2"}
    assert len(irf.to_frame()) == 4 * 4
    assert len(irf.to_frame(0, 1)) == 4


def test_intercept_prior_std_honoured_without_minnesota_prior():
    """With minnesota_prior=False the intercept prior used to be hard-wired to
    N(0, 1e4); intercept_prior_std must be honoured in both prior modes."""
    Y = np.random.default_rng(0).standard_normal((80, 2)) + 3.0
    r_wide = bvar_sv(Y, lags=1, n_draws=40, n_burn=10, n_chains=1, minnesota_prior=False,
                     intercept_prior_std=1e3, seed=5)
    r_tight = bvar_sv(Y, lags=1, n_draws=40, n_burn=10, n_chains=1, minnesota_prior=False,
                      intercept_prior_std=1e-3, seed=5)
    assert not np.array_equal(r_wide.beta_draws, r_tight.beta_draws)
    assert np.all(np.abs(r_tight.intercept_draws) < 0.05)
    assert np.mean(r_wide.intercept_draws) > 0.5


def test_holdout_log_score_is_out_of_sample():
    """log_score(holdout) evaluates observations that were not used in
    estimation, projecting the volatility forward from the last in-sample
    state; it must be a finite float, reproducible for a seed, and distinct
    from the in-sample lppd."""
    rng = np.random.default_rng(3)
    T = 130
    Y = np.zeros((T, 2))
    h = np.zeros((T, 2))
    for t in range(1, T):
        h[t] = 0.9 * h[t - 1] + 0.3 * rng.standard_normal(2)
        Y[t] = 0.5 * Y[t - 1] + np.exp(h[t] / 2.0) * rng.standard_normal(2)
    df = pd.DataFrame(Y, columns=["a", "b"])
    train, test = df.iloc[:118], df.iloc[118:]
    res = bvar_sv(train, lags=1, n_draws=150, n_burn=100, n_chains=1, seed=2)

    score = res.log_score(test, seed=0)
    assert isinstance(score, float) and np.isfinite(score)
    assert score == res.log_score(test, seed=0)
    pointwise = res.log_score(test, point_by_point=True, seed=0)
    assert pointwise.shape == (len(test),)
    np.testing.assert_allclose(pointwise.sum(), score)
    assert score != res.predictive_log_score()

    # Hand computation of the first hold-out observation with the same projection
    rng2 = np.random.default_rng(0)
    D = res.n_total_draws
    h_t = res.mu_draws + res.phi_draws * (res.h_draws[:, -1, :] - res.mu_draws) \
        + res.sigma_h_draws * rng2.standard_normal((D, 2))
    x_t = np.concatenate([[1.0], train.values[-1]])
    u = test.values[0][None, :] - res.beta_draws.transpose(0, 2, 1) @ x_t
    nu = np.einsum("dij,dj->di", res.a_draws, u)
    lp = -np.log(2 * np.pi) - 0.5 * h_t.sum(1) - 0.5 * (np.exp(-h_t) * nu ** 2).sum(1)
    np.testing.assert_allclose(pointwise[0], logsumexp(lp) - np.log(D))

    with pytest.raises(ValueError, match="holdout must have shape"):
        res.log_score(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="at least one observation"):
        res.log_score(np.zeros((0, 2)))


def test_forecast_and_fan_chart():
    """forecast(horizon) simulates posterior-predictive paths (parameter,
    volatility and shock uncertainty) with dated index and fan charts."""
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((90, 2)).cumsum(0) * 0.05 + rng.standard_normal((90, 2))
    df = pd.DataFrame(Y, index=pd.date_range("2000-01-01", periods=90, freq="QE"), columns=["a", "b"])
    res = bvar_sv(df, lags=2, n_draws=60, n_burn=20, n_chains=1, seed=4)

    fc = res.forecast(6, ci=0.8, seed=9)
    assert isinstance(fc, BVAR_SVForecast)
    assert fc.paths.shape == (60, 6, 2)
    assert fc.h_paths.shape == (60, 6, 2)
    assert fc.horizon == 6
    assert fc.median.shape == (6, 2) and fc.lower.shape == (6, 2) and fc.upper.shape == (6, 2)
    assert np.all(fc.lower <= fc.median) and np.all(fc.median <= fc.upper)
    assert fc.index[0] == pd.Timestamp("2022-09-30") and len(fc.index) == 6
    np.testing.assert_allclose(fc.quantile(0.5), fc.median)
    # Wider band for a wider ci
    fc95 = res.forecast(6, ci=0.95, seed=9)
    assert np.all((fc95.upper - fc95.lower) >= (fc.upper - fc.lower) - 1e-12)
    # Reproducible for a seed
    np.testing.assert_array_equal(fc.paths, fc95.paths)

    tab = fc.to_frame()
    assert list(tab.columns) == ["horizon", "period", "variable", "median", "mean", "lower", "upper"]
    assert len(tab) == 12

    fig = fc.plot()
    assert len(fig.axes) == 2
    plt.close("all")
    fig = fc.plot(var_idx=1, levels=(0.5, 0.9))
    assert len(fig.axes) == 1
    plt.close("all")
    _, axes = plt.subplots(1, 2)
    assert fc.plot(ax=axes) is axes
    plt.close("all")
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        res.forecast(0)

    # Integer index when the data carries no dates
    res_int = bvar_sv(Y, lags=1, n_draws=10, n_burn=2, n_chains=1, seed=4)
    fc_int = res_int.forecast(3, seed=0)
    assert list(fc_int.index) == [90, 91, 92]
