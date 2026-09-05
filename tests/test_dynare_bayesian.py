"""Tests for general Bayesian DSGE Estimation pipeline (puremacro.dsge.bayesian)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    BayesianEstimationResult,
    estimate_dsge_bayesian,
    BetaPrior,
    InvGammaPrior,
    NormalPrior,
    GammaPrior,
    UniformPrior,
)
from puremacro.state_space import StateSpaceModel, kalman_filter


def _generate_synthetic_ar1(
    rho: float = 0.7,
    sigma: float = 0.4,
    T: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic stationary AR(1) state-space series."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + eps[t]
    return x[:, None]


def test_estimate_dsge_bayesian_synthetic_ar1():
    """Test mode finding, Laplace approximation, and RWMH sampling on synthetic DSGE."""
    rho_true = 0.7
    sigma_true = 0.4
    y = _generate_synthetic_ar1(rho=rho_true, sigma=sigma_true, T=300, seed=42)

    def log_likelihood_fn(params):
        if isinstance(params, dict):
            rho = params["rho"]
            sigma = params["sigma"]
        else:
            rho, sigma = params[0], params[1]

        ssm = StateSpaceModel(
            T=np.array([[rho]]),
            Z=np.array([[1.0]]),
            R=np.array([[1.0]]),
            Q=np.array([[sigma ** 2]]),
            H=np.array([[1e-6]]),
        )
        return kalman_filter(y, ssm)["loglik"]

    priors = {
        "rho": BetaPrior(mean=0.6, std=0.15, lb=0.01, ub=0.99),
        "sigma": InvGammaPrior(mean=0.3, std=2.0, lb=0.01, ub=3.0),
    }

    res = estimate_dsge_bayesian(
        log_likelihood_fn=log_likelihood_fn,
        priors=priors,
        initial_params=np.array([0.6, 0.3]),
        n_draws=300,
        n_burn=50,
        n_chains=2,
        target_accept=0.28,
        tune_interval=20,
        seed=42,
    )

    # 1. Verify result type and structure
    assert isinstance(res, BayesianEstimationResult)
    assert res.param_names == ["rho", "sigma"]
    assert res.chains.shape == (2, 300, 2)
    assert np.all(np.isfinite(res.chains))

    # 2. Mode is close to DGP parameters within 2 standard deviations
    dgp = np.array([rho_true, sigma_true])
    diff = np.abs(res.mode - dgp)
    assert np.all(diff <= 2.0 * res.mode_se), (
        f"Mode {res.mode} is farther than 2 SE ({res.mode_se}) from DGP {dgp}. "
        f"Diff / SE = {diff / res.mode_se}"
    )

    # 3. Acceptance rate is within [0.15, 0.45]
    assert 0.15 <= res.acceptance_rate <= 0.45, (
        f"Acceptance rate {res.acceptance_rate:.3f} outside [0.15, 0.45]"
    )

    # 4. Summary table checks
    summary = res.summary()
    assert isinstance(summary, pd.DataFrame)
    expected_cols = ["mean", "std", "16%", "50%", "84%", "5%", "95%"]
    assert list(summary.columns) == expected_cols
    assert list(summary.index) == ["rho", "sigma"]
    assert summary.loc["rho", "mean"] > 0.5
    assert summary.loc["sigma", "mean"] > 0.2

    # to_frame() matches summary()
    pd.testing.assert_frame_equal(res.to_frame(), summary)

    # 5. Output formatters
    latex = res.to_latex()
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex
    assert "rho" in latex and "sigma" in latex

    typst = res.to_typst()
    assert isinstance(typst, str)
    assert "#table(" in typst
    assert "rho" in typst and "sigma" in typst

    md = res.to_markdown()
    assert isinstance(md, str)
    assert "mean" in md and "std" in md
    assert "rho" in md and "sigma" in md

    # 6. Plotting methods
    fig_priors_post = res.plot_priors_posteriors()
    assert isinstance(fig_priors_post, Figure)
    plt.close(fig_priors_post)

    fig_post = res.plot_posteriors(style="publication")
    assert isinstance(fig_post, Figure)
    plt.close(fig_post)

    # 7. Diagnostics
    assert "acceptance_rate" in res.diagnostics
    assert "r_hat_rho" in res.diagnostics
    assert "r_hat_sigma" in res.diagnostics
    assert "geweke_z_rho" in res.diagnostics
    assert "geweke_z_sigma" in res.diagnostics
    assert res.diagnostics["r_hat_max"] < 1.25


def test_bayesian_priors_classes():
    """Verify Prior classes behave as expected."""
    beta = BetaPrior(mean=0.7, std=0.1, lb=0.01, ub=0.99)
    assert beta.dist == "beta"
    assert beta["mean"] == 0.7
    assert beta["lb"] == 0.01
    assert "std" in beta
    assert beta.logpdf(0.7) > -np.inf
    assert beta.logpdf(1.5) == -np.inf  # Out of bounds
    assert beta.pdf(0.7) > 0.0

    ig = InvGammaPrior(mean=0.2, std=2.0, lb=0.01, ub=5.0)
    assert ig.dist == "invgamma"
    assert ig.s == 0.2
    assert ig.nu == 2.0
    assert ig.logpdf(0.2) > -np.inf
    assert ig.logpdf(-0.1) == -np.inf

    norm = NormalPrior(mean=0.0, std=1.0)
    assert norm.dist == "normal"
    assert norm.pdf(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi), rel=1e-3)

    gamma = GammaPrior(mean=1.0, std=0.5, lb=0.001)
    assert gamma.dist == "gamma"
    assert gamma.logpdf(1.0) > -np.inf

    uni = UniformPrior(lb=0.0, ub=2.0)
    assert uni.dist == "uniform"
    assert uni.pdf(1.0) == pytest.approx(0.5)
    assert uni.logpdf(2.5) == -np.inf


def test_bayesian_estimation_result_frozen():
    """Verify BayesianEstimationResult is immutable."""
    dummy_summary = pd.DataFrame({"mean": [1.0]}, index=["theta"])
    res = BayesianEstimationResult(
        mode=np.array([1.0]),
        mode_se=np.array([0.1]),
        param_names=["theta"],
        log_posterior_mode=-10.0,
        chains=np.zeros((1, 10, 1)),
        acceptance_rate=0.28,
        posterior_summary=dummy_summary,
        diagnostics={"r_hat": 1.0},
    )
    with pytest.raises(Exception):
        res.acceptance_rate = 0.5
