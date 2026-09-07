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
    # mean/std are Dynare's PRIOR_P1/PRIOR_P2 -- the mean and sd of the
    # prior itself. `.s` / `.nu` are Dynare's INTERNAL pair, obtained from
    # inverse_gamma_specification. Before 2.4.1 the constructor stored
    # (mean, std) verbatim as (s, nu), so `ig.s` was 0.2 and `ig.nu` 2.0.
    assert ig.mean == 0.2
    assert ig.std == 2.0
    assert ig.s == pytest.approx(0.0256894, rel=1e-5)
    assert ig.nu == pytest.approx(2.0063588, rel=1e-6)
    assert ig.logpdf(0.2) > -np.inf
    assert ig.logpdf(-0.1) == -np.inf

    # The explicit (s, nu) form round-trips back to the same prior.
    ig2 = InvGammaPrior(s=ig.s, nu=ig.nu, lb=0.01, ub=5.0)
    assert ig2.mean == pytest.approx(0.2, rel=1e-9)
    assert ig2.std == pytest.approx(2.0, rel=1e-9)
    assert ig2.logpdf(0.35) == pytest.approx(ig.logpdf(0.35), rel=1e-12)
    with pytest.raises(ValueError, match="not both"):
        InvGammaPrior(mean=0.2, std=2.0, s=0.1, nu=3.0)

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


# ---------------------------------------------------------------------------
# 2.4.1 regression tests (DSGE estimation audit)
# ---------------------------------------------------------------------------

def test_a_broken_log_likelihood_raises_instead_of_returning_a_result():
    """`except Exception: return -inf` around the user's log_likelihood_fn
    turned an outright bug into a 'result': mode [0.], acceptance 0.0, a
    constant chain and a posterior std of 0, with no warning at all."""
    calls = {"n": 0}

    def broken_log_likelihood(params):
        calls["n"] += 1
        # A plain bug, not an infeasible draw.
        return params.this_attribute_does_not_exist

    priors = {"theta": NormalPrior(mean=0.0, std=10.0, lb=0.0, ub=5.0)}
    with pytest.raises(AttributeError):
        estimate_dsge_bayesian(
            log_likelihood_fn=broken_log_likelihood,
            priors=priors,
            n_draws=20, n_burn=5, n_chains=1, seed=0,
        )
    assert calls["n"] <= 3, (
        f"the broken likelihood was called {calls['n']} times before the "
        "error surfaced; it should fail on the first evaluation"
    )


def test_persistently_infeasible_likelihood_raises_rather_than_flatlining():
    def always_infeasible(theta):
        raise np.linalg.LinAlgError("filter F is singular")

    priors = {"theta": NormalPrior(mean=0.0, std=1.0, lb=-2.0, ub=2.0)}
    with pytest.raises(RuntimeError, match="consecutive draws"):
        estimate_dsge_bayesian(
            log_likelihood_fn=always_infeasible,
            priors=priors,
            n_draws=2000, n_burn=100, n_chains=1, seed=0,
        )


def test_boundary_mode_reports_nan_se_and_a_moving_chain():
    """A monotone likelihood pushes the mode onto the lower prior bound.

    There the Hessian stencil measures the 1e20 penalty cliff, not
    curvature: mode_se floored at 1e-6 and -- because the same matrix built
    the proposal -- the chain froze, reporting a posterior std of ~4e-12
    for a posterior whose true sd is ~0.1.
    """
    def monotone_ll(theta):
        return -10.0 * float(theta[0])

    priors = {"theta": NormalPrior(mean=0.0, std=10.0, lb=0.0, ub=5.0)}
    with pytest.warns(UserWarning, match="Laplace approximation is undefined"):
        res = estimate_dsge_bayesian(
            log_likelihood_fn=monotone_ll,
            priors=priors,
            n_draws=2000, n_burn=500, n_chains=2, seed=0,
        )
    assert res.mode[0] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(res.mode_se[0]), (
        f"mode_se={res.mode_se} pretends to be a Laplace SE at a boundary mode"
    )
    # The true posterior is ~Exp(10) truncated to [0, 5]: sd ~ 0.1.
    post_sd = float(res.summary().loc["theta", "std"])
    assert 0.03 < post_sd < 0.4, f"posterior std {post_sd!r} is degenerate"


def test_infeasible_beta_prior_is_rejected_by_name():
    def ll(theta):
        return 0.0

    priors = {"p": {"dist": "beta", "mean": 0.5, "std": 0.6,
                    "lb": 0.0, "ub": 1.0}}
    with pytest.raises(ValueError, match=r"estimate_dsge_bayesian: prior 'p'"):
        estimate_dsge_bayesian(
            log_likelihood_fn=ll, priors=priors,
            n_draws=10, n_burn=5, n_chains=1, seed=0,
        )
