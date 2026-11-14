"""Tests for DSGE prior-posterior diagnostics and empirical IRF validation scorecards."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.dsge import (
    BetaPrior,
    GammaPrior,
    IRFScorecardResult,
    NormalPrior,
    PriorPosteriorComparisonResult,
    bayesian_irf,
    build_dynare,
    load_hank_mod,
    nuts_sample,
    prior_posterior_comparison,
    score_irf_against_targets,
)
from puremacro.dsge.bayesian import BayesianEstimationResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.lp import lp_hac
from puremacro.state_space import StateSpaceModel
from puremacro.structural.bridge import MomentTargets
from puremacro.structural.lp import lp_moment_targets


def _ar1_state_space(params: dict) -> StateSpaceModel:
    rho = params["rho"]
    sigma = params["sigma"]
    return StateSpaceModel(
        T=np.array([[rho]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma ** 2]]),
        H=np.array([[1e-8]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate_ar1(rho_true: float, sigma_true: float, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def test_conjugate_normal_closed_form_parity():
    """Verify NUTS sampling + prior_posterior_comparison match conjugate Normal moments."""
    prior_mu, prior_sd = 1.0, 1.0
    sigma_lik = 0.5
    rng = np.random.default_rng(7)
    y = rng.normal(loc=2.0, scale=sigma_lik, size=40)

    prec_post = (1.0 / prior_sd ** 2) + len(y) / (sigma_lik ** 2)
    post_var = 1.0 / prec_post
    post_mu = post_var * (prior_mu / prior_sd ** 2 + float(y.sum()) / (sigma_lik ** 2))
    post_sd = math.sqrt(post_var)

    def target(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu = float(theta[0])
        lp = -0.5 * ((mu - prior_mu) / prior_sd) ** 2 - 0.5 * float(np.sum(((y - mu) / sigma_lik) ** 2))
        grad = -((mu - prior_mu) / prior_sd ** 2) + float(np.sum((y - mu) / sigma_lik ** 2))
        return lp, np.array([grad], dtype=float)

    priors = {"mu": NormalPrior(mean=prior_mu, std=prior_sd)}
    nuts_res = nuts_sample(
        target,
        init_params=np.array([post_mu]),
        n_draws=400,
        n_chains=2,
        warmup=200,
        param_names=("mu",),
        seed=12,
        priors=priors,
    )
    assert nuts_res.priors is not None

    # Zero-boilerplate: relies on automatically populated nuts_res.priors
    comp = prior_posterior_comparison(nuts_res)
    assert isinstance(comp, PriorPosteriorComparisonResult)
    row = comp.table.iloc[0]

    assert row["prior_status"] == "ok"
    assert abs(row["prior_mean"] - prior_mu) < 1e-12
    assert abs(row["prior_std"] - prior_sd) < 1e-12
    assert abs(row["post_mean"] - post_mu) < 0.03
    assert abs(row["post_std"] - post_sd) < 0.03
    assert abs(row["std_ratio"] - post_sd / prior_sd) < 0.03
    assert row["rhat"] < 1.05
    assert row["ess_bulk"] >= 100.0
    assert row["ess_tail"] >= 100.0
    assert not bool(row["weakly_identified"])
    assert not bool(row["near_boundary"])
    assert not bool(row["unconverged_or_divergent"])


def test_estimate_dsge_populates_priors_zero_boilerplate():
    """Verify estimate_dsge populates result.priors for direct prior_posterior_comparison."""
    priors = {
        "rho": {"dist": "beta", "mean": 0.6, "std": 0.15, "lb": 0.01, "ub": 0.99},
        "sigma": {"dist": "gamma", "mean": 0.4, "std": 0.15, "lb": 0.01, "ub": 3.0},
    }
    data = _simulate_ar1(0.65, 0.35, T=80, seed=4)
    res = estimate_dsge(
        data,
        observation_eq=_ar1_state_space,
        priors=priors,
        observed_vars=["y"],
        initial_params={"rho": 0.55, "sigma": 0.35},
        n_chains=2,
        n_draws=120,
        burn_in=40,
        seed=3,
    )
    assert res.priors is not None and set(res.priors) == {"rho", "sigma"}

    comp = prior_posterior_comparison(res)
    assert list(comp.table["param"]) == ["rho", "sigma"]
    assert np.all(np.isfinite(comp.table["post_mean"]))
    assert np.all(np.isfinite(comp.table["rhat"]))


def test_raw_invgamma_and_truncated_quadrature_parity():
    """Cover raw invgamma (type1 nu<=2, type2 nu<=4) and truncated moment quadrature."""
    rng = np.random.default_rng(19)
    chains = rng.normal(0.5, 0.1, size=(2, 120, 4))

    priors = {
        # Raw type-1 with nu=1.5 <= 2 on finite [0.05, 1.5] -> finite numerical moments
        "ig1_bnd": {"dist": "invgamma", "s": 0.02, "nu": 1.5, "lb": 0.05, "ub": 1.5, "kind": "type1"},
        # Raw type-2 with nu=3.0 <= 4 on unbounded [0, inf) -> prior_std=NaN, undefined_prior_variance
        "ig2_unbnd": {"dist": "invgamma", "s": 0.04, "nu": 3.0, "lb": 0.0, "ub": math.inf, "kind": "type2"},
        # Heavily truncated normal -> exact agreement with scipy.stats.truncnorm
        "trunc_n": {"dist": "normal", "mean": 0.0, "std": 1.0, "lb": 0.2, "ub": 2.0},
        # Slightly truncated gamma (< 1e-6 omitted mass) -> exact closed form
        "clean_g": {"dist": "gamma", "mean": 0.5, "std": 0.1, "lb": 0.0, "ub": math.inf},
    }
    post = BayesianEstimationResult(
        mode={"ig1_bnd": 0.2, "ig2_unbnd": 0.05, "trunc_n": 0.7, "clean_g": 0.5},
        mode_se={"ig1_bnd": 0.05, "ig2_unbnd": 0.05, "trunc_n": 0.1, "clean_g": 0.1},
        param_names=("ig1_bnd", "ig2_unbnd", "trunc_n", "clean_g"),
        log_posterior_mode=0.0,
        chains=chains,
        acceptance_rate=0.35,
        posterior_summary=pd.DataFrame(),
        diagnostics={},
    )

    comp = prior_posterior_comparison(post, priors)
    by_p = comp.table.set_index("param")

    assert by_p.loc["ig1_bnd", "prior_status"] == "truncated_numerical"
    assert math.isfinite(by_p.loc["ig1_bnd", "prior_mean"])
    assert math.isfinite(by_p.loc["ig1_bnd", "prior_std"]) and by_p.loc["ig1_bnd", "prior_std"] > 0.0

    # Exact closed-form cross-check via Y = 1/X^2 ~ Gamma(a=nu/2, scale=2/s) on grid
    grid = np.linspace(0.05, 1.5, 4000)
    dx = grid[1] - grid[0]
    w = (grid ** (-(1.5 + 1.0))) * np.exp(-0.02 / (2.0 * grid ** 2))
    w /= np.sum(w * dx)
    ref_mean = float(np.sum(grid * w * dx))
    ref_std = math.sqrt(float(np.sum(((grid - ref_mean) ** 2) * w * dx)))
    assert abs(by_p.loc["ig1_bnd", "prior_mean"] - ref_mean) < 1e-3
    assert abs(by_p.loc["ig1_bnd", "prior_std"] - ref_std) < 1e-3

    assert by_p.loc["ig2_unbnd", "prior_status"] == "undefined_prior_variance"
    assert math.isnan(by_p.loc["ig2_unbnd", "prior_std"])
    assert not bool(by_p.loc["ig2_unbnd", "weakly_identified"])

    tn_mean, tn_var = stats.truncnorm.stats(0.2, 2.0, loc=0.0, scale=1.0, moments="mv")
    assert by_p.loc["trunc_n", "prior_status"] == "truncated_numerical"
    assert abs(by_p.loc["trunc_n", "prior_mean"] - float(tn_mean)) < 1e-7
    assert abs(by_p.loc["trunc_n", "prior_std"] - math.sqrt(float(tn_var))) < 1e-7

    assert by_p.loc["clean_g", "prior_status"] == "ok"
    assert abs(by_p.loc["clean_g", "prior_mean"] - 0.5) < 1e-12
    assert abs(by_p.loc["clean_g", "prior_std"] - 0.1) < 1e-12


def test_diagnostic_flags_rhat_ess_divergence_weak_and_boundary():
    """Verify Vehtari R-hat, bulk/tail ESS, divergence rate, weak ID, and boundary flags."""
    rng = np.random.default_rng(23)
    # 4 params, 2 chains, 80 draws
    draws = np.zeros((2, 80, 4))
    # p_unconv: two non-mixing parallel chains -> high R-hat
    draws[0, :, 0] = rng.normal(0.20, 0.02, size=80)
    draws[1, :, 0] = rng.normal(0.80, 0.02, size=80)
    # p_weak: posterior spread equals prior spread (mean~0.5, std~0.1) & random walk low ESS
    draws[:, :, 1] = np.clip(0.5 + 0.1 * np.cumsum(rng.normal(0.0, 0.02, size=(2, 80)), axis=1), 0.1, 0.9)
    # p_bnd: concentrated right against lower bound 0.0 of Uniform(0, 1)
    draws[:, :, 2] = rng.uniform(0.001, 0.012, size=(2, 80))
    # p_div: normal chain, tested with divergent NUTSResult
    draws[:, :, 3] = rng.normal(0.5, 0.02, size=(2, 80))

    divs = np.zeros((2, 80), dtype=bool)
    divs[:, :10] = True  # 12.5% divergences > 1%

    priors = {
        "p_unconv": BetaPrior(mean=0.5, std=0.2),
        "p_weak": BetaPrior(mean=0.5, std=0.1),
        "p_bnd": {"dist": "uniform", "lb": 0.0, "ub": 1.0},
        "p_div": NormalPrior(mean=0.5, std=0.3),
    }

    nuts_obj = nuts_sample(
        lambda x: (-0.5 * float(np.sum((x - 0.5) ** 2)), -(x - 0.5)),
        init_params=np.zeros(4),
        n_draws=20,
        n_chains=2,
        warmup=10,
        param_names=("p_unconv", "p_weak", "p_bnd", "p_div"),
        seed=5,
        priors=priors,
    )
    # Swap in controlled multichain NUTSResult draws & divergence mask via real dataclass constructor
    from dataclasses import replace

    controlled_nuts = replace(
        nuts_obj,
        draws=draws,
        divergences=divs,
        priors=priors,
    )
    comp = prior_posterior_comparison(controlled_nuts, ess_threshold=40.0)
    by_p = comp.table.set_index("param")

    assert by_p.loc["p_unconv", "rhat"] > 1.05
    assert bool(by_p.loc["p_unconv", "unconverged_or_divergent"])
    assert bool(by_p.loc["p_bnd", "near_boundary"])
    assert bool(by_p.loc["p_div", "unconverged_or_divergent"])
    assert "unconverged_or_divergent" in by_p.loc["p_div", "flags"]

    # Export quintet + .plot(ax=ax) is ax
    assert isinstance(comp.summary(), str) and "Prior-Posterior Comparison" in comp.summary()
    assert isinstance(comp.to_frame(), pd.DataFrame)
    assert isinstance(comp.to_dataframe(), pd.DataFrame)
    assert "|" in comp.to_markdown()
    assert "\\begin{tabular}" in comp.to_latex()
    assert "#table(" in comp.to_typst()

    fig = comp.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert comp.plot(ax=ax) is ax


def _build_small_nk_model():
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
    }
    variables = ["y", "pi", "r"]
    shocks = ["e_m"]
    steady_state = {v: 0.0 for v in variables}

    def eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
        ]

    m = build_dynare(
        eqs,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    priors = {
        "kappa": GammaPrior(mean=0.15, std=0.04),
        "rho_r": BetaPrior(mean=0.70, std=0.08),
        "phi_pi": NormalPrior(mean=1.50, std=0.15),
    }
    return m, priors


def test_score_irf_bayesian_from_irf_lp_targets_and_rank_deficient_cov():
    """Score BayesianIRFResult against MomentTargets.from_irf, lp_moment_targets, and Mapping[LPResult]."""
    m, priors = _build_small_nk_model()
    b_irf = bayesian_irf(m, priors=priors, shock="e_m", periods=6, n_draws=20, bands=(0.90,), seed=11)

    # 1. MomentTargets.from_irf ('y<-e_m[h=0]' label style) + rank-deficient covariance
    cov_full = np.eye(5) * 0.04
    cov_full[3, :] = 0.0
    cov_full[:, 3] = 0.0  # singular mode -> rank 4 < 5
    cov_full[0, 1] = cov_full[1, 0] = 0.015

    # Exact self-match should yield near-zero RMSE and chi2_stat == 0, chi2_pvalue == 1.0
    t_self = MomentTargets.from_irf(
        b_irf.median["y"].iloc[:5].to_numpy(),
        covariance=cov_full,
        response="y",
        shock="e_m",
        units="pct",
        metadata={"shock_size": 1.0},
    )
    sc_self = score_irf_against_targets(b_irf, t_self)
    assert isinstance(sc_self, IRFScorecardResult)
    assert sc_self.weight_mode_used == "covariance"
    assert sc_self.chi2_df == 4
    assert sc_self.chi2_stat < 1e-12
    assert abs(sc_self.chi2_pvalue - 1.0) < 1e-12
    assert sc_self.sign_concordance == 1.0
    assert sc_self.band_coverage == 1.0
    assert math.isfinite(sc_self.determinacy_rate)
    assert any("rank-deficient" in w for w in sc_self.warnings)

    # Non-zero discrepancy cross-check vs chi2.sf
    perturbed_vals = t_self.values + np.array([0.1, -0.05, 0.02, 0.5, 0.04])
    t_pert = MomentTargets(values=perturbed_vals, covariance=cov_full, labels=t_self.labels, metadata=t_self.metadata)
    sc_pert = score_irf_against_targets(b_irf, t_pert)
    assert sc_pert.chi2_df == 4
    assert abs(sc_pert.chi2_pvalue - float(stats.chi2.sf(sc_pert.chi2_stat, df=4))) < 1e-12

    # 2. lp_moment_targets ('y:eps:h=0' label style)
    rng = np.random.default_rng(31)
    T_obs = 60
    eps_s = rng.standard_normal(T_obs)
    y_s = np.zeros(T_obs)
    pi_s = np.zeros(T_obs)
    for t in range(1, T_obs):
        y_s[t] = 0.5 * y_s[t - 1] - 0.3 * eps_s[t] + 0.2 * rng.standard_normal()
        pi_s[t] = 0.4 * pi_s[t - 1] - 0.1 * eps_s[t] + 0.1 * rng.standard_normal()
    lp_df = pd.DataFrame({"y": y_s, "pi": pi_s, "e_m": eps_s})

    mt_lp = lp_moment_targets(
        lp_df,
        responses=("y", "pi"),
        shock="e_m",
        response_units={"y": "pct", "pi": "pct"},
        shock_unit="pct",
        frequency="QE",
        horizons=(0, 1, 2, 3),
        lags=1,
    )
    sc_lp = score_irf_against_targets(b_irf, mt_lp)
    assert len(sc_lp.table) == 8
    assert sc_lp.weight_mode_used == "covariance"
    assert math.isfinite(sc_lp.wse) and math.isfinite(sc_lp.chi2_pvalue)

    # 3. Mapping[str, LPResult] -> diagonal weights and chi2_pvalue=NaN
    lp_y = lp_hac(lp_df, "y", "e_m", horizons=range(4), lags=1)
    lp_pi = lp_hac(lp_df, "pi", "e_m", horizons=range(4), lags=1)
    sc_diag = score_irf_against_targets(b_irf, {"y": lp_y, "pi": lp_pi})
    assert sc_diag.weight_mode_used == "diagonal"
    assert math.isfinite(sc_diag.wse)
    assert math.isnan(sc_diag.chi2_pvalue)
    with pytest.raises(ValueError, match="covariance"):
        score_irf_against_targets(b_irf, {"y": lp_y}, weight_mode="covariance")

    # Quintet + .plot(ax=ax) is ax
    assert isinstance(sc_lp.summary(), str) and "IRF Validation Scorecard" in sc_lp.summary()
    assert isinstance(sc_lp.to_frame(), pd.DataFrame)
    assert isinstance(sc_lp.to_dataframe("metrics"), pd.DataFrame)
    assert "|" in sc_lp.to_markdown()
    assert "\\begin{tabular}" in sc_lp.to_latex()
    assert "#table(" in sc_lp.to_typst()

    fig = sc_lp.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    _, ax = plt.subplots()
    assert sc_lp.plot(variable="y", ax=ax) is ax


def test_score_irf_hank_sign_scale_warning():
    """Verify HANKResult negative/small default shock magnitude triggers clear warning."""
    mod_text = """
    var Y C pi i r;
    varexo eps_m;
    parameters beta gamma phi_pi kappa r_ss;
    beta = 0.96;
    gamma = 2.0;
    phi_pi = 1.5;
    kappa = 0.1;
    r_ss = 0.02;
    model;
        Y = C;
        pi = beta * pi(+1) + kappa * Y;
        i = phi_pi * pi + eps_m;
        r = i - pi(+1);
    end;
    hetagent_block;
        model = hank_sequence_space;
    end;
    """
    hank_m = load_hank_mod(mod_text)
    hank_res = hank_m.simulate(shock="eps_m", magnitude=-0.0025, horizon=8)

    t_unit = MomentTargets.from_irf(
        np.array([-0.5, -0.3, -0.15, -0.05]),
        covariance=np.eye(4) * 0.05,
        response="Y",
        shock="eps_m",
        units="pct",
        metadata={"shock_size": 1.0},
    )
    sc_warn = score_irf_against_targets(hank_res, t_unit, hank_unit_mode="warn")
    assert any("opposite sign" in w.lower() for w in sc_warn.warnings)
    assert any("scale" in w.lower() for w in sc_warn.warnings)

    sc_scaled = score_irf_against_targets(hank_res, t_unit, hank_unit_mode="scale_to_target")
    assert any("Scaled HANKResult" in w for w in sc_scaled.warnings)
