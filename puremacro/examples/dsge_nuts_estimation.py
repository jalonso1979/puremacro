"""Bayesian DSGE Estimation via No-U-Turn Sampler (NUTS) Hamiltonian Monte Carlo.

Demonstrates gradient-based Markov Chain Monte Carlo (MCMC) estimation of a
three-equation New Keynesian (NK) DSGE model using the puremacro 3.0.0 NUTS engine:
1. Defining a 3-equation New Keynesian DSGE model in Dynare format.
2. Generating simulated observations for output gap y and inflation pi.
3. Estimating structural parameters (sigma, kappa) using `method="nuts"`
   driven by exact analytical Kalman likelihood score gradients.
4. Printing MCMC diagnostics: split-R_hat, bulk ESS, tail ESS, E-BFMI, and
   acceptance statistics.
5. Saving publication-ready diagnostic plots (traces, posteriors, autocorrelation,
   and energy distributions).

Outputs:
  puremacro/examples/output/dsge_nuts_estimation.png
  puremacro/examples/output/dsge_nuts_trace.png
  puremacro/examples/output/dsge_nuts_posterior.png
  puremacro/examples/output/dsge_nuts_autocorr.png

Español
--------
Demuestra la estimación bayesiana mediante el muestreador No-U-Turn (NUTS)
en un modelo neokeynesiano (NK) de tres ecuaciones con gradientes analíticos
exactos:
1. Definición del modelo DSGE de 3 ecuaciones en formato Dynare.
2. Generación de datos simulados para la brecha de producto y la inflación.
3. Estimación de parámetros estructurales (sigma, kappa) con `method="nuts"`.
4. Diagnósticos MCMC completos: R_hat, ESS bulk/tail, E-BFMI y divergencias.
5. Generación de gráficos de trazas, densidades a posteriori y autocorrelación.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.dynare import load_mod
from puremacro.dsge._results import NUTSResult


# 3-equation New Keynesian DSGE Specification
NK_3EQ_MOD = """
// 3-Equation New Keynesian DSGE Model
// Clarida, Galí & Gertler (1999 JEL), Woodford (2003)

var y pi r g;
varexo eps_r eps_g;

parameters beta sigma kappa phi_pi rho_g;
beta   = 0.99;    // Quarterly discount factor
sigma  = 1.0;     // Intertemporal elasticity of substitution
kappa  = 0.10;    // Slope of New Keynesian Phillips Curve
phi_pi = 1.50;    // Taylor rule inflation coefficient
rho_g  = 0.80;    // Persistence of demand shock

model;
  // 1. Dynamic IS curve
  y = y(+1) - (1/sigma)*(r - pi(+1)) + g;

  // 2. New Keynesian Phillips Curve
  pi = beta*pi(+1) + kappa*y;

  // 3. Taylor monetary policy rule
  r = phi_pi*pi + eps_r;

  // 4. Exogenous demand shock process
  g = rho_g*g(-1) + eps_g;
end;

steady_state_model;
  y  = 0;
  pi = 0;
  r  = 0;
  g  = 0;
end;

varobs y pi;

estimated_params;
  sigma, gamma_pdf, 1.0, 0.20;
  kappa, gamma_pdf, 0.1, 0.03;
end;
"""


def main():
    print("=" * 75)
    print("puremacro 3.0.0 -- DSGE NUTS Hamiltonian Monte Carlo Estimation")
    print("Exact Analytical Likelihood Gradients & Stan-Style Dual Averaging")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # 1. Build and Load 3-Equation NK Model
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading 3-Equation New Keynesian DSGE Model...")
    model = load_mod(NK_3EQ_MOD)
    print(f"  Variables: {model.variables}")
    print(f"  States:    {model.states}")
    print(f"  Controls:  {model.controls}")
    print(f"  Shocks:    {model.shocks}")
    print(f"  Observed:  {model._varobs}")

    # -----------------------------------------------------------------------
    # 2. Simulate Synthetic Observable Data
    # -----------------------------------------------------------------------
    n_periods = 120
    seed_sim = 101
    print(f"\n[Step 2] Simulating {n_periods} quarters of synthetic data (seed={seed_sim})...")
    sim_df = model.simulate(periods=n_periods, seed=seed_sim)
    data = sim_df[["y", "pi"]].copy()

    print("  Data head (y, pi):")
    print(data.head(3).to_string())

    # -----------------------------------------------------------------------
    # 3. Estimate Structural Parameters via NUTS with Analytic Score
    # -----------------------------------------------------------------------
    n_draws = 150
    n_chains = 2
    burn_in = 75
    seed_nuts = 42

    print("\n[Step 3] Estimating DSGE Parameters via NUTS:")
    print("  method='nuts' with exact analytic score gradients via forward Kalman recursion")
    print(f"  Chains: {n_chains} | Draws: {n_draws} | Warmup/burn-in: {burn_in} | Seed: {seed_nuts}")

    res = model.estimate(
        data,
        method="nuts",
        n_draws=n_draws,
        n_chains=n_chains,
        burn_in=burn_in,
        seed=seed_nuts,
    )

    # -----------------------------------------------------------------------
    # 4. Summary Statistics & Convergence Diagnostics
    # -----------------------------------------------------------------------
    print("\n[Step 4] NUTS Posterior Parameter Estimation Summary:")
    summary_df = res.summary()
    print(summary_df.to_string())

    print("\nMarkdown Summary:")
    print(res.to_markdown())

    diag = res.diagnostics
    print("\nNUTS Sampling Diagnostics:")
    print(f"  Divergences:            {diag['n_divergences']} (rate: {diag['divergence_rate']:.3%})")
    print(f"  Mean tree depth:        {diag['mean_tree_depth']:.2f}")
    print(f"  Max tree depth reached: {diag['max_tree_depth_hit_rate']:.1%}")
    print(f"  Mean acceptance rate:   {diag['mean_accept_rate']:.3f} (target: 0.800)")
    print(f"  Adapted step sizes:     {[f'{s:.4f}' for s in diag['step_sizes']]}")
    print(f"  E-BFMI per chain:       {[f'{e:.3f}' for e in diag['ebfmi']]}")

    # Check Gelman-Rubin convergence
    max_rhat = float(summary_df["r_hat"].max())
    min_ess = float(summary_df["ess_bulk"].min())
    print("\nConvergence Assessment:")
    print(f"  Max split-R_hat: {max_rhat:.4f} (threshold < 1.05: {'PASS' if max_rhat < 1.05 else 'CHECK'})")
    print(f"  Min bulk ESS:    {min_ess:.1f}")

    # -----------------------------------------------------------------------
    # 5. Save Publication Figures
    # -----------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 5] Saving diagnostic plots to {out_dir}:")

    # Figure 1: Comprehensive Multi-Panel Estimation Summary
    out_all = out_dir / "dsge_nuts_estimation.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=150)
    fig.patch.set_facecolor("#fafbfc")

    colors = ["#1f77b4", "#ff7f0e"]

    # Panel A: Trace Plot for sigma
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    for c in range(n_chains):
        ax_a.plot(res.draws[c, :, 0], color=colors[c % len(colors)], alpha=0.8, lw=1.2, label=f"Chain {c+1}")
    ax_a.axhline(1.0, color="black", linestyle="--", lw=1.0, label="True sigma (1.0)")
    ax_a.set_title(r"(a) Trace Plot: $\sigma$ (IES Parameter)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Iteration")
    ax_a.set_ylabel(r"$\sigma$")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel B: Trace Plot for kappa
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    for c in range(n_chains):
        ax_b.plot(res.draws[c, :, 1], color=colors[c % len(colors)], alpha=0.8, lw=1.2, label=f"Chain {c+1}")
    ax_b.axhline(0.1, color="black", linestyle="--", lw=1.0, label="True kappa (0.10)")
    ax_b.set_title(r"(b) Trace Plot: $\kappa$ (Phillips Curve Slope)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel(r"$\kappa$")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel C: Posterior Density of sigma
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    sigma_draws = res.draws[:, :, 0].ravel()
    ax_c.hist(sigma_draws, bins=25, density=True, alpha=0.65, color="#1f77b4", edgecolor="white", label="Posterior Draws")
    if res.mode is not None and "sigma" in res.mode:
        ax_c.axvline(res.mode["sigma"], color="red", linestyle="--", lw=1.5, label=f"Mode ({res.mode['sigma']:.3f})")
    ax_c.axvline(1.0, color="green", linestyle=":", lw=1.5, label="True (1.00)")
    ax_c.set_title(r"(c) Posterior Distribution: $\sigma$", fontsize=11, fontweight="bold")
    ax_c.set_xlabel(r"$\sigma$")
    ax_c.set_ylabel("Density")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel D: Posterior Density of kappa
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    kappa_draws = res.draws[:, :, 1].ravel()
    ax_d.hist(kappa_draws, bins=25, density=True, alpha=0.65, color="#2ca02c", edgecolor="white", label="Posterior Draws")
    if res.mode is not None and "kappa" in res.mode:
        ax_d.axvline(res.mode["kappa"], color="red", linestyle="--", lw=1.5, label=f"Mode ({res.mode['kappa']:.3f})")
    ax_d.axvline(0.10, color="green", linestyle=":", lw=1.5, label="True (0.10)")
    ax_d.set_title(r"(d) Posterior Distribution: $\kappa$", fontsize=11, fontweight="bold")
    ax_d.set_xlabel(r"$\kappa$")
    ax_d.set_ylabel("Density")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_all}")

    # Also save native NUTSResult plots
    fig_trace, _ = res.plot_trace()
    fig_trace.savefig(out_dir / "dsge_nuts_trace.png", bbox_inches="tight")
    plt.close(fig_trace)
    print(f"  -> {out_dir / 'dsge_nuts_trace.png'}")

    fig_post, _ = res.plot_posterior()
    fig_post.savefig(out_dir / "dsge_nuts_posterior.png", bbox_inches="tight")
    plt.close(fig_post)
    print(f"  -> {out_dir / 'dsge_nuts_posterior.png'}")

    fig_ac, _ = res.plot_autocorr()
    fig_ac.savefig(out_dir / "dsge_nuts_autocorr.png", bbox_inches="tight")
    plt.close(fig_ac)
    print(f"  -> {out_dir / 'dsge_nuts_autocorr.png'}")

    print("=" * 75)
    print("DSGE NUTS Estimation completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
