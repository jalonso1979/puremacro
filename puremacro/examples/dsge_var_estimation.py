"""DSGE-VAR Hybrid Estimation & Marginal Data Density Optimization.

Demonstrates the Del Negro & Schorfheide (2004) DSGE-VAR(lambda) methodology:
1. Specifying a structural 3-equation New Keynesian DSGE model with technology,
   cost-push, and monetary policy innovations.
2. Simulating synthetic macroeconomic quarterly time series for output gap (y),
   inflation (pi), and policy rate (r).
3. Mapping DSGE cross-equation autocovariances Gamma_k(theta) into an informative
   Normal-Inverted-Wishart prior on an unrestricted VAR(p).
4. Evaluating the closed-form log marginal data density ln p(Y | lambda, theta)
   across a grid of prior weight parameters lambda in [0.2, 5.0].
5. Optimizing the continuous misspecification metric hat{lambda} via Brent line search.
6. Computing structural impulse response functions under the DSGE structural rotation Q*
   and out-of-sample forecasts with analytical confidence bands.
7. Generating and saving publication-ready diagnostic plots to output/dsge_var_estimation.png.

Run:
    python -m puremacro.examples.dsge_var_estimation

Español
-------
Estimación híbrida DSGE-VAR y optimización de densidad marginal de los datos.

Demuestra la metodología DSGE-VAR(lambda) de Del Negro & Schorfheide (2004):
1. Especificación de un modelo DSGE neokeynesiano estructural de 3 ecuaciones con
   choques de tecnología, costos y política monetaria.
2. Simulación de series temporales macroeconómicas trimestrales sintéticas de la brecha
   del producto (y), inflación (pi) y tasa de interés (r).
3. Mapeo de las autocovarianzas teóricas Gamma_k(theta) del DSGE a una distribución
   a priori conjugada Normal-Wishart Invertida sobre un VAR(p) no restringido.
4. Evaluación en forma cerrada de la log-densidad marginal de los datos ln p(Y | lambda, theta)
   sobre una rejilla del parámetro de peso a priori lambda en [0.2, 5.0].
5. Optimización del estadístico de desalineación estructural hat{lambda} mediante búsqueda de Brent.
6. Cálculo de funciones de impulso-respuesta estructurales con rotación DSGE Q* y
   pronósticos fuera de muestra con bandas de confianza analíticas.
7. Generación y guardado de gráficos de diagnóstico en output/dsge_var_estimation.png.

Ejecución:
    python -m puremacro.examples.dsge_var_estimation
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.dsge_var import DSGEVARResult, estimate_dsge_var


# ------------------------------------------------------------------------------
# 1. Structural Model Specifications (True DGP & Informative DSGE Prior)
# ------------------------------------------------------------------------------

# True Data Generating Process (used to generate synthetic observations)
NK_DGP_MOD = """
var y pi r a u;
varexo eps_a eps_u eps_r;

parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
beta   = 0.99;
sigma  = 1.00;
kappa  = 0.50;
phi_pi = 1.50;
phi_y  = 0.50;
rho_a  = 0.75;  // True technology persistence
rho_u  = 0.65;  // True cost-push persistence

model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y + eps_r;
  a = rho_a*a(-1) + eps_a;
  u = rho_u*u(-1) + eps_u;
end;

shocks;
  var eps_a; stderr 0.010;
  var eps_u; stderr 0.010;
  var eps_r; stderr 0.005;
end;
"""

# Structural Prior Model (theoretical DSGE with slight misspecification)
NK_PRIOR_MOD = """
var y pi r a u;
varexo eps_a eps_u eps_r;

parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
beta   = 0.99;
sigma  = 1.00;
kappa  = 0.50;
phi_pi = 1.50;
phi_y  = 0.50;
rho_a  = 0.70;  // Theoretical prior assumption
rho_u  = 0.70;  // Theoretical prior assumption

model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y + eps_r;
  a = rho_a*a(-1) + eps_a;
  u = rho_u*u(-1) + eps_u;
end;

shocks;
  var eps_a; stderr 0.010;
  var eps_u; stderr 0.010;
  var eps_r; stderr 0.005;
end;
"""


def main() -> DSGEVARResult:
    """Execute DSGE-VAR estimation pipeline."""
    print("=" * 80)
    print("PUREMACRO: DSGE-VAR ESTIMATION (DEL NEGRO & SCHORFHEIDE 2004)")
    print("=" * 80)

    # 1. Compile DGP and simulate observations
    m_dgp = build_dynare(NK_DGP_MOD)
    sample_size = 300
    print(f"Simulating {sample_size} quarterly observations from New Keynesian DGP...")
    sim_data = m_dgp.simulate(sample_size, seed=42)[["y", "pi", "r"]]
    print(f"  Summary of simulated series (std dev):")
    for col in sim_data.columns:
        print(f"    {col:4s}: mean={sim_data[col].mean():+.4f}, std={sim_data[col].std():.4f}")
    print("-" * 80)

    # 2. Compile Informative DSGE Prior Model
    m_prior = build_dynare(NK_PRIOR_MOD)
    print("Informative DSGE Prior Model compiled successfully.")
    print(f"  Variables : {list(m_prior.variables)}")
    print(f"  Shocks    : {list(m_prior.shocks)}")
    print("-" * 80)

    # 3. Evaluate Marginal Data Density across candidate lambda grid
    lambda_candidates = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    print(f"Evaluating Log Marginal Data Density over grid lambda in [0.2, 5.0]...")

    res = estimate_dsge_var(
        model=m_prior,
        data=sim_data,
        p=1,
        lamb="optimal",
        lambda_grid=lambda_candidates,
        identification="dsge",
    )

    # 4. Print Summary Table
    print("\n" + res.summary())

    print("\nLog Marginal Data Density Evaluation Grid:")
    if res.log_mdd_grid is not None:
        print(res.log_mdd_grid.round(4).to_string(index=False))

    # 5. Assess Optimal Weight hat_lambda
    hat_lambda = res.hat_lambda
    print("\n" + "=" * 80)
    print(f"OPTIMAL PRIOR WEIGHT (MISSPECIFICATION METRIC): hat_lambda = {hat_lambda:.4f}")
    print(f"  Admissibility lower bound lambda_min = {res.lambda_min:.4f}")
    print(f"  Log MDD at optimum                  = {res.log_mdd:.4f}")
    print("=" * 80)

    assert hat_lambda is not None
    assert 0.2 < hat_lambda < 5.0, f"Expected interior optimum, got {hat_lambda}"

    # Also compute OLS VAR and pure DSGE limits for comparison
    res_ols = estimate_dsge_var(m_prior, sim_data, p=1, lamb=res.lambda_min + 1e-4, check_bounds=False)
    res_dsge = estimate_dsge_var(m_prior, sim_data, p=1, lamb=1000.0, check_bounds=False)

    # 6. Compute Forecasts & Structural IRFs
    print("\nComputing out-of-sample forecasts (8 quarters ahead)...")
    fc = res.forecast(horizon=8, ci=0.90)
    print("  Forecast Mean:")
    print(fc.mean.round(4).to_string())

    # 7. Generate Multi-Panel Publication Figure
    print("\nRendering publication diagnostic figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#fafafa")

    # Panel A: Log Marginal Data Density Surface
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    if res.log_mdd_grid is not None:
        ax_a.plot(
            res.log_mdd_grid["lambda"],
            res.log_mdd_grid["log_mdd"],
            marker="o",
            color="#1f77b4",
            lw=2.0,
            label=r"$\ln p(Y \mid \lambda, \theta)$",
        )
    ax_a.axvline(
        hat_lambda,
        color="#d62728",
        linestyle="--",
        lw=1.8,
        label=f"Optimum $\\hat{{\\lambda}} = {hat_lambda:.2f}$",
    )
    ax_a.set_title(r"(a) Log Marginal Data Density $\ln p(Y \mid \lambda, \theta)$", fontsize=11, fontweight="bold")
    ax_a.set_xlabel(r"Prior Weight $\lambda$")
    ax_a.set_ylabel("Log MDD")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel B: Structural IRF: Output gap to Technology Shock
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    h_max = 12
    irf_opt = res.irf(horizon=h_max, shock="eps_a")
    irf_ols = res_ols.irf(horizon=h_max, shock="eps_a")
    irf_dsge = res_dsge.irf(horizon=h_max, shock="eps_a")

    ax_b.plot(irf_opt.index, irf_opt["y"], color="#1f77b4", lw=2.2, label=rf"DSGE-VAR ($\hat{{\lambda}}={hat_lambda:.2f}$)")
    ax_b.plot(irf_ols.index, irf_ols["y"], color="#7f7f7f", linestyle=":", lw=1.6, label="Unrestricted OLS VAR")
    ax_b.plot(irf_dsge.index, irf_dsge["y"], color="#2ca02c", linestyle="--", lw=1.8, label="Pure DSGE Prior")
    ax_b.axhline(0, color="black", linestyle="-", lw=0.8, alpha=0.6)
    ax_b.set_title(r"(b) Structural IRF: Output Gap ($y$) $\leftarrow \varepsilon_a$", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Horizon (quarters)")
    ax_b.set_ylabel("Percentage Deviation")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="best", frameon=True, fontsize=9)

    # Panel C: Structural IRF: Inflation to Monetary Policy Shock
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    irf_r_opt = res.irf(horizon=h_max, shock="eps_r")
    irf_r_ols = res_ols.irf(horizon=h_max, shock="eps_r")
    irf_r_dsge = res_dsge.irf(horizon=h_max, shock="eps_r")

    ax_c.plot(irf_r_opt.index, irf_r_opt["pi"], color="#d62728", lw=2.2, label=rf"DSGE-VAR ($\hat{{\lambda}}={hat_lambda:.2f}$)")
    ax_c.plot(irf_r_ols.index, irf_r_ols["pi"], color="#7f7f7f", linestyle=":", lw=1.6, label="Unrestricted OLS VAR")
    ax_c.plot(irf_r_dsge.index, irf_r_dsge["pi"], color="#2ca02c", linestyle="--", lw=1.8, label="Pure DSGE Prior")
    ax_c.axhline(0, color="black", linestyle="-", lw=0.8, alpha=0.6)
    ax_c.set_title(r"(c) Structural IRF: Inflation ($\pi$) $\leftarrow \varepsilon_r$", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Horizon (quarters)")
    ax_c.set_ylabel("Percentage Deviation")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="best", frameon=True, fontsize=9)

    # Panel D: Out-of-Sample Forecast of Output Gap with 90% Confidence Interval
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    h_fc = fc.mean.index
    ax_d.plot(h_fc, fc.mean["y"], color="#1f77b4", lw=2.0, marker="o", label="DSGE-VAR Point Forecast")
    ax_d.fill_between(
        h_fc,
        fc.lower["y"],
        fc.upper["y"],
        color="#1f77b4",
        alpha=0.25,
        label="90% Confidence Band",
    )
    ax_d.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.6)
    ax_d.set_title(r"(d) Out-of-Sample Forecast: Output Gap ($y$)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Forecast Horizon (quarters ahead)")
    ax_d.set_ylabel("Deviation")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dsge_var_estimation.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_file}")

    # Also save to local package examples output directory if present
    repo_out = Path(__file__).resolve().parent / "output"
    try:
        repo_out.mkdir(parents=True, exist_ok=True)
        fig.savefig(repo_out / "dsge_var_estimation.png", dpi=150, bbox_inches="tight")
    except Exception:
        pass

    print("\nDSGE-VAR estimation completed successfully.")
    return res


if __name__ == "__main__":
    main()
