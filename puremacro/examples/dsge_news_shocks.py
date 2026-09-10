"""DSGE News & Anticipated Shocks Engine: State Augmentation & Variance Decomposition.

Demonstrates the macroeconomic modeling of news and anticipated structural shocks
(Beaudry & Portier 2006; Schmitt-Grohe & Uribe 2012):
1. Specifying a forward-looking New Keynesian DSGE model with technology shock a_t.
2. Formulating the companion state-space augmentation V_t = K_H V_{t-1} + eta_t
   where K_H is an H x H nilpotent matrix with all eigenvalues identically 0,
   strictly preserving Blanchard-Kahn saddle-path determinacy.
3. Solving for the impulse response function to an anticipated 4-quarter lead news shock:
   news_irf(model, shock="eps_a", lead=4, horizon=20).
4. Demonstrating the three fundamental macroeconomic properties of news shocks:
   a) Zero revision for predetermined physical states (a_t = 0 for t < 4).
   b) Immediate forward-looking jump of controls (output gap y_0, inflation pi_0) at announcement.
   c) Exact physical realization of the shock innovation at date t = 4.
5. Computing the automated forecast error variance decomposition across surprise and news
   components up to lead H=8 via decompose_news().
6. Plotting multi-lead trajectory comparisons via plot_news_vs_surprise(leads=[0, 2, 4, 8])
   and saving publication-ready figures to output/dsge_news_shocks.png.

Run:
    python -m puremacro.examples.dsge_news_shocks

Español
-------
Motor de perturbaciones anticipadas y de noticias (News Shocks) en modelos DSGE.

Demuestra la modelización macroeconómica de noticias y choques anticipados
(Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012):
1. Especificación de un modelo DSGE neokeynesiano con shock tecnológico prospectivo a_t.
2. Formulación del espacio de estados aumentado V_t = K_H V_{t-1} + eta_t donde K_H
   es una matriz nilpotente de orden H con todos sus autovalores idénticamente 0,
   preservando estrictamente la determinabilidad de Blanchard-Kahn.
3. Solución de la función de impulso-respuesta a una noticia con anticipación de 4 trimestres:
   news_irf(model, shock="eps_a", lead=4, horizon=20).
4. Demostración empírica de las tres propiedades macroeconómicas cardinales:
   a) Cero revisión en estados físicos predeterminados (a_t = 0 para t < 4).
   b) Salto inmediato en variables de control prospectivas (brecha y_0, inflación pi_0) en t=0.
   c) Realización física exacta de la perturbación en la fecha convenida t = 4.
5. Descomposición automática de la varianza del error de pronóstico entre componentes sorpresa
   y noticias hasta anticipación H=8 con decompose_news().
6. Comparación gráfica multilead con plot_news_vs_surprise(leads=[0, 2, 4, 8]) y guardado de
   figuras en output/dsge_news_shocks.png.

Ejecución:
    python -m puremacro.examples.dsge_news_shocks
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.news import (
    NewsDecompositionResult,
    NewsIRFResult,
    decompose_news,
    news_irf,
    plot_news_vs_surprise,
)


# ------------------------------------------------------------------------------
# 1. Structural Model: 4-Variable Forward-Looking New Keynesian DSGE Block
# ------------------------------------------------------------------------------

NK_NEWS_MOD = """
// Forward-Looking New Keynesian Model with Exogenous Productivity Shock
var y pi r a;
varexo eps_a;

parameters beta sigma kappa phi_pi rho_a;
beta   = 0.99;   // Quarterly discount factor
sigma  = 1.00;   // Intertemporal elasticity of substitution
kappa  = 0.50;   // Slope of New Keynesian Phillips Curve
phi_pi = 1.50;   // Taylor rule inflation response
rho_a  = 0.85;   // Autoregressive technology persistence

model;
  // 1. Dynamic IS Curve with technological displacement:
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);

  // 2. New Keynesian Phillips Curve:
  pi = beta*pi(+1) + kappa*y;

  // 3. Monetary Policy Rule:
  r = phi_pi*pi;

  // 4. Technology Shock Process (predetermined state):
  a = rho_a*a(-1) + eps_a;
end;

shocks;
  var eps_a; stderr 0.01;
end;
"""


def main() -> tuple[NewsIRFResult, NewsDecompositionResult]:
    """Execute news shocks simulation and decomposition pipeline."""
    print("=" * 80)
    print("PUREMACRO: DSGE NEWS & ANTICIPATED SHOCKS ENGINE")
    print("=" * 80)

    # 1. Compile DSGE model
    model = build_dynare(NK_NEWS_MOD)
    print("Structural Model compiled successfully:")
    print(f"  Variables : {list(model.variables)}")
    print(f"  States    : {list(model.states)}")
    print(f"  Shocks    : {list(model.shocks)}")
    print(f"  BK Status : {'Determinate' if model.is_determinate else 'Indeterminate'}")
    print("-" * 80)

    # 2. Compute News IRF with lead k = 4 periods (1 year ahead announcement)
    lead_k = 4
    horizon = 20
    print(f"Computing News IRF for technology shock 'eps_a' with lead k={lead_k} periods...")
    res_news = news_irf(model, shock="eps_a", lead=lead_k, horizon=horizon, size=1.0)
    res_surp = news_irf(model, shock="eps_a", lead=0, horizon=horizon, size=1.0)

    # 3. Print Summary Table
    print("\n" + res_news.summary())

    # 4. Demonstrate Fundamental Properties of Anticipated News Shocks
    irf = res_news.irf
    print("\n" + "=" * 80)
    print("MATHEMATICAL & NUMERICAL VERIFICATION OF NEWS PROPERTIES")
    print("=" * 80)

    # a) Zero revision for predetermined states before realization (t < k)
    pre_realiz_state = irf.loc[: lead_k - 1, "a"].to_numpy()
    max_pre_rev = float(np.max(np.abs(pre_realiz_state)))
    print(f"  1. Predetermined State (a_t) Revision for t < {lead_k}:")
    print(f"     Max absolute deviation: {max_pre_rev:.2e} (Strictly 0.0 to machine precision)")
    assert max_pre_rev < 1e-12, f"State a_t must not move before realization; max={max_pre_rev}"

    # b) Immediate jump of forward-looking variables at announcement (t=0)
    y_0 = float(irf.loc[0, "y"])
    pi_0 = float(irf.loc[0, "pi"])
    r_0 = float(irf.loc[0, "r"])
    print(f"  2. Immediate Forward-Looking Controls Jump at Announcement (t=0):")
    print(f"     Output Gap (y_0)  : {y_0:+.6f} (≠ 0 jump on impact)")
    print(f"     Inflation (pi_0)   : {pi_0:+.6f} (≠ 0 jump on impact)")
    print(f"     Interest (r_0)    : {r_0:+.6f} (≠ 0 jump on impact)")
    assert abs(pi_0) > 1e-4, "Forward-looking inflation must jump on announcement impact"

    # c) Realization of the shock at date t=k
    a_k = float(irf.loc[lead_k, "a"])
    print(f"  3. Shock Realization at Date t = {lead_k}:")
    print(f"     Realized State (a_{lead_k}) : {a_k:.6f} (Exact unit impulse realization)")
    assert np.isclose(a_k, 1.0, atol=1e-10), f"Technology state must realize 1.0 at t={lead_k}"

    print("-" * 80)
    print("Trajectory Comparison (t=0 to t=6):")
    sub_cols = ["a", "y", "pi", "r"]
    comp_df = irf.loc[:6, sub_cols].copy()
    comp_df["phase"] = [
        "t=0 (announcement)" if h == 0 else (f"t={h} (pre-realiz)" if h < lead_k else (f"t={h} (REALIZATION)" if h == lead_k else f"t={h} (decay)"))
        for h in comp_df.index
    ]
    print(comp_df.round(5).to_string())

    # 5. Forecast Error Variance Decomposition (FEVD) across Surprise and News Leads
    print("\n" + "=" * 80)
    print("FORECAST ERROR VARIANCE DECOMPOSITION: SURPRISE VS NEWS LEADS")
    print("=" * 80)
    decomp = decompose_news(model, shock="eps_a", horizon=horizon, max_lead=8)
    print("\nVariance Shares at Horizon H=20 (Rows strictly sum to 1.0000):")
    print(decomp.variance_shares.round(4).to_string())

    # Verify unit row sums
    row_sums = decomp.variance_shares.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10, err_msg="Variance shares must sum to 1.0")
    print("\n  Confirmed: All variance decomposition rows sum to 1.0000.")

    # 6. Generate Publication Figure comparing Leads [0, 2, 4, 8]
    print("\nRendering multi-lead trajectory comparison figure (leads=[0, 2, 4, 8])...")
    leads_to_plot = (0, 2, 4, 8)
    fig, axes = plot_news_vs_surprise(
        model=model,
        shock="eps_a",
        leads=leads_to_plot,
        variables=["a", "y", "pi", "r"],
        horizon=16,
        size=1.0,
        figsize=(11, 7.5),
    )

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dsge_news_shocks.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_file}")

    # Also save to local package examples output directory if present
    repo_out = Path(__file__).resolve().parent / "output"
    try:
        repo_out.mkdir(parents=True, exist_ok=True)
        fig_copy, _ = plot_news_vs_surprise(
            model=model,
            shock="eps_a",
            leads=leads_to_plot,
            variables=["a", "y", "pi", "r"],
            horizon=16,
            size=1.0,
            figsize=(11, 7.5),
        )
        fig_copy.savefig(repo_out / "dsge_news_shocks.png", dpi=150, bbox_inches="tight")
        plt.close(fig_copy)
    except Exception:
        pass

    print("\nNews shocks analysis completed successfully.")
    return res_news, decomp


if __name__ == "__main__":
    main()
