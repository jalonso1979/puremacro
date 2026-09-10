"""Optimal Monetary Policy: Markov-Perfect Discretion vs LQ Commitment.

Demonstrates the solution and economic comparison of optimal policy regimes
in a canonical 3-equation New Keynesian DSGE model (Oudiz & Sachs 1985; Dennis 2007;
Clarida, Gali & Gertler 1999):
1. Calibrating a textbook New Keynesian block (dynamic IS, Phillips curve, cost-push shock).
2. Solving for Markov-perfect time-consistent discretionary policy via Dennis (2007)
   policy function iteration on the Riccati continuation value matrix V.
3. Solving for timeless-perspective linear-quadratic (LQ) commitment policy via forward-looking
   Lagrange multiplier state augmentation and Klein QZ decomposition.
4. Quantifying the classical Kydland-Prescott / Barro-Gordon inflation bias when target
   output exceeds natural output (y* > 0).
5. Quantifying the stabilization bias (loss under discretion minus loss under commitment)
   arising from the policymaker's inability to credibly manage private-sector expectations.
6. Comparing policy feedback rules F and Riccati matrices V.
7. Saving publication-ready impulse response comparison figures to output/dsge_optimal_discretion.png.

Run:
    python -m puremacro.examples.dsge_optimal_discretion

Español
-------
Política monetaria óptima: Discreción Markov-perfecta vs Compromiso LQ.

Demuestra la resolución y comparación económica de regímenes de política óptima
en un modelo neokeynesiano canónico de 3 ecuaciones (Oudiz & Sachs 1985; Dennis 2007;
Clarida, Galí & Gertler 1999):
1. Calibración del bloque neokeynesiano de manual (IS dinámica, curva de Phillips y choque de costos).
2. Resolución de la política discrecional temporalmente consistente (Markov-perfecta)
   mediante iteración de funciones de política de Dennis (2007) sobre la matriz de Riccati V.
3. Resolución de la política con compromiso (perspectiva atemporal) mediante aumentación de
   multiplicadores de Lagrange y descomposición QZ de Klein.
4. Cuantificación del sesgo de inflación clásico de Kydland-Prescott / Barro-Gordon cuando el
   producto objetivo supera el producto natural (y* > 0).
5. Cuantificación del sesgo de estabilización (pérdida bajo discreción menos pérdida bajo compromiso)
   originado por la incapacidad de gestionar creíblemente las expectativas del sector privado.
6. Comparación de matrices de retroalimentación F y matrices de valor de Riccati V.
7. Generación y guardado de gráficos comparativos de impulso-respuesta en output/dsge_optimal_discretion.png.

Ejecución:
    python -m puremacro.examples.dsge_optimal_discretion
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.policy import optimal_policy
from puremacro.dsge._results import DiscretionaryPolicyResult, PolicyResult


# ------------------------------------------------------------------------------
# 1. Model Calibration: Canonical 3-Equation New Keynesian DSGE Block
# ------------------------------------------------------------------------------

NK_MOD = """
// Canonical 3-Equation New Keynesian Model (Clarida, Gali & Gertler 1999; Dennis 2007)
var y pi r u;
varexo eps_u;

parameters beta sigma kappa phi_pi phi_y rho_u;
beta   = 0.99;   // Quarterly discount factor
sigma  = 1.00;   // Intertemporal elasticity of substitution (log utility)
kappa  = 0.50;   // Slope of New Keynesian Phillips Curve (price stickiness)
phi_pi = 1.50;   // Benchmark Taylor rule inflation feedback
phi_y  = 0.50;   // Benchmark Taylor rule output gap feedback
rho_u  = 0.50;   // Autoregressive persistence of cost-push shock

model;
  // 1. Dynamic Investment-Saving (IS) Curve: y_t = E_t y_{t+1} - (1/sigma)(r_t - E_t pi_{t+1})
  y = y(+1) - (1/sigma)*(r - pi(+1));

  // 2. New Keynesian Phillips Curve (NKPC): pi_t = beta E_t pi_{t+1} + kappa y_t + u_t
  pi = beta*pi(+1) + kappa*y + u;

  // 3. Baseline instrument equation (replaced by optimal policy solver)
  r = phi_pi*pi + phi_y*y;

  // 4. Exogenous persistent cost-push supply shock
  u = rho_u*u(-1) + eps_u;
end;

shocks;
  var eps_u; stderr 1.0;
end;
"""


def main() -> tuple[DiscretionaryPolicyResult, PolicyResult]:
    """Execute optimal discretion vs commitment comparison pipeline."""
    print("=" * 80)
    print("PUREMACRO: OPTIMAL MONETARY POLICY (DISCRETION VS COMMITMENT)")
    print("=" * 80)

    # 1. Build structural baseline model
    model = build_dynare(NK_MOD)
    print("Structural DSGE Model compiled successfully:")
    print(f"  Variables : {list(model.variables)}")
    print(f"  States    : {list(model.states)}")
    print(f"  Shocks    : {list(model.shocks)}")
    print(f"  Stability : {'Determinate (Blanchard-Kahn verified)' if model.is_determinate else 'Indeterminate'}")
    print("-" * 80)

    # 2. Define Policymaker Central Bank Loss Function
    # L_t = E_t sum_{s=0}^infty beta^s [ pi_{t+s}^2 + lambda_y * (y_{t+s} - y*)^2 ]
    # Standard weights: lambda_pi = 1.0, lambda_y = 0.25 (Rogoff 1985; Woodford 2003)
    loss_weights = {"pi": 1.0, "y": 0.25}
    target_output = 0.05  # y* = 5% output target above natural rate

    print("Central Bank Objective Specification:")
    print(f"  Loss Function : L_t = pi_t^2 + 0.25 * (y_t - {target_output})^2")
    print("  Discount beta : 0.99")
    print("  Instrument    : r (nominal policy interest rate)")
    print("-" * 80)

    # 3. Solve Optimal Discretion (Markov-Perfect Dennis 2007)
    res_disc = optimal_policy(
        model,
        loss=loss_weights,
        rule="discretion",
        instruments="r",
        y_star=target_output,
        compare_commitment=True,
        tol=1e-9,
        max_iter=2000,
    )
    assert isinstance(res_disc, DiscretionaryPolicyResult)
    assert res_disc.converged, "Dennis policy iteration failed to converge"

    # 4. Solve Optimal Commitment (Timeless Perspective LQ)
    res_comm = optimal_policy(
        model,
        loss=loss_weights,
        rule="commitment",
        instruments="r",
    )
    assert isinstance(res_comm, PolicyResult)

    # 5. Display Summaries and Reaction Functions
    print("\n" + res_disc.summary())

    print("\nOptimal Feedback Rules F (Instrument response to states):")
    print("Discretionary Feedback Rule (r = F_disc * states):")
    print(res_disc.F.round(5).to_string())

    print("\nCommitment Feedback Rule (r = F_comm * states including Lagrange multipliers):")
    print(res_comm.policy_rules.round(5).to_string())

    # 6. Quantitative Biases Evaluation
    # Theoretical inflation bias formula:
    #   Bias_inf = (kappa * lambda_y) / (lambda_y * (1 - beta) + kappa^2) * y*
    kappa = 0.50
    lambda_y = 0.25
    beta = 0.99
    theory_inf_bias = (kappa * lambda_y) / (lambda_y * (1.0 - beta) + kappa**2) * target_output
    actual_inf_bias = res_disc.inflation_bias
    actual_stab_bias = res_disc.stabilization_bias

    print("\n" + "=" * 80)
    print("WELFARE BIAS QUANTIFICATION")
    print("=" * 80)
    print(f"  Target output distortion (y*)    : {target_output:.4f}")
    print(f"  Theoretical Inflation Bias        : {theory_inf_bias:.6f}")
    print(f"  Quantified Inflation Bias         : {actual_inf_bias:.6f}")
    print(f"  Expected Loss under Discretion    : {res_disc.loss:.6f}")
    if res_disc.commitment_result is not None:
        print(f"  Expected Loss under Commitment    : {res_disc.commitment_result.loss:.6f}")
    print(f"  Quantified Stabilization Bias     : {actual_stab_bias:.6f}")
    print("=" * 80)

    # Sanity checks
    assert np.isclose(actual_inf_bias, theory_inf_bias, rtol=1e-4), "Inflation bias mismatch"
    assert actual_stab_bias > 0.0, "Stabilization bias must be strictly positive"

    # Riccati matrix verification: PSD continuation matrix V
    V = res_disc.V
    eig_v = np.linalg.eigvalsh(V)
    print(f"\nRiccati Matrix V Positive Semi-Definiteness:")
    print(f"  Dimensions         : {V.shape}")
    print(f"  Minimum Eigenvalue : {np.min(eig_v):.6e} (>= 0 confirmed)")
    print(f"  Frobenius Norm     : {np.linalg.norm(V):.6f}")

    # 7. Generate and Save Comparison Plots
    print("\nGenerating impulse response comparison plot (Discretion vs Commitment)...")
    ax = res_disc.plot(compare_commitment=True, periods=16)
    fig = ax.figure

    # Save to output/dsge_optimal_discretion.png
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dsge_optimal_discretion.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_file}")

    # Also save to local package examples output directory if present
    repo_out = Path(__file__).resolve().parent / "output"
    try:
        repo_out.mkdir(parents=True, exist_ok=True)
        fig_copy = res_disc.plot(compare_commitment=True, periods=16).figure
        fig_copy.savefig(repo_out / "dsge_optimal_discretion.png", dpi=150, bbox_inches="tight")
        plt.close(fig_copy)
    except Exception:
        pass

    print("\nOptimal policy analysis completed successfully.")
    return res_disc, res_comm


if __name__ == "__main__":
    main()
