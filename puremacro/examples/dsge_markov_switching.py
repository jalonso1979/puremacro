"""Markov-Switching DSGE Perturbation: Monetary Policy Regimes & Analytical GIRFs.

Demonstrates Markov-Switching DSGE (MS-DSGE) perturbation modeling following
Foerster, Rubio-Ramírez, Waggoner & Zha (2016) and Farmer, Waggoner & Zha (2011):
1. Formulating a 3-equation New Keynesian DSGE model with regime-switching monetary policy:
   - Dynamic IS Curve:  y_t = E_t[y_{t+1}] - (1/sigma) * (i_t - E_t[pi_{t+1}]) + eps_y,t
   - NK Phillips Curve: pi_t = beta * E_t[pi_{t+1}] + kappa * y_t + eps_pi,t
   - Switching Taylor:  i_t = rho_i * i_{t-1} + (1 - rho_i) * (phi_pi(s_t) * pi_t + phi_x * y_t) + eps_i,t
2. Specifying policy regimes:
   - Regime 1 (Hawkish): phi_pi = 1.8 > 1 (Active monetary policy, determinate in isolation)
   - Regime 2 (Dovish) : phi_pi = 0.8 < 1 (Passive monetary policy, indeterminate in isolation)
3. Transition probability matrix P with persistence p_11 = 0.90, p_22 = 0.80.
4. Solving the coupled quadratic matrix equations for Minimal State Variable (MSV) decision rules
   y_t = T(s_t) * y_{t-1} + R(s_t) * eps_t via analytical block Newton-Raphson.
5. Verifying Mean-Square Stability (MSS) rho(M_2) < 1.0 and first-moment stability rho(M_1) < 1.0,
   confirming global determinacy despite the local indeterminacy of the Dovish regime.
6. Computing ergodic stationary distributions pi_infty and unconditional Lyapunov covariances.
7. Evaluating closed-form analytical Generalized Impulse Response Functions (GIRF)
   via the operator (1_S' (x) I_n) * M_1^h * z_0 in machine precision without Monte Carlo noise.
8. Saving publication-ready figures to output/dsge_markov_switching.png.

Run:
    python -m puremacro.examples.dsge_markov_switching

Español
-------
Modelos DSGE con cambio de régimen de Markov y funciones de impulso-respuesta generalizadas.

Demuestra la resolución mediante perturbación de modelos DSGE con cambio de régimen
(Foerster, Rubio-Ramírez, Waggoner & Zha 2016; Farmer, Waggoner & Zha 2011):
1. Formulación de un modelo neokeynesiano de 3 ecuaciones con regla de Taylor sujeta a cambio de régimen:
   - Curva IS dinámica:        y_t = E_t[y_{t+1}] - (1/sigma) * (i_t - E_t[pi_{t+1}]) + eps_y,t
   - Curva de Phillips NK:     pi_t = beta * E_t[pi_{t+1}] + kappa * y_t + eps_pi,t
   - Regla de Taylor cambiante: i_t = rho_i * i_{t-1} + (1 - rho_i) * (phi_pi(s_t) * pi_t + phi_x * y_t) + eps_i,t
2. Definición de regímenes de política monetaria:
   - Régimen 1 (Hawkish): phi_pi = 1.8 > 1 (Política activa, determinable en aislamiento)
   - Régimen 2 (Dovish) : phi_pi = 0.8 < 1 (Política pasiva, indeterminable en aislamiento)
3. Matriz de transición de Markov P con persistencias p_11 = 0.90, p_22 = 0.80.
4. Solución de las ecuaciones cuadráticas matriciales acopladas para reglas de decisión MSV
   y_t = T(s_t) * y_{t-1} + R(s_t) * eps_t mediante Newton-Raphson por bloques analítico.
5. Verificación de estabilidad en media cuadrática (MSS) rho(M_2) < 1.0 y primer momento rho(M_1) < 1.0,
   demostrando estabilidad global a pesar de la indeterminación local del régimen Dovish.
6. Cálculo de la distribución estacionaria ergódica pi_infty y covarianza de Lyapunov incondicional.
7. Evaluación analítica exacta de funciones de impulso-respuesta generalizadas (GIRF)
   mediante el operador (1_S' (x) I_n) * M_1^h * z_0 a precisión de máquina.
8. Guardado de figuras editoriales en output/dsge_markov_switching.png.

Ejecución:
    python -m puremacro.examples.dsge_markov_switching
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.markov_switching import (
    MSDSGEResult,
    solve_ms_dsge,
)


def main() -> tuple[MSDSGEResult, pd.DataFrame]:
    """Execute Markov-switching DSGE equilibrium solution and GIRF analysis."""
    print("=" * 80)
    print("PUREMACRO: MARKOV-SWITCHING DSGE (FOERSTER ET AL. 2016 / FARMER ET AL. 2011)")
    print("=" * 80)

    # 1. Structural Parameters
    beta = 0.99      # Quarterly discount factor
    sigma = 1.00     # Intertemporal elasticity of substitution
    kappa = 0.10     # Slope of New Keynesian Phillips curve
    rho_i = 0.80     # Interest rate smoothing parameter
    phi_x = 0.10     # Taylor rule response to output gap

    # Regime-specific monetary policy parameters
    phi_pi_list = [1.80, 0.80]   # [Hawkish (active), Dovish (passive)]
    regime_names = ["Hawkish", "Dovish"]
    variable_names = ["output_gap", "inflation", "interest_rate"]
    shock_names = ["demand", "cost_push", "monetary_policy"]

    # 2. Markov Transition Matrix P: p_ij = Pr(s_{t+1} = j | s_t = i)
    P = np.array([
        [0.90, 0.10],   # Hawkish persistence = 90% (expected duration 10 quarters)
        [0.20, 0.80],   # Dovish persistence  = 80% (expected duration 5 quarters)
    ])

    print("1. Specifying Structural System and Policy Regimes:")
    print(f"   Variables : {variable_names}")
    print(f"   Shocks    : {shock_names}")
    print(f"   Regimes   : {regime_names}")
    print(f"   Inflation response phi_pi: Hawkish={phi_pi_list[0]}, Dovish={phi_pi_list[1]}")
    print(f"   Transition Matrix P:\n{P}")
    print("-" * 80)

    # 3. Assemble Structural Matrices A(s), B(s), C(s), D(s)
    # A(s) E_t[y_{t+1}] + B(s) y_t + C(s) y_{t-1} + D(s) eps_t = 0
    S = 2
    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(S):
        phi_pi = phi_pi_list[s]
        # Equation 1: Dynamic IS
        # y_t - E_t[y_{t+1}] + (1/sigma)*(i_t - E_t[pi_{t+1}]) - eps_y,t = 0
        # Equation 2: NKPC
        # pi_t - beta*E_t[pi_{t+1}] - kappa*y_t - eps_pi,t = 0
        # Equation 3: Taylor Rule
        # i_t - rho_i*i_{t-1} - (1-rho_i)*(phi_pi*pi_t + phi_x*y_t) - eps_i,t = 0

        As = np.array([
            [1.0, 1.0 / sigma, 0.0],
            [0.0, beta,        0.0],
            [0.0, 0.0,         0.0],
        ])
        Bs = np.array([
            [-1.0, 0.0, -1.0 / sigma],
            [kappa, -1.0, 0.0],
            [(1.0 - rho_i) * phi_x, (1.0 - rho_i) * phi_pi, -1.0],
        ])
        Cs = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, rho_i],
        ])
        Ds = np.eye(3)

        A_list.append(As)
        B_list.append(Bs)
        C_list.append(Cs)
        D_list.append(Ds)

    # 4. Solve MS-DSGE via Block Newton-Raphson Solver
    print("2. Solving Coupled Quadratic Matrix Equations via Block Newton-Raphson...")
    res = solve_ms_dsge(
        A_list,
        B_list,
        C_list,
        D_list,
        P,
        regime_names=regime_names,
        variable_names=variable_names,
        shock_names=shock_names,
        method="newton",
        tol=1e-10,
    )

    print(f"   Convergence Status : {'CONVERGED' if res.converged else 'FAILED'}")
    print(f"   Iterations required: {res.iterations}")
    print(f"   Residual norm      : {res.diff:.2e}")
    print(f"   Mean Stability     : {'STABLE' if res.spectral_radius_mean < 1.0 else 'UNSTABLE'} (rho(M1)={res.spectral_radius_mean:.4f})")
    print(f"   Mean-Square Stable : {res.mean_square_stable} (rho(M2)={res.spectral_radius_mss:.4f})")
    assert res.converged, "MS-DSGE solver must converge"
    assert res.mean_square_stable, "Equilibrium must satisfy Mean-Square Stability"
    print("-" * 80)

    # 5. Ergodic Regime Distribution & Moments
    print("3. Ergodic Distribution & Long-Run Moments:")
    pi_hawkish = float(res.ergodic_distribution["Hawkish"])
    pi_dovish = float(res.ergodic_distribution["Dovish"])
    print(f"   Ergodic Probability [Hawkish]: {pi_hawkish:.4f} (theoretical 2/3 = {2/3:.4f})")
    print(f"   Ergodic Probability [Dovish] : {pi_dovish:.4f} (theoretical 1/3 = {1/3:.4f})")
    print("\n   Unconditional Covariance Matrix (Discrete Lyapunov Solution):")
    print(res.ergodic_cov.round(4).to_string())
    print("-" * 80)

    # 6. Generalized Impulse Response Functions (GIRF)
    horizon = 16
    print(f"4. Computing Closed-Form Analytical GIRFs (horizon={horizon} quarters)...")
    print("   Evaluating operator (1_S' (x) I_n) * M_1^h * z_0 to contractionary monetary shock...")
    girf_hawkish = res.girf("Hawkish", shock="monetary_policy", horizon=horizon)
    girf_dovish = res.girf("Dovish", shock="monetary_policy", horizon=horizon)

    # Verify macroeconomic logic:
    # Contractionary policy raises interest rate and depresses output gap & inflation on impact
    r_impact_h = float(girf_hawkish.loc[0, "interest_rate"])
    y_impact_h = float(girf_hawkish.loc[0, "output_gap"])
    pi_impact_h = float(girf_hawkish.loc[0, "inflation"])
    print(f"   Impact Responses (Hawkish Regime Initialized):")
    print(f"     Interest Rate (i_0) : {r_impact_h:+.4f} (> 0 interest rate hike)")
    print(f"     Output Gap (y_0)    : {y_impact_h:+.4f} (< 0 demand contraction)")
    print(f"     Inflation (pi_0)    : {pi_impact_h:+.4f} (< 0 price deceleration)")
    assert r_impact_h > 0, "Interest rate must rise on contractionary monetary policy shock"
    assert y_impact_h < 0, "Output gap must contract on policy tightening"
    assert pi_impact_h < 0, "Inflation must decrease on policy tightening"

    # Print summary text
    print("\n" + res.summary())

    # 7. Render publication-grade figure
    print("\n5. Rendering multi-panel impulse response and GIRF figure...")
    fig, axes = res.plot(regime="Hawkish", girf=True, horizon=horizon, shock="monetary_policy")

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dsge_markov_switching.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved figure: {out_file}")

    # Fallback to local examples output directory
    repo_out = Path(__file__).resolve().parent / "output"
    try:
        repo_out.mkdir(parents=True, exist_ok=True)
        fig_copy, _ = res.plot(regime="Hawkish", girf=True, horizon=horizon, shock="monetary_policy")
        fig_copy.savefig(repo_out / "dsge_markov_switching.png", dpi=150, bbox_inches="tight")
        plt.close(fig_copy)
    except Exception:
        pass

    print("\nMarkov-switching DSGE analysis completed successfully.")
    return res, girf_hawkish


if __name__ == "__main__":
    main()
