"""Nonlinear DSGE Particle Filtering with Stochastic Volatility (BPF vs Constant Baseline).

Demonstrates pure-Python vectorized Sequential Monte Carlo particle filtering
(Gordon et al. 1993; Fernández-Villaverde & Rubio-Ramírez 2007) for nonlinear DSGE models:
1. Specifying and compiling a nonlinear neoclassical stochastic growth (RBC) DSGE model.
2. Solving the 2nd-order pruned perturbation system (Kim et al. 2008) capturing
   precautionary saving motives, curvature, and variance risk.
3. Augmenting the state-space with autoregressive Stochastic Volatility (SV):
   sigma_t = bar{sigma} * exp(h_t),  h_t = rho_h * h_{t-1} + sigma_eta * eta_t.
4. Evaluating the exact nonlinear likelihood via the vectorized Bootstrap Particle Filter
   (BPF) with systematic resampling across N = 2,000 particles with zero Python loops.
5. Benchmarking against a constant-volatility baseline filter to quantify the likelihood
   gain and precautionary state shifts induced by time-varying macroeconomic volatility.
6. Diagnostic tracking of Effective Sample Size (ESS) trajectories and resampling frequency.
7. Saving publication-ready figures to output/dsge_particle_filter_sv.png.

Run:
    python -m puremacro.examples.dsge_particle_filter_sv

Español
-------
Filtrado de partículas en modelos DSGE no lineales con volatilidad estocástica.

Demuestra el filtrado Monte Carlo secuencial vectorizado puro en Python
(Gordon et al. 1993; Fernández-Villaverde & Rubio-Ramírez 2007) para modelos DSGE:
1. Especificación y compilación de un modelo DSGE neoclásico no lineal de crecimiento (RBC).
2. Solución del sistema de perturbación podado de segundo orden (Kim et al. 2008) que captura
   el motivo de ahorro precautorio, curvatura y riesgo de varianza.
3. Aumento del espacio de estados con volatilidad estocástica autorregresiva (SV):
   sigma_t = bar{sigma} * exp(h_t),  h_t = rho_h * h_{t-1} + sigma_eta * eta_t.
4. Evaluación de la verosimilitud no lineal exacta mediante el filtro de partículas Bootstrap
   (BPF) vectorizado con remuestreo sistemático con N = 2,000 partículas sin bucles en Python.
5. Comparación frente a un modelo de volatilidad constante para cuantificar la ganancia en
   verosimilitud y el desplazamiento precautorio inducido por volatilidad cambiante en el tiempo.
6. Diagnóstico del tamaño muestral efectivo (ESS) y frecuencia de remuestreo.
7. Guardado de figuras de calidad editorial en output/dsge_particle_filter_sv.png.

Ejecución:
    python -m puremacro.examples.dsge_particle_filter_sv
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.dsge.dynare import load_mod
from puremacro.dsge.particle_filter import (
    ParticleFilterResult,
    StochasticVolatilitySpec,
    particle_filter,
)


# ------------------------------------------------------------------------------
# 1. Structural Model: Nonlinear Stochastic Neoclassical Growth (RBC)
# ------------------------------------------------------------------------------

RBC_MOD = """
var c k z;
varexo eps;

parameters beta alpha delta rho sigma_pref sigma_eps;
beta       = 0.99;   // Subjective discount factor
alpha      = 0.33;   // Capital elasticity of output
delta      = 0.025;  // Capital depreciation rate
rho        = 0.95;   // Persistence of technology shock
sigma_pref = 1.00;   // Relative risk aversion coefficient
sigma_eps  = 0.01;   // Base technology shock innovation standard deviation

model;
  // Euler equation for capital accumulation:
  exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);

  // Resource constraint:
  exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));

  // Technology AR(1) process:
  z - rho * z(-1) - sigma_eps * eps;
end;

initval;
  k = 3.8;
  c = 0.8;
  z = 0.0;
end;

steady;
"""


def main() -> tuple[ParticleFilterResult, ParticleFilterResult]:
    """Execute nonlinear particle filter with stochastic volatility pipeline."""
    print("=" * 80)
    print("PUREMACRO: NONLINEAR DSGE PARTICLE FILTER WITH STOCHASTIC VOLATILITY")
    print("=" * 80)

    # 1. Compile and solve 2nd-order pruned perturbation system
    print("1. Compiling DSGE model and solving 2nd-order pruned perturbation...")
    model = load_mod(RBC_MOD)
    sol2 = model.solve(order=2)
    print(f"   Model variables    : {list(model.variables)}")
    print(f"   Pruned solution    : {type(sol2).__name__}")
    print(f"   Steady state k*    : {sol2.steady_state['k']:.4f}")
    print(f"   Steady state c*    : {sol2.steady_state['c']:.4f}")
    print("-" * 80)

    # 2. Simulate synthetic observable path under true time-varying volatility
    T = 32
    print(f"2. Simulating synthetic macroeconomic observables (T={T} periods)...")
    sim = sol2.simulate(periods=T, seed=101).to_frame() + sol2.steady_state
    varobs = ["c", "k"]
    print(f"   Observables tracked: {varobs}")
    print(f"   First 3 observations:\n{sim[varobs].head(3).round(4).to_string()}")
    print("-" * 80)

    # 3. Specify Stochastic Volatility: sigma_t = bar{sigma} * exp(h_t)
    sv_spec = StochasticVolatilitySpec(
        rho=0.85,
        sigma_eta=0.25,
        base_scale=0.01,
        h0=0.0,
    )
    n_particles = 2_000
    print(f"3. Running Bootstrap Particle Filter (N={n_particles:,} particles)...")
    print(f"   a) Filter with Stochastic Volatility (rho_h={sv_spec.rho}, sigma_eta={sv_spec.sigma_eta})...")
    res_sv = particle_filter(
        sol2,
        sim,
        varobs,
        n_particles=n_particles,
        method="bootstrap",
        resampling_method="systematic",
        stochastic_volatility=sv_spec,
        seed=42,
    )
    print(f"      Log-Likelihood (SV) : {res_sv.log_likelihood:.4f}")
    print(f"      Mean ESS            : {res_sv.ess.mean():.1f} / {n_particles}")
    print(f"      Resampling rate     : {res_sv.resampling_frequency:.1%}")

    print("\n   b) Filter with Constant Volatility Baseline...")
    res_const = particle_filter(
        sol2,
        sim,
        varobs,
        n_particles=n_particles,
        method="bootstrap",
        resampling_method="systematic",
        stochastic_volatility=None,
        seed=42,
    )
    print(f"      Log-Likelihood (Const): {res_const.log_likelihood:.4f}")
    print(f"      Mean ESS              : {res_const.ess.mean():.1f} / {n_particles}")
    print(f"      Resampling rate       : {res_const.resampling_frequency:.1%}")
    print("-" * 80)

    # 4. Comparative Metrics & Precautionary State Shift
    print("4. Statistical & Economic Comparison:")
    print(f"   Log-Likelihood Difference (SV - Const): {res_sv.log_likelihood - res_const.log_likelihood:+.4f}")
    mean_k_sv = float(res_sv.filtered_states["k"].mean())
    mean_k_const = float(res_const.filtered_states["k"].mean())
    print(f"   Mean Filtered Capital (SV)           : {mean_k_sv:.4f}")
    print(f"   Mean Filtered Capital (Const)        : {mean_k_const:.4f}")
    print(f"   Precautionary Shift Delta_k          : {mean_k_sv - mean_k_const:+.4f}")
    assert np.isfinite(res_sv.log_likelihood), "SV log-likelihood must be finite"
    assert np.isfinite(res_const.log_likelihood), "Constant log-likelihood must be finite"
    assert res_sv.filtered_volatility is not None, "SV filter must track volatility trajectory"
    assert res_sv.filtered_states.shape == (T, 3), f"Filtered states shape must be ({T}, 3)"

    # Print summary table
    print("\nParticle Filter Result Summary (SV Model):")
    print(res_sv.summary().to_string())

    # 5. Render publication-ready diagnostic plot
    print("\n5. Rendering diagnostic figure...")
    fig = res_sv.plot(variables=["k", "c", "z"], figsize=(10, 8.5))

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dsge_particle_filter_sv.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved figure: {out_file}")

    # Fallback to local examples output directory
    repo_out = Path(__file__).resolve().parent / "output"
    try:
        repo_out.mkdir(parents=True, exist_ok=True)
        fig_copy = res_sv.plot(variables=["k", "c", "z"], figsize=(10, 8.5))
        fig_copy.savefig(repo_out / "dsge_particle_filter_sv.png", dpi=150, bbox_inches="tight")
        plt.close(fig_copy)
    except Exception:
        pass

    print("\nParticle filter simulation completed successfully.")
    return res_sv, res_const


if __name__ == "__main__":
    main()
