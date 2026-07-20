"""Hamilton (1989)-style two-regime MS-VAR for a recession dating exercise.

We simulate a 2-variable system that switches between a "high-growth,
low-volatility" regime and a "recession" regime with negative mean and
elevated volatility. The MS-VAR fit recovers smoothed regime
probabilities; with reasonable signal-to-noise the smoother nails the
true regime path > 95% of the time.

The output is the standard Hamilton-style picture: the smoothed
P(recession | data) line tracks the true regime indicator.

Run:
    python -m puremacro.examples.ms_var_business_cycle

Español
-------
VAR con cambio de régimen (MS-VAR) de dos regímenes al estilo Hamilton
(1989) para un ejercicio de datación de recesiones.

Se simula un sistema de 2 variables que alterna entre un régimen de
"alto crecimiento y baja volatilidad" y un régimen de "recesión" con
media negativa y volatilidad elevada. El ajuste del MS-VAR recupera las
probabilidades de régimen suavizadas; con una relación señal-ruido
razonable, el suavizador identifica correctamente la trayectoria de
régimen verdadera en más del 95% de las observaciones.

El resultado es el gráfico estándar al estilo Hamilton: la línea
P(recesión | datos) suavizada sigue el indicador de régimen verdadero.

Ejecución:
    python -m puremacro.examples.ms_var_business_cycle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.regime import ms_var_fit


def _simulate(T: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    P_true = np.array([[0.97, 0.03], [0.10, 0.90]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice([0, 1], p=P_true[s[t - 1]])
    mu = np.array([[0.6, 0.3], [-1.4, -0.5]])
    A = np.array([[0.4, 0.05], [0.0, 0.5]])
    Sigma = [np.diag([0.4, 0.4]), np.diag([1.6, 0.9])]
    Y = np.zeros((T, 2))
    for t in range(1, T):
        e = np.linalg.cholesky(Sigma[s[t]]) @ rng.standard_normal(2)
        Y[t] = mu[s[t]] + A @ Y[t - 1] + e
    return Y, s


def run_demo() -> dict:
    Y, s_true = _simulate()
    fit = ms_var_fit(Y, K=2, p=1, n_iter=80)
    sm = fit.smoothed_probs  # (T-1, 2)
    s_true_eff = s_true[1:]
    # Resolve label switching: pick the regime whose mu is smaller as "recession".
    rec_label = int(np.argmin(fit.mu[:, 0]))
    p_recession = sm[:, rec_label]
    classified = (p_recession > 0.5).astype(int)
    accuracy = float(np.mean(classified == s_true_eff))
    return {
        "Y": Y, "s_true": s_true, "s_true_eff": s_true_eff,
        "fit": fit, "p_recession": p_recession,
        "rec_label": rec_label, "accuracy": accuracy,
    }


def main() -> None:
    out = run_demo()
    fit = out["fit"]
    print("Hamilton-style MS-VAR with two regimes (mean & variance switching)")
    print(f"  T = {len(out['Y'])} obs,  EM iters = {fit.n_iter}, "
          f"converged = {fit.converged}")
    print(f"  Log-likelihood: {fit.loglik:.2f}")
    print()
    print("  Estimated regime intercepts (rec label = "
          f"{out['rec_label']}):")
    for k in range(2):
        tag = "RECESSION" if k == out["rec_label"] else "Expansion"
        print(f"    Regime {k} ({tag}): mu = {np.round(fit.mu[k], 3).tolist()}")
    print()
    print(f"  Estimated transition matrix:\n{np.round(fit.P, 3)}")
    print()
    print(f"  Classification accuracy (smoothed prob > 0.5): "
          f"{out['accuracy']*100:.1f}%")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.0), sharex=True)
        axes[0].plot(out["Y"][:, 0], color="0.0", lw=0.7)
        axes[0].set_ylabel("y_0 (growth)")
        axes[0].axhline(0, color="0.5", lw=0.4)
        axes[1].plot(out["s_true_eff"], color="0.5", lw=0.5,
                     drawstyle="steps-post", label="true regime")
        axes[1].plot(out["p_recession"], color="0.0", lw=0.9,
                     label="P(recession | data)")
        axes[1].set_ylabel("Probability")
        axes[1].set_xlabel("t")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(frameon=False, loc="best")
        fig.suptitle(f"MS-VAR recession dating (acc {out['accuracy']*100:.1f}%)")
        fig.tight_layout()
        fig.savefig(out_dir / "ms_var_business_cycle.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'ms_var_business_cycle.png'}")


if __name__ == "__main__":
    main()
