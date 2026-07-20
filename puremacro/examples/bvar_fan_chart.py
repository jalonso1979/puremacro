"""Density forecast (fan chart) from a Minnesota-prior BVAR via Gibbs.

Posterior-mean BVAR gives only a point forecast. The full Normal-
Inverse-Wishart posterior — sampled with ``minnesota_gibbs`` — yields
a *density* forecast: at each horizon we have a posterior over both
the VAR coefficients and Sigma, so simulating one path per draw gives
a full predictive distribution.

The fan chart below shows posterior 10/30/50/70/90 quantiles at
horizons 1..H for a small 3-variable VAR.

Run:
    python -m puremacro.examples.bvar_fan_chart

Español
-------
Proyección de densidad (gráfico de abanico) a partir de un BVAR con
prior de Minnesota estimado mediante Gibbs.

El BVAR de media posterior solo ofrece una proyección puntual. El
posterior Normal-Wishart-Inversa completo — muestreado con
``minnesota_gibbs`` — produce una proyección de *densidad*: a cada
horizonte se cuenta con un posterior sobre los coeficientes del VAR y
Sigma, de modo que simular una trayectoria por cada extracción da lugar
a una distribución predictiva completa.

El gráfico de abanico muestra los cuantiles posteriores 10/30/50/70/90
en los horizontes 1..H para un VAR pequeño de 3 variables.

Ejecución:
    python -m puremacro.examples.bvar_fan_chart
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..var.bvar import minnesota_gibbs


def _simulate(T: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.7, 0.10, 0.0],
        [0.05, 0.6, 0.10],
        [0.0, 0.05, 0.55],
    ])
    Sigma = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return Y


def run_fan_chart(H: int = 12, n_draws: int = 500) -> dict:
    Y = _simulate()
    g = minnesota_gibbs(Y, p=1, n_draws=n_draws, burn=200,
                        rng=np.random.default_rng(0))
    n = Y.shape[1]
    paths = np.zeros((n_draws, H + 1, n))
    for d in range(n_draws):
        A_d = g["A_draws"][d, 0]
        c_d = g["intercept_draws"][d]
        S_d = g["Sigma_draws"][d]
        L_d = np.linalg.cholesky(S_d + 1e-12 * np.eye(n))
        y = Y[-1].copy()
        paths[d, 0] = y
        rng = np.random.default_rng(d + 1)
        for h in range(1, H + 1):
            y = c_d + A_d @ y + L_d @ rng.standard_normal(n)
            paths[d, h] = y
    quants = np.percentile(paths, [10, 30, 50, 70, 90], axis=0)  # (5, H+1, n)
    return {"data": Y, "paths": paths, "quants": quants, "H": H}


def main() -> None:
    out = run_fan_chart()
    q = out["quants"]
    H = out["H"]
    print("BVAR (Minnesota Gibbs) fan-chart density forecast")
    print(f"  T training = {len(out['data'])},  horizons = 1..{H},  draws = {len(out['paths'])}")
    print()
    print("Median forecast and 80% predictive band, variable 0:")
    for h in range(0, H + 1):
        med = q[2, h, 0]
        lo, hi = q[0, h, 0], q[4, h, 0]
        print(f"  h={h:>2d}:  median={med:+.3f}   80%=[{lo:+.3f}, {hi:+.3f}]")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
        h_arr = np.arange(H + 1)
        last_val = out["data"][-1]
        for i, ax in enumerate(axes):
            ax.fill_between(h_arr, q[0, :, i], q[4, :, i],
                            color="0.85", label="80%")
            ax.fill_between(h_arr, q[1, :, i], q[3, :, i],
                            color="0.65", label="40%")
            ax.plot(h_arr, q[2, :, i], color="0.0", lw=1.0,
                    label="median")
            ax.axhline(last_val[i], color="0.4", lw=0.4)
            ax.set_xlabel("Horizon h")
            ax.set_title(f"y_{i}")
            if i == 0:
                ax.set_ylabel("Predictive level")
                ax.legend(frameon=False, loc="best")
        fig.suptitle("BVAR Gibbs fan chart (10/30/50/70/90 quantiles)")
        fig.tight_layout()
        fig.savefig(out_dir / "bvar_fan_chart.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'bvar_fan_chart.png'}")


if __name__ == "__main__":
    main()
