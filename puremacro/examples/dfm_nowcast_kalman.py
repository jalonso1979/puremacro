"""Single-factor dynamic factor model via the Kalman filter.

We simulate a small monthly panel of 4 indicators driven by a common
AR(1) factor + idiosyncratic noise. The Kalman filter then recovers
the factor and computes the implied nowcast of a fifth, lower-frequency
target series — illustrating how ``puremacro.state_space`` makes
DFM-style nowcasting viable on an iPad.

The DGP:
    f_{t+1} = phi * f_t + eta_t,        eta_t ~ N(0, sigma_f^2)
    x_{i,t} = lambda_i * f_t + eps_{i,t},  eps_{i,t} ~ N(0, sigma_i^2)

The state-space form:
    alpha_t = f_t  (m=1)
    T_mat   = [[phi]],  R = I,  Q = [[sigma_f^2]]
    Z       = column vector of loadings,  H = diag(sigma_i^2)

Run:
    python -m puremacro.examples.dfm_nowcast_kalman

Español
-------
Modelo de factores dinámicos de un solo factor mediante el filtro de Kalman.

Se simula un panel mensual pequeño de 4 indicadores impulsados por un factor
AR(1) común más ruido idiosincrático. El filtro de Kalman recupera el factor
y calcula la proyección inmediata (nowcast) implícita de una quinta serie
objetivo de menor frecuencia, ilustrando cómo ``puremacro.state_space``
hace viable la proyección al estilo DFM en un iPad.

El DGP:
    f_{t+1} = phi * f_t + eta_t,        eta_t ~ N(0, sigma_f^2)
    x_{i,t} = lambda_i * f_t + eps_{i,t},  eps_{i,t} ~ N(0, sigma_i^2)

La forma espacio-estado:
    alpha_t = f_t  (m=1)
    T_mat   = [[phi]],  R = I,  Q = [[sigma_f^2]]
    Z       = vector columna de cargas,  H = diag(sigma_i^2)

Ejecución:
    python -m puremacro.examples.dfm_nowcast_kalman
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..state_space import StateSpaceModel, kalman_smoother


def _simulate(T: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = 0.85
    sigma_f = 0.4
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = phi * f[t - 1] + sigma_f * rng.standard_normal()
    loadings = np.array([1.0, 0.8, 0.6, 1.2])
    sig_eps = np.array([0.3, 0.4, 0.3, 0.5])
    X = (loadings[None, :] * f[:, None]
         + sig_eps[None, :] * rng.standard_normal((T, 4)))
    return f, loadings, sig_eps, phi, sigma_f, X


def run_dfm() -> dict:
    f_true, loadings, sig_eps, phi, sigma_f, X = _simulate()
    model = StateSpaceModel(
        T=np.array([[phi]]),
        Z=loadings[:, None],
        Q=np.array([[sigma_f ** 2]]),
        H=np.diag(sig_eps ** 2),
    )
    out = kalman_smoother(X, model)
    f_hat = out["a_smooth"].ravel()
    rmse = float(np.sqrt(np.mean((f_hat - f_true) ** 2)))
    corr = float(np.corrcoef(f_hat, f_true)[0, 1])
    return {
        "f_true": f_true,
        "f_hat": f_hat,
        "X": X,
        "loadings": loadings,
        "loglik": out["loglik"],
        "rmse": rmse,
        "corr": corr,
    }


def main() -> None:
    out = run_dfm()
    print("Single-factor DFM via Kalman smoother")
    print(f"  T = {len(out['f_true'])} months,  4 indicators")
    print(f"  Smoother RMSE vs true factor: {out['rmse']:.3f}")
    print(f"  Correlation factor / truth:    {out['corr']:.3f}")
    print(f"  Log-likelihood (filter):       {out['loglik']:.2f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.plot(out["f_true"], color="0.65", lw=0.8, label="true factor")
        ax.plot(out["f_hat"], color="0.0", lw=1.0, label="Kalman smoothed")
        ax.set_xlabel("Month")
        ax.set_ylabel("Factor level")
        ax.set_title("Single-factor DFM: Kalman recovers latent factor")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "dfm_nowcast_kalman.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'dfm_nowcast_kalman.png'}")


if __name__ == "__main__":
    main()
