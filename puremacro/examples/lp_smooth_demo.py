"""Barnichon-Brownlees (2019) smoothed local projections on synthetic data.

We simulate a stable AR-augmented DGP where ``y`` responds to ``x`` with
a hump-shaped IRF that peaks around horizon 5. Running unsmoothed Jorda
LP with HAC SE produces a noisy IRF surface; B-spline smoothing with
GCV-chosen ``lambda`` recovers a much cleaner trajectory.

Español
-------
Proyecciones locales suavizadas al estilo Barnichon-Brownlees (2019)
sobre datos sintéticos.

Simulamos un DGP estable aumentado con autorregresiones en el que ``y``
responde a ``x`` con una función de impulso-respuesta en forma de joroba
que alcanza su máximo en torno al horizonte 5. Estimar la proyección local
de Jordà sin suavizar con errores estándar HAC produce una superficie IRF
ruidosa; el suavizado por B-spline con ``lambda`` elegido por validación
cruzada generalizada (GCV) recupera una trayectoria mucho más limpia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..lp.smooth import lp_smooth


def _simulate(T: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    # Hump-shaped lag polynomial on x: peak at lag 5
    weights = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
    y = np.zeros(T)
    eps = 0.5 * rng.standard_normal(T)
    for t in range(len(weights), T):
        y[t] = 0.4 * y[t - 1] + sum(weights[k] * x[t - k] for k in range(len(weights))) + eps[t]
    df = pd.DataFrame({"y": y, "x": x})
    df.index = pd.RangeIndex(T)
    return df


def run_smooth_lp(
    T: int = 400,
    seed: int = 0,
    horizons=range(0, 13),
    n_knots: int = 6,
) -> dict:
    """Estimate smoothed LP on the synthetic DGP. Returns a dict."""
    df = _simulate(T=T, seed=seed)
    irf = lp_smooth(
        df, y="y", x="x", horizons=horizons,
        n_lags=2, n_knots=n_knots, alpha=0.10,
    )
    return {"irf_smooth": irf, "data": df}


def main() -> None:
    out = run_smooth_lp()
    irf = out["irf_smooth"]
    print("Barnichon-Brownlees smoothed LP on synthetic hump-DGP")
    print(f"  GCV-chosen lambda: {float(irf['lambda'].iloc[0]):.4g}")
    print(irf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
