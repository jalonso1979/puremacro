"""Uhlig (2005)-style sign-restricted SVAR on a synthetic 3-variable DGP.

We simulate a stable VAR(2) in ``[output, prices, rate]`` with a known
structural impact matrix that places a contractionary monetary-policy
shock as the first structural shock. Then we recover that shock with
Rubio-Ramirez-Waggoner-Zha sign-restriction draws: at h=0 we require the
target shock to push prices and output negative and the policy rate
positive.

Español
-------
SVAR con restricciones de signo al estilo Uhlig (2005) sobre un DGP
sintético de 3 variables.

Simulamos un VAR(2) estable en ``[output, prices, rate]`` con una matriz
de impacto estructural conocida que ubica el choque de política monetaria
contractiva como primer choque estructural. Luego recuperamos ese choque
mediante sorteos con restricciones de signo al estilo Rubio-Ramirez-Waggoner-Zha:
en h=0 requerimos que el choque objetivo empuje los precios y el producto
hacia valores negativos, y la tasa de política hacia valores positivos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..var.identify.sign import sign_restriction_svar


_A1 = np.array([
    [0.6, 0.0, -0.1],
    [0.2, 0.5,  0.0],
    [0.0, 0.1,  0.7],
])
_A2 = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.1],
])
# Structural impact: column 0 = monetary-policy shock with
# (output negative, prices negative, rate positive).
_B0 = np.array([
    [-0.5,  0.6,  0.0],
    [-0.4,  0.0,  0.3],
    [ 0.4,  0.0,  0.5],
])


def _simulate(T: int = 400, seed: int = 0, burn: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = _B0.shape[0]
    eps = rng.standard_normal((T + burn, n))
    Y = np.zeros((T + burn, n))
    for t in range(2, T + burn):
        Y[t] = _A1 @ Y[t - 1] + _A2 @ Y[t - 2] + _B0 @ eps[t]
    return Y[burn:]


def run_uhlig(
    T: int = 400,
    seed: int = 0,
    horizon: int = 16,
    n_draws: int = 500,
) -> dict:
    """Run sign-restricted SVAR on the synthetic DGP. Returns IRF arrays."""
    Y = _simulate(T=T, seed=seed)
    # Sign restrictions for the target (column 0) shock at h=0:
    #   output (var 0) -> negative,  prices (var 1) -> negative,
    #   rate   (var 2) -> positive.
    restrictions = {0: [-1, -1, +1]}
    res = sign_restriction_svar(
        Y, p=2, horizon=horizon, restrictions=restrictions,
        n_draws=n_draws, ci=0.9, seed=seed,
    )
    return {
        "median_irf": res.irf_median,
        "irf_lower": res.irf_lower,
        "irf_upper": res.irf_upper,
        "Y": pd.DataFrame(Y, columns=["output", "prices", "rate"]),
        "horizon": horizon,
    }


def main() -> None:
    res = run_uhlig()
    median = res["median_irf"]
    print("Uhlig (2005) sign-restricted SVAR on synthetic 3-variable DGP")
    print(f"  IRF shape (H+1, n, n): {median.shape}")
    print("  Median impact (h=0) of target shock on each variable:")
    for i, name in enumerate(["output", "prices", "rate"]):
        print(f"    {name:7s}: {median[0, i, 0]:+.3f}")


if __name__ == "__main__":
    main()
