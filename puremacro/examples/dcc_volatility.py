"""Engle (2002) DCC: time-varying correlations among three return series.

Simulate a 3-variable return panel with time-varying correlation
(spike during a "crisis" sub-sample), then fit DCC(1,1) to recover
the latent correlation path. Demonstrates the multivariate-volatility
extension that the original ``puremacro.garch`` lacked.

Run:
    python -m puremacro.examples.dcc_volatility

Español
-------
DCC de Engle (2002): correlaciones variables en el tiempo entre tres series de rendimientos.

Se simula un panel de rendimientos de tres variables con correlación variable en el tiempo
(con un pico durante una submuestra de "crisis"), y luego se ajusta un DCC(1,1) para recuperar
la trayectoria latente de correlación. Demuestra la extensión de volatilidad multivariada
que el módulo ``puremacro.garch`` original no incluía.

Ejecución:
    python -m puremacro.examples.dcc_volatility
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..garch.dcc import dcc_fit


def _simulate(T: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rho_path = np.where((np.arange(T) > 800) & (np.arange(T) < 1100), 0.75, 0.20)
    n = 3
    out = np.zeros((T, n))
    for t in range(T):
        rho = rho_path[t]
        R = np.array([[1.0, rho, rho * 0.6],
                      [rho, 1.0, rho * 0.5],
                      [rho * 0.6, rho * 0.5, 1.0]])
        L = np.linalg.cholesky(R + 1e-9 * np.eye(n))
        out[t] = L @ rng.standard_normal(n)
    out *= 0.8
    return pd.DataFrame(out, columns=["A", "B", "C"])


def run_dcc() -> dict:
    rets = _simulate()
    fit = dcc_fit(rets)
    rho_hat = fit.R[:, 0, 1]  # A-B correlation path
    return {"data": rets, "fit": fit, "rho_AB": rho_hat}


def main() -> None:
    out = run_dcc()
    fit = out["fit"]
    rho = out["rho_AB"]
    print("DCC(1,1) on synthetic 3-asset returns with regime-switching ρ")
    print(f"  T = {len(out['data'])} obs")
    print(f"  DCC params: a = {fit.a:.3f},  b = {fit.b:.3f}, "
          f"  a+b = {fit.a + fit.b:.3f}")
    print(f"  Sample mean ρ_AB: {rho.mean():.3f}")
    print(f"  Mean ρ_AB in low-correlation regime  (t < 800): "
          f"{rho[:800].mean():.3f}  (true 0.20)")
    print(f"  Mean ρ_AB in high-correlation regime (800-1100): "
          f"{rho[800:1100].mean():.3f}  (true 0.75)")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(rho, color="0.0", lw=0.9, label="DCC ρ̂_AB(t)")
        ax.axhline(0.20, ls="--", color="0.5", lw=0.6)
        ax.axhline(0.75, ls="--", color="0.5", lw=0.6)
        ax.axvspan(800, 1100, color="0.85", alpha=0.5,
                   label="true high-correlation regime")
        ax.set_xlabel("t")
        ax.set_ylabel("Conditional correlation A–B")
        ax.set_title("DCC time-varying correlation recovers regime shift")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "dcc_volatility.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'dcc_volatility.png'}")


if __name__ == "__main__":
    main()
