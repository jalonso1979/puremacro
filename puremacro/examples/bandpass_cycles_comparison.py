"""Business cycle filtering comparison: Baxter-King, Christiano-Fitzgerald, Beveridge-Nelson & Hamilton.

Extracts business-cycle fluctuations (6 to 32 quarters) from a non-stationary
macroeconomic series, contrasting symmetric lead-lag truncation (Baxter-King)
with full-sample asymmetric random-walk projection (Christiano-Fitzgerald) and
companion-matrix stochastic trend forecasting (Beveridge-Nelson).

Run:
    python -m puremacro.examples.bandpass_cycles_comparison

Español
-------
Comparación de filtros de ciclos económicos: Baxter-King, Christiano-Fitzgerald,
Beveridge-Nelson y Hamilton.

Extrae las fluctuaciones del ciclo económico (6 a 32 trimestres) de una serie
macroeconómica no estacionaria, contrastando el filtrado simétrico (Baxter-King),
la proyección asimétrica de muestra completa (Christiano-Fitzgerald) y la
descomposición permanente-transitoria por matriz compañera (Beveridge-Nelson).

Ejecución:
    python -m puremacro.examples.bandpass_cycles_comparison
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cycles import (
    baxter_king_filter,
    beveridge_nelson_filter,
    christiano_fitzgerald_filter,
    hamilton_filter,
    hp_filter,
)


def run_cycle_comparison(T: int = 200, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # 1. Stochastic Trend (Random walk with drift)
    drift = 0.006
    trend_shocks = rng.normal(0, 0.008, size=T)
    tau_true = 100.0 + np.cumsum(drift + trend_shocks)

    # 2. True Business Cycle (Period = 32 quarters, damped AR(2))
    r, theta = 0.92, 2 * np.pi / 32.0
    phi1, phi2 = 2 * r * np.cos(theta), -r**2
    c_true = np.zeros(T)
    cycle_shocks = rng.normal(0, 0.012, size=T)
    for t in range(2, T):
        c_true[t] = phi1 * c_true[t - 1] + phi2 * c_true[t - 2] + cycle_shocks[t]

    noise = rng.normal(0, 0.003, size=T)
    y = tau_true + c_true + noise

    # Apply filters
    bk_cycle, bk_trend = baxter_king_filter(y, low=6, high=32, K=12)
    cf_cycle, cf_trend = christiano_fitzgerald_filter(y, low=6, high=32, drift=True)
    bn_cycle, bn_trend = beveridge_nelson_filter(y, p=4)
    ham_cycle, ham_trend = hamilton_filter(y, h=8, p=4)
    hp_cycle, hp_trend = hp_filter(y, lamb=1600.0)

    return {
        "y": y,
        "tau_true": tau_true,
        "c_true": c_true,
        "bk_cycle": bk_cycle,
        "cf_cycle": cf_cycle,
        "bn_cycle": bn_cycle,
        "bn_trend": bn_trend,
        "ham_cycle": ham_cycle,
        "hp_cycle": hp_cycle,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Business Cycle Filtering Comparison...")
    res = run_cycle_comparison()

    T = len(res["y"])
    valid_idx = np.arange(12, T - 12)
    c_true = res["c_true"]

    corr_bk = np.corrcoef(c_true[valid_idx], res["bk_cycle"][valid_idx])[0, 1]
    corr_cf = np.corrcoef(c_true[valid_idx], res["cf_cycle"][valid_idx])[0, 1]
    corr_hp = np.corrcoef(c_true[valid_idx], res["hp_cycle"][valid_idx])[0, 1]

    print(f"  Correlation with Planted Cycle:")
    print(f"    Baxter-King (K=12):          {corr_bk:.4f}")
    print(f"    Christiano-Fitzgerald:       {corr_cf:.4f}")
    print(f"    Hodrick-Prescott:            {corr_hp:.4f}")

    time_axis = np.arange(T) / 4.0

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)

    axes[0].plot(time_axis, res["y"], color="0.10", lw=1.6, label=r"Observed Aggregate $y_t$")
    axes[0].plot(time_axis, res["tau_true"], color="0.50", ls="--", lw=1.4, label=r"True Trend $\tau_t$")
    axes[0].plot(time_axis, res["bn_trend"], color="0.25", ls=":", lw=1.5, label="Beveridge-Nelson Trend")
    axes[0].set_title("(a) Aggregate Series and Permanent Trends", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Level")
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)

    axes[1].plot(time_axis, res["c_true"], color="0.75", lw=2.5, label="True Planted Cycle")
    axes[1].plot(time_axis, res["bk_cycle"], color="0.00", lw=1.6, label=f"Baxter-King [r={corr_bk:.2f}]")
    axes[1].plot(time_axis, res["cf_cycle"], color="0.40", ls="--", lw=1.5, label=f"Christiano-Fitzgerald [r={corr_cf:.2f}]")
    axes[1].axhline(0, color="0.70", ls=":", lw=0.8)
    axes[1].set_title("(b) Bandpass Approximations (6-32 Quarters)", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Cycle (% dev)")
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)

    axes[2].plot(time_axis, res["c_true"], color="0.75", lw=2.5, label="True Planted Cycle")
    axes[2].plot(time_axis, res["hp_cycle"], color="0.20", ls="-.", lw=1.4, label=f"Hodrick-Prescott [r={corr_hp:.2f}]")
    axes[2].plot(time_axis, res["ham_cycle"], color="0.50", ls=":", lw=1.5, label="Hamilton OLS Projection")
    axes[2].plot(time_axis, res["bn_cycle"], color="0.00", lw=1.2, label="Beveridge-Nelson Transitory")
    axes[2].axhline(0, color="0.70", ls=":", lw=0.8)
    axes[2].set_title("(c) Econometric Alternatives: HP, Hamilton, and Beveridge-Nelson", loc="left", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Time (Years)")
    axes[2].set_ylabel("Cycle (% dev)")
    axes[2].legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig_path = out_dir / "bandpass_cycles_comparison.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
