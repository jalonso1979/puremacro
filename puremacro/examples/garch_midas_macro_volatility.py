"""GARCH-MIDAS mixed-frequency volatility model (Engle, Ghysels & Sohn 2013).

Decomposes high-frequency daily asset returns into a short-run daily GARCH(1,1)
component and a low-frequency secular macro trend driven by monthly macroeconomic
uncertainty / business cycle regimes via a Beta lag polynomial.

Run:
    python -m puremacro.examples.garch_midas_macro_volatility

Español
-------
Modelo de volatilidad de frecuencia mixta GARCH-MIDAS (Engle, Ghysels & Sohn 2013).

Descompone los rendimientos diarios de activos en un componente GARCH(1,1) diario
de corto plazo y una tendencia secular macroeconómica de baja frecuencia impulsada
por incertidumbre mensual a través de un polinomio de rezagos Beta.

Ejecución:
    python -m puremacro.examples.garch_midas_macro_volatility
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..midas import garch_midas


def run_garch_midas_demo(N_months: int = 36, K: int = 22, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    T_days = N_months * K

    # Monthly macro uncertainty index
    x_lf = np.zeros(N_months)
    x_lf[14:24] = 2.4 + rng.normal(0, 0.25, size=10)
    x_lf[:14] = rng.normal(0, 0.2, size=14)
    x_lf[24:] = rng.normal(0, 0.2, size=N_months - 24)

    # True parameters
    m_true = 0.0
    theta_true = 0.50
    alpha_true = 0.08
    beta_true = 0.86
    L_lags = 6

    weights_true = np.array([(1.0 - (l + 1) / (L_lags + 1)) ** 2.0 for l in range(L_lags)])
    weights_true /= np.sum(weights_true)

    log_tau = np.full(N_months, m_true)
    for t in range(L_lags, N_months):
        log_tau[t] = m_true + theta_true * np.dot(weights_true, x_lf[t - L_lags:t][::-1])
    tau_monthly = np.exp(log_tau)
    tau_daily = np.repeat(tau_monthly, K)

    g_daily = np.ones(T_days)
    returns_daily = np.zeros(T_days)

    for t in range(1, T_days):
        g_daily[t] = (1.0 - alpha_true - beta_true) + alpha_true * (returns_daily[t - 1]**2 / tau_daily[t - 1]) + beta_true * g_daily[t - 1]
        returns_daily[t] = np.sqrt(tau_daily[t] * g_daily[t]) * rng.standard_normal()

    # Fit GARCH-MIDAS
    res = garch_midas(returns_daily, x_lf=x_lf, K=K, L=L_lags)

    return {
        "returns": returns_daily,
        "x_lf": x_lf,
        "tau_true": tau_daily,
        "res": res,
        "K": K,
        "N_months": N_months,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting GARCH-MIDAS Volatility Model...")
    out = run_garch_midas_demo()
    res = out["res"]
    print(res.summary())

    T_days = len(out["returns"])
    months_axis = np.arange(T_days) / out["K"]

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True)

    # Panel 1: Daily returns
    axes[0].plot(months_axis, out["returns"], color="0.25", lw=0.8, alpha=0.85, label="Daily Asset Returns $r_{i,t}$")
    axes[0].axvspan(14, 24, color="0.85", alpha=0.5, label="Macro Crisis Window")
    axes[0].set_title("(a) High-Frequency Asset Returns", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Daily Return")
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)

    # Panel 2: Secular trend vs monthly macro driver
    ax2_twin = axes[1].twinx()
    p1 = axes[1].plot(months_axis, np.sqrt(res.tau), color="0.00", lw=2.0, label=r"GARCH-MIDAS $\sqrt{\tau_t}$")
    p1_true = axes[1].plot(months_axis, np.sqrt(out["tau_true"]), color="0.50", ls="--", lw=1.5, label="True Trend")
    p2 = ax2_twin.step(np.arange(out["N_months"]), out["x_lf"], color="0.65", where="post", lw=1.2, ls=":", label="Monthly Macro Driver $X_t$")
    axes[1].set_title(r"(b) Monthly Macro Driver and Extracted Secular Risk $\sqrt{\tau_t}$", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel(r"Secular Floor $\sqrt{\tau_t}$")
    ax2_twin.set_ylabel("Macro Driver $X_t$", color="0.40")
    lines = p1 + p1_true + p2
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc="upper left", fontsize=8.5, frameon=False)

    # Panel 3: Total Conditional Volatility
    axes[2].plot(months_axis, res.sigma, color="0.10", lw=1.2, label=r"Total Volatility $\sigma_{i,t} = \sqrt{\tau_t g_{i,t}}$")
    axes[2].plot(months_axis, np.sqrt(res.tau), color="0.45", ls="--", lw=1.8, label=r"Secular Baseline Floor $\sqrt{\tau_t}$")
    axes[2].set_title("(c) Total Volatility and Decoupled Secular Baseline", loc="left", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Time (Months)")
    axes[2].set_ylabel("Standard Deviation")
    axes[2].legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig_path = out_dir / "garch_midas_macro_volatility.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
