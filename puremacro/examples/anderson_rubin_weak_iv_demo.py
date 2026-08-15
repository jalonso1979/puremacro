"""Weak-instrument-robust Anderson-Rubin inference for local projections (LP-IV).

Compares conventional 2SLS Wald confidence intervals (which suffer from severe
undercoverage and false precision under weak identification) with exact
closed-form Anderson-Rubin (1949) confidence sets under Newey-West HAC errors.

Run:
    python -m puremacro.examples.anderson_rubin_weak_iv_demo

Español
-------
Inferencia robusta a instrumentos débiles de Anderson-Rubin para proyecciones locales (LP-IV).

Compara los intervalos de confianza asintóticos de Wald/MC2E con los conjuntos
exactos de Anderson-Rubin (1949) bajo errores HAC de Newey-West ante instrumentos
fuertes y débiles.

Ejecución:
    python -m puremacro.examples.anderson_rubin_weak_iv_demo
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..lp import lp_iv


def run_anderson_rubin_demo(T: int = 240, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    horizons = list(range(0, 10))

    # Plant structural shock and dynamic propagation
    shock = rng.normal(0, 1.0, size=T)
    x = 0.85 * shock + rng.normal(0, 1.1, size=T)

    beta_true = np.array([0.0, -0.4, -0.75, -1.0, -1.15, -1.2, -1.05, -0.85, -0.6, -0.3])
    y = np.zeros(T)
    for t in range(T):
        for h in range(len(beta_true)):
            if t - h >= 0:
                y[t] += beta_true[h] * shock[t - h]
        y[t] += rng.normal(0, 0.5)

    # Strong instrument (F > 20)
    z_strong = 0.9 * shock + rng.normal(0, 0.4, size=T)

    # Weak instrument (F < 5)
    z_weak = 0.12 * shock + rng.normal(0, 1.0, size=T)

    df_strong = pd.DataFrame({"y": y, "x": x, "z": z_strong})
    df_weak = pd.DataFrame({"y": y, "x": x, "z": z_weak})

    res_strong = lp_iv(
        df=df_strong, y="y", x="x", z="z",
        horizons=horizons, n_lags=2, anderson_rubin=True,
    )

    res_weak = lp_iv(
        df=df_weak, y="y", x="x", z="z",
        horizons=horizons, n_lags=2, anderson_rubin=True,
    )

    return {
        "beta_true": beta_true,
        "res_strong": res_strong,
        "res_weak": res_weak,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Estimating LP-IV under Strong and Weak Instruments with Anderson-Rubin...")
    out = run_anderson_rubin_demo()
    res_s = out["res_strong"]
    res_w = out["res_weak"]

    print(f"  Strong Instrument First-Stage F: {res_s['first_stage_f'].mean():.2f}")
    print(f"  Weak Instrument First-Stage F:   {res_w['first_stage_f'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=True)

    # Panel 1: Strong Instrument
    ax = axes[0]
    ax.plot(res_s["h"], res_s["beta"], color="0.00", lw=2.0, label=r"LP-IV $\hat{\beta}_h$")
    ax.plot(res_s["h"], out["beta_true"], color="0.60", ls="--", lw=1.5, label="True Effect")
    ax.fill_between(res_s["h"], res_s["lo"], res_s["hi"], color="0.75", alpha=0.5, label="Wald 95% CI")
    ax.plot(res_s["h"], res_s["ar_lo"], color="0.00", ls=":", lw=1.5, label="AR 95% Set")
    ax.plot(res_s["h"], res_s["ar_hi"], color="0.00", ls=":", lw=1.5)
    ax.axhline(0, color="0.70", ls=":", lw=0.8)
    ax.set_title(f"(a) Strong Instrument ($F \\approx {res_s['first_stage_f'].mean():.1f}$)", loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Horizon $h$ (Quarters)")
    ax.set_ylabel("Impulse Response")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    # Panel 2: Weak Instrument
    ax = axes[1]
    ax.plot(res_w["h"], res_w["beta"], color="0.00", lw=2.0, label=r"LP-IV $\hat{\beta}_h$")
    ax.plot(res_w["h"], out["beta_true"], color="0.60", ls="--", lw=1.5, label="True Effect")
    ax.fill_between(res_w["h"], np.clip(res_w["lo"], -6, 4), np.clip(res_w["hi"], -6, 4), color="0.75", alpha=0.5, label="Wald 95% (Naive)")
    
    bounded = res_w["ar_set_type"] == "bounded"
    ax.plot(res_w.loc[bounded, "h"], res_w.loc[bounded, "ar_lo"], color="0.00", ls=":", lw=1.8, label="AR 95% Set")
    ax.plot(res_w.loc[bounded, "h"], res_w.loc[bounded, "ar_hi"], color="0.00", ls=":", lw=1.8)
    ax.axhline(0, color="0.70", ls=":", lw=0.8)
    ax.set_title(f"(b) Weak Instrument ($F \\approx {res_w['first_stage_f'].mean():.1f}$)", loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Horizon $h$ (Quarters)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig_path = out_dir / "anderson_rubin_weak_iv_demo.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
