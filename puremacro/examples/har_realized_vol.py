"""Corsi (2009) HAR-RV regression on simulated intra-day returns.

We simulate ~3 years of daily 5-minute return paths, compute daily
realized variance and bipower variation, and fit the heterogeneous
autoregressive model

    log RV_t = β_0 + β_d log RV_{t-1}
                    + β_w mean(log RV_{t-2..t-5})
                    + β_m mean(log RV_{t-6..t-22})
                    + ε_t.

The DGP gives RV genuine multi-scale persistence so the daily / weekly
/ monthly coefficients should each be detectable.

Run:
    python -m puremacro.examples.har_realized_vol

Español
-------
Regresión HAR-RV de Corsi (2009) sobre retornos intradía simulados.

Se simulan aproximadamente 3 años de trayectorias de retornos diarios a 5 minutos, se
computan la varianza realizada diaria y la variación bipotencia, y se ajusta el modelo
autoregresivo heterogéneo

    log RV_t = β_0 + β_d log RV_{t-1}
                    + β_w mean(log RV_{t-2..t-5})
                    + β_m mean(log RV_{t-6..t-22})
                    + ε_t.

El proceso generador de datos otorga a la volatilidad realizada persistencia genuina a múltiples
escalas, de modo que los coeficientes diario, semanal y mensual deberían ser detectables
individualmente.

Ejecución:
    python -m puremacro.examples.har_realized_vol
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..realized_vol import (
    realized_variance, bipower_variation, realized_jump, har_rv,
)


def _simulate(n_days: int = 700, n_intraday: int = 78, seed: int = 0):
    """Multi-scale stochastic-volatility daily DGP, then within-day
    Brownian increments to provide intra-day returns."""
    rng = np.random.default_rng(seed)
    log_sigma = np.zeros(n_days)
    short = 0.0; mid = 0.0; long = 0.0
    for t in range(1, n_days):
        short = 0.7 * short + rng.standard_normal() * 0.3
        mid = 0.93 * mid + rng.standard_normal() * 0.10
        long = 0.99 * long + rng.standard_normal() * 0.04
        log_sigma[t] = -1.0 + short + mid + long
    sigma = np.exp(log_sigma)
    intraday_returns = []
    for t in range(n_days):
        daily_ret_var = sigma[t] ** 2
        per_step = np.sqrt(daily_ret_var / n_intraday)
        intraday_returns.append(per_step * rng.standard_normal(n_intraday))
    return intraday_returns, sigma


def run_demo() -> dict:
    intraday, sigma_true = _simulate()
    rv = np.array([realized_variance(d) for d in intraday])
    bv = np.array([bipower_variation(d) for d in intraday])
    rj = np.array([realized_jump(rv[t], bv[t]) for t in range(len(rv))])
    har = har_rv(rv)
    return {"sigma_true": sigma_true, "rv": rv, "bv": bv, "rj": rj,
            "har": har}


def main() -> None:
    out = run_demo()
    har = out["har"]
    print("Corsi (2009) HAR-RV regression on simulated intra-day returns")
    print(f"  N days = {len(out['rv'])},  intra-day obs/day = "
          f"{out['rv'].shape}")
    print()
    print(f"  Coefficients (HAC SE):")
    print(f"    β_0   = {har['intercept']:+.3f}  ({har['se_intercept']:.3f})")
    print(f"    β_d   = {har['beta_d']:+.3f}  ({har['se_d']:.3f})  daily")
    print(f"    β_w   = {har['beta_w']:+.3f}  ({har['se_w']:.3f})  weekly")
    print(f"    β_m   = {har['beta_m']:+.3f}  ({har['se_m']:.3f})  monthly")
    print(f"  R²    = {har['R2']:.3f},  HAC bandwidth = {har['hac_lags']}")
    print(f"  Mean realized jump share: {out['rj'].mean():.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
        axes[0].plot(out["sigma_true"], color="0.6", lw=0.6,
                     label="true σ_t (log)")
        axes[0].set_ylabel("true daily σ")
        axes[0].set_yscale("log")
        axes[0].legend(frameon=False, loc="upper right")

        log_rv = np.log(np.maximum(out["rv"], 1e-12))
        # Plot fitted vs realised on the (T-22) post-window.
        target = log_rv[22:]
        fitted = har["fitted"]
        axes[1].plot(np.arange(22, 22 + len(target)), target,
                     color="0.4", lw=0.6, label="actual log RV")
        axes[1].plot(np.arange(22, 22 + len(fitted)), fitted,
                     color="0.0", lw=0.9, label=f"HAR fit (R²={har['R2']:.2f})")
        axes[1].set_xlabel("day"); axes[1].set_ylabel("log RV")
        axes[1].legend(frameon=False, loc="upper right")
        fig.suptitle("HAR-RV: heterogeneous-autoregressive model fit")
        fig.tight_layout()
        fig.savefig(out_dir / "har_realized_vol.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'har_realized_vol.png'}")


if __name__ == "__main__":
    main()
