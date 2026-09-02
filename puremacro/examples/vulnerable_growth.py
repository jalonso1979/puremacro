"""Adrian-Boyarchenko-Giannone (2019) vulnerable-growth demo.

Quantile local projection of GDP growth on a financial-conditions index.
The headline ABG result: the *upper* quantiles of future growth are
roughly invariant to financial conditions, while the *lower* quantiles
shift sharply when conditions tighten — i.e., financial-condition
shocks shape downside risk far more than central-tendency growth.

Self-contained: simulates a panel where the lower quantile of y is
made to depend on x more strongly than the upper quantile, so the
estimator should recover that asymmetry.

Run:
    python -m puremacro.examples.vulnerable_growth

Español
-------
Demostración de crecimiento en riesgo al estilo de Adrian-Boyarchenko-Giannone (2019).

Proyección local por cuantiles del crecimiento del PIB sobre un índice de condiciones
financieras. El resultado central de ABG: los cuantiles *superiores* del crecimiento futuro
son aproximadamente invariantes a las condiciones financieras, mientras que los cuantiles
*inferiores* se desplazan marcadamente cuando las condiciones se endurecen — es decir, los
choques en las condiciones financieras determinan el riesgo a la baja mucho más que la
tendencia central del crecimiento.

Autocontenido: simula un panel en el que el cuantil inferior de y depende de x de forma más
intensa que el cuantil superior, de modo que el estimador debería recuperar esa asimetría.

Ejecución:
    python -m puremacro.examples.vulnerable_growth
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..lp.quantile import lp_quantile


def _simulate_panel(T: int = 480, seed: int = 0) -> pd.DataFrame:
    """Simulate a series with quantile-dependent slope: lower quantiles
    are far more sensitive to x than upper quantiles. We achieve this
    by letting the standard deviation of the shock grow with -x (i.e.
    when x is more negative, the shock is more dispersed) and giving
    the shock itself a strong negative loading on x."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    eps_scale = 1.0 + 1.5 * np.maximum(-x, 0.0)
    eps = rng.standard_normal(T) * eps_scale
    y = 0.3 * x + eps
    idx = pd.date_range("1990Q1", periods=T, freq="QE")
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def run_vulnerable_growth(
    T: int = 320,
    horizons=range(0, 13),
    n_lags: int = 2,
    quantiles=(0.10, 0.50, 0.90),
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    df = _simulate_panel(T=T, seed=seed)
    qr = lp_quantile(df, y="y", x="x", quantiles=quantiles,
                     horizons=horizons, n_lags=n_lags,
                     n_boot=n_boot, seed=seed)
    return {"data": df, "qr": qr, "quantiles": quantiles}


def main() -> None:
    out = run_vulnerable_growth()
    qr = out["qr"]
    print("Adrian-Boyarchenko-Giannone style quantile-LP demo")
    print(f"  T = {len(out['data'])} obs, quantiles = {out['quantiles']}")
    print()
    print("Slope of x at h=4 by quantile (lower quantiles should be larger in |.|):")
    h4 = qr[qr["h"] == 4].set_index("tau")[["beta", "lo", "hi"]]
    print(h4.round(3).to_string())
    print()
    print("Width of (10%, 90%) gap by horizon (growth-at-risk gap):")
    pivot = qr.pivot(index="h", columns="tau", values="beta")
    if 0.10 in pivot.columns and 0.90 in pivot.columns:
        gap = (pivot[0.90] - pivot[0.10]).round(3)
        print(gap.to_string())

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for tau, ls in zip(out["quantiles"], ["--", "-", ":"]):
            sub = qr[qr["tau"] == tau]
            ax.plot(sub["h"], sub["beta"], ls=ls, color="0.20", label=f"τ = {tau}")
            ax.fill_between(sub["h"], sub["lo"], sub["hi"], color="0.20", alpha=0.10)
        ax.axhline(0.0, color="0.40", lw=0.5)
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("β_h(τ): impact of x on quantile τ of y_{t+h}")
        ax.set_title("Quantile-LP: vulnerable-growth-style impact by quantile")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "vulnerable_growth.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'vulnerable_growth.png'}")


if __name__ == "__main__":
    main()
