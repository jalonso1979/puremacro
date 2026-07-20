"""Wavelet variance decomposition of a synthetic GDP-like series.

Build a series with three superposed cyclical components: high-
frequency noise, a 2-year (8-quarter) cycle, and a 5-year (20-quarter)
cycle. Then decompose its variance across wavelet scales to show
which scales carry which fraction of the total variance — the
wavelet analogue of the Welch-PSD business-cycle band test.

Also demonstrate wavelet coherence with a co-moving series that lags
the long cycle by 4 quarters.

Run:
    python -m puremacro.examples.wavelet_business_cycle

Español
-------
Descomposición de la varianza por wavelets de una serie sintética similar al PIB.

Se construye una serie con tres componentes cíclicos superpuestos: ruido de
alta frecuencia, un ciclo de 2 años (8 trimestres) y un ciclo de 5 años
(20 trimestres). Luego se descompone su varianza entre escalas de wavelet para
mostrar qué fracción de la varianza total corresponde a cada escala — el
análogo por wavelets de la prueba de banda del ciclo económico de Welch-PSD.

También se ilustra la coherencia wavelet con una serie que covaría con la
variable original y está rezagada 4 trimestres en el ciclo largo.

Ejecución:
    python -m puremacro.examples.wavelet_business_cycle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..wavelet import wavelet_variance, wavelet_coherence


def _simulate(T: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    short_cycle = 0.5 * np.cos(2 * np.pi * t / 4.0)    # 4-quarter cycle
    biz_cycle = 1.0 * np.sin(2 * np.pi * t / 16.0)     # 4-year (16-q) cycle
    long_cycle = 0.8 * np.sin(2 * np.pi * t / 64.0)    # 16-year (64-q) cycle
    noise = 0.3 * rng.standard_normal(T)
    x = short_cycle + biz_cycle + long_cycle + noise
    # Lagged co-mover: shifts long cycle by 4 quarters.
    y = (short_cycle + biz_cycle
         + 0.8 * np.sin(2 * np.pi * (t - 4) / 64.0)
         + 0.3 * rng.standard_normal(T))
    return x, y


def run_demo() -> dict:
    x, y = _simulate()
    wv = wavelet_variance(x, level=6)
    wc = wavelet_coherence(x, y, level=6)
    return {"x": x, "y": y, "variance": wv, "coherence": wc}


def main() -> None:
    out = run_demo()
    wv = out["variance"]
    wc = out["coherence"]
    print("Wavelet variance decomposition of a 3-cycle series")
    print(f"  T = {len(out['x'])} observations,  total variance = "
          f"{wv.total:.3f}")
    print()
    print("  scale  period band (obs)   variance   share")
    for j, ((lo, hi), v, s) in enumerate(zip(wv.period_bands,
                                              wv.variance_per_scale,
                                              wv.share), start=1):
        flag = ""
        if 4 in range(lo, hi + 1) or 16 in range(lo, hi + 1) or 64 in range(lo, hi + 1):
            flag = "  ← built-in cycle"
        print(f"   j={j}   [{lo:>4d}, {hi:>4d}]      {v:>7.3f}    "
              f"{s:>5.2%}{flag}")
    print()
    print("Wavelet coherence between x and lagged-y (true coupling at long cycle):")
    print("  scale  period band (obs)   coherence")
    for j, ((lo, hi), c) in enumerate(zip(wc.period_bands,
                                            wc.coherence_per_scale),
                                       start=1):
        print(f"   j={j}   [{lo:>4d}, {hi:>4d}]      {c:+.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5))
        axes[0].plot(out["x"], color="0.0", lw=0.7)
        axes[0].set_title("Synthetic series with 4-, 16-, and 64-quarter cycles")
        axes[0].set_xlabel("t (quarters)")
        # Bar chart of variance shares.
        levels = np.arange(1, 7)
        axes[1].bar(levels - 0.2, wv.share, width=0.4, color="0.2",
                     label="variance share")
        axes[1].bar(levels + 0.2, wc.coherence_per_scale,
                     width=0.4, color="0.6", label="coherence with y")
        axes[1].set_xticks(levels)
        axes[1].set_xticklabels([f"j={j}\n[{lo},{hi}]"
                                  for j, (lo, hi) in enumerate(
                                      wv.period_bands, start=1)])
        axes[1].legend(frameon=False)
        axes[1].set_title("Per-scale variance share + coherence")
        fig.tight_layout()
        fig.savefig(out_dir / "wavelet_business_cycle.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'wavelet_business_cycle.png'}")


if __name__ == "__main__":
    main()
