"""Spectral analysis of a synthetic business-cycle-style series.

Simulate a quarterly series with both a strong 4-year cyclical
component and a high-frequency noise floor. Welch's PSD should peak
at frequency 1/16 cycles per quarter, and the business-cycle band
power (6–32 quarters) should dominate.

Pair with a co-moving series (the same cycle plus phase shift) to
demonstrate the cross-spectrum: high coherence at the cyclical
frequency, gain ≈ 1, phase ≈ a fixed lag.

Run:
    python -m puremacro.examples.spectral_business_cycle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spectral import welch_psd, welch_cross, business_cycle_band_power


def _simulate(T: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    cycle = np.sin(2 * np.pi * t / 16.0)  # 16-quarter (4-year) cycle
    noise = rng.standard_normal(T) * 0.4
    x = cycle + noise
    y = np.sin(2 * np.pi * (t - 2) / 16.0) + rng.standard_normal(T) * 0.3
    return x, y


def run_demo() -> dict:
    x, y = _simulate()
    f, Pxx = welch_psd(x, fs=1.0, n_seg=64)
    cross = welch_cross(x, y, fs=1.0, n_seg=64)
    bc_power = business_cycle_band_power(x, fs=1.0,
                                          low_period=6, high_period=32)
    return {"x": x, "y": y, "f": f, "Pxx": Pxx,
            "cross": cross, "bc_power": bc_power}


def main() -> None:
    out = run_demo()
    f = out["f"]
    Pxx = out["Pxx"]
    peak_f = float(f[np.argmax(Pxx)])
    peak_period = 1.0 / peak_f if peak_f > 0 else float("inf")
    print("Spectral analysis of synthetic quarterly business-cycle series")
    print(f"  T = {len(out['x'])} quarters")
    print(f"  PSD peak frequency: {peak_f:.4f} cycles/quarter "
          f"(period ≈ {peak_period:.1f} quarters; true 16)")
    print(f"  Business-cycle band (6–32 q) variance share: "
          f"{out['bc_power']:.2%}")
    print()
    cross = out["cross"]
    peak_idx = int(np.argmax(out["Pxx"]))
    print(f"  At peak frequency:")
    print(f"    coherence = {cross.coherence[peak_idx]:.3f}")
    print(f"    gain      = {cross.gain[peak_idx]:.3f}  (true 1.0)")
    print(f"    phase     = {cross.phase[peak_idx]:+.3f} rad   "
          f"(equiv. {cross.phase[peak_idx] * peak_period / (2*np.pi):+.2f} "
          f"quarter shift)")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5))
        axes[0].plot(f, Pxx, color="0.0", lw=1.0)
        axes[0].axvspan(1.0 / 32.0, 1.0 / 6.0, color="0.85", alpha=0.5,
                        label="business-cycle band")
        axes[0].axvline(1.0 / 16.0, color="0.5", lw=0.6, ls="--",
                        label="true cycle freq")
        axes[0].set_xlabel("Frequency (cycles/quarter)")
        axes[0].set_ylabel("PSD")
        axes[0].set_title("Welch periodogram of x")
        axes[0].legend(frameon=False)
        axes[1].plot(f, cross.coherence, color="0.0", lw=1.0,
                     label="coherence(x, y)")
        axes[1].axvline(1.0 / 16.0, color="0.5", lw=0.6, ls="--")
        axes[1].set_xlabel("Frequency (cycles/quarter)")
        axes[1].set_ylabel("Coherence")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Cross-spectrum: coherence between x and y")
        fig.tight_layout()
        fig.savefig(out_dir / "spectral_business_cycle.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'spectral_business_cycle.png'}")


if __name__ == "__main__":
    main()
