"""MIDAS nowcasting: quarterly y on monthly x.

Simulate a quarterly outcome whose true DGP weights only the most
recent two months of a monthly indicator, then compare:
  - U-MIDAS: unrestricted (3 separate coefficients on the 3 monthly lags),
  - Beta-MIDAS: a 2-parameter Beta-pdf kernel.

Beta-MIDAS should recover monotonically declining weights even in
small samples; U-MIDAS suffers from variance with K large.

Run:
    python -m puremacro.examples.midas_quarterly_monthly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..midas import u_midas, beta_midas


def _simulate(n_low: int = 120, K: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_hf = rng.standard_normal(n_low * K)
    # Smooth the monthly indicator a bit so it looks like a real series.
    for i in range(1, len(x_hf)):
        x_hf[i] = 0.4 * x_hf[i - 1] + x_hf[i]
    true_w = np.array([0.7, 0.2, 0.1])  # most-recent month carries most weight
    y_lf = np.zeros(n_low)
    for i in range(n_low):
        seg = x_hf[i * K:(i + 1) * K][::-1]  # most-recent first
        y_lf[i] = 0.5 + 0.8 * (true_w @ seg) + rng.standard_normal() * 0.4
    return y_lf, x_hf, true_w


def run_demo() -> dict:
    y_lf, x_hf, true_w = _simulate()
    K = 3
    um = u_midas(y_lf, x_hf, K=K)
    bm = beta_midas(y_lf, x_hf, K=K)
    return {"y_lf": y_lf, "x_hf": x_hf, "K": K, "true_w": true_w,
            "u_midas": um, "beta_midas": bm}


def main() -> None:
    out = run_demo()
    um = out["u_midas"]; bm = out["beta_midas"]
    print("MIDAS nowcasting (quarterly y on K=3 monthly x lags)")
    print(f"  N quarters = {len(out['y_lf'])}")
    print()
    print(f"  True weights (most-recent first): {out['true_w'].tolist()}")
    print()
    print(f"  U-MIDAS:")
    print(f"    intercept = {um.intercept:+.3f}")
    print(f"    coefficients (most-recent first) = "
          f"{np.round(um.beta, 3).tolist()}")
    print(f"    R² = {um.R2:.3f}")
    print()
    print(f"  Beta-MIDAS:")
    print(f"    intercept = {bm.intercept:+.3f}")
    print(f"    slope β   = {bm.beta:+.3f}")
    print(f"    weights   = {np.round(bm.weights, 3).tolist()}")
    print(f"    θ1, θ2    = {bm.theta1:.2f}, {bm.theta2:.2f}")
    print(f"    R² = {bm.R2:.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
        # Left: weights
        x_pos = np.arange(out["K"])
        axes[0].bar(x_pos - 0.2, out["true_w"], width=0.2, color="0.7",
                    label="true")
        # U-MIDAS weights are unnormalised regression coefficients —
        # rescale by their sum to make the comparison fair.
        u_w = um.beta / max(np.abs(um.beta).sum(), 1e-9)
        axes[0].bar(x_pos, u_w, width=0.2, color="0.4",
                    label="U-MIDAS (norm.)")
        axes[0].bar(x_pos + 0.2, bm.weights, width=0.2, color="0.0",
                    label="Beta-MIDAS")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"L{k}" for k in range(out["K"])])
        axes[0].set_xlabel("Monthly lag (most recent → oldest)")
        axes[0].set_ylabel("Weight")
        axes[0].legend(frameon=False)
        axes[0].set_title("Recovered MIDAS weights")
        # Right: in-sample fit
        axes[1].plot(out["y_lf"], color="0.4", lw=0.7, label="actual")
        axes[1].plot(um.fitted, color="0.0", lw=1.0,
                     label=f"U-MIDAS fit (R²={um.R2:.2f})")
        axes[1].plot(bm.fitted, color="0.5", ls="--", lw=1.0,
                     label=f"Beta-MIDAS (R²={bm.R2:.2f})")
        axes[1].set_xlabel("Quarter")
        axes[1].set_ylabel("y")
        axes[1].legend(frameon=False)
        axes[1].set_title("MIDAS in-sample fit")
        fig.tight_layout()
        fig.savefig(out_dir / "midas_quarterly_monthly.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'midas_quarterly_monthly.png'}")


if __name__ == "__main__":
    main()
