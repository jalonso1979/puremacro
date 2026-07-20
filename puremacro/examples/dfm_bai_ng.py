"""High-dimensional dynamic factor model: Bai-Ng IC + PCA factors.

Simulate a 30-variable monthly panel driven by 2 latent factors plus
idiosyncratic noise. Bai-Ng (2002) IC1/IC2/IC3 are evaluated on a
grid k = 0..8 to choose the number of factors; PCA then extracts the
factors and we compare to the true latent paths.

Run:
    python -m puremacro.examples.dfm_bai_ng
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..factor import bai_ng_ic, pca_factors


def _simulate(T: int = 240, n: int = 30, k_true: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Persistent latent factors.
    F = np.zeros((T, k_true))
    for t in range(1, T):
        F[t] = 0.85 * F[t - 1] + rng.standard_normal(k_true) * 0.4
    Lambda = rng.standard_normal((n, k_true)) * 0.8
    eps = rng.standard_normal((T, n)) * 0.6
    X = F @ Lambda.T + eps
    return X, F, Lambda


def run_demo() -> dict:
    X, F_true, Lambda_true = _simulate()
    ic = bai_ng_ic(X, kmax=8)
    fact = pca_factors(X, k=ic["k_hat_IC2"])
    return {"X": X, "F_true": F_true, "Lambda_true": Lambda_true,
            "ic": ic, "fact": fact}


def main() -> None:
    out = run_demo()
    print("Static DFM with Bai-Ng (2002) factor selection")
    print(f"  X shape: {out['X'].shape},  true k = 2")
    print()
    print("Bai-Ng IC table (k → IC values):")
    print(out["ic"]["table"].round(3).to_string(index=False))
    print()
    print(f"  k̂ via IC1: {out['ic']['k_hat_IC1']}")
    print(f"  k̂ via IC2: {out['ic']['k_hat_IC2']}")
    print(f"  k̂ via IC3: {out['ic']['k_hat_IC3']}")
    print()
    fact = out["fact"]
    print(f"  Recovered factors variance shares: "
          f"{np.round(fact['variance_share'], 3).tolist()}")
    print(f"  Cumulative shares: "
          f"{np.round(fact['cumulative_share'], 3).tolist()}")
    # Test how well the recovered factors span the true factors.
    F_true = out["F_true"]
    F_hat = fact["factors"]
    proj = np.linalg.lstsq(F_true, F_hat, rcond=None)[0]
    F_hat_recon = F_true @ proj
    r2 = 1 - ((F_hat - F_hat_recon) ** 2).sum() / (F_hat ** 2).sum()
    print(f"  R² of recovered factors on true factors: {r2:.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
        for k in range(F_hat.shape[1]):
            sign = np.sign(np.corrcoef(F_hat[:, k], F_true[:, k])[0, 1])
            axes[0].plot(F_true[:, k], color=f"{0.6 - k*0.3:.2f}", lw=0.7,
                         label=f"true F_{k}")
            axes[0].plot(sign * F_hat[:, k], color=f"{0.0 + k*0.3:.2f}", lw=1.0,
                         ls="--", label=f"PCA F̂_{k}")
        axes[0].set_title("True factors vs PCA-recovered factors (sign-aligned)")
        axes[0].legend(frameon=False, fontsize=8)
        ic_df = out["ic"]["table"]
        for ic_name in ("IC1", "IC2", "IC3"):
            axes[1].plot(ic_df["k"], ic_df[ic_name], marker="o",
                         label=ic_name, lw=0.8)
        axes[1].axvline(out["ic"]["k_hat_IC2"], color="0.0", lw=0.5,
                         ls=":", label=f"k̂={out['ic']['k_hat_IC2']}")
        axes[1].set_xlabel("k"); axes[1].set_ylabel("IC value")
        axes[1].set_title("Bai-Ng information criteria")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "dfm_bai_ng.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'dfm_bai_ng.png'}")


if __name__ == "__main__":
    main()
