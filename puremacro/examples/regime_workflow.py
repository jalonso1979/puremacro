"""End-to-end regime workflow: Bai-Perron breaks → regime indicator
→ regime-conditional LP.

We simulate a two-regime DGP (a single break in the AR(1) coefficient),
detect the break with ``bai_perron_regression``, convert the break date
to a regime indicator with ``breaks_to_regimes``, then fit
``lp_state_dep`` using the indicator as the state variable. The
regime-dependent slopes recovered by the LP should match the true
high/low-regime slopes.

Run:
    python -m puremacro.examples.regime_workflow
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..tests.breaks import bai_perron_regression
from ..regimes import breaks_to_regimes
from ..lp.state_dep import lp_state_dep


def _simulate(T: int = 400, T_break: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T) * 0.6
    y = np.zeros(T)
    # Slope on x switches from 0.2 (low regime) to 1.2 (high regime) at T_break.
    for t in range(1, T):
        slope = 0.2 if t < T_break else 1.2
        y[t] = 0.5 * y[t - 1] + slope * x[t] + rng.standard_normal() * 0.4
    idx = pd.date_range("1990", periods=T, freq="QE")
    return pd.DataFrame({"y": y, "x": x}, index=idx), T_break


def run_demo() -> dict:
    df, T_break = _simulate()

    # Step 1: detect breaks in the relationship y_t ~ x_t (with intercept).
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    bp = bai_perron_regression(df["y"].values, X, max_breaks=3)
    selected = bp["selected_m"]
    breaks = bp["breaks"][selected]

    # Step 2: convert breaks → regime indicator.
    reg = breaks_to_regimes(breaks, len(df))
    df["state"] = reg.astype(float)

    # Step 3: state-dependent LP using the threshold-style indicator.
    # We use lp_state_dep with the threshold transition (state >= 0.5 → high).
    irf = lp_state_dep(df, y="y", x="x", state="state",
                       horizons=range(0, 9), n_lags=2,
                       transition="threshold", threshold=0.0,
                       alpha=0.10)
    return {"df": df, "T_break_true": T_break, "bp": bp,
            "regimes": reg, "irf": irf}


def main() -> None:
    out = run_demo()
    bp = out["bp"]
    print("Regime workflow: Bai-Perron → regimes → state-dep LP")
    print(f"  Selected number of breaks (sequential F): m̂ = "
          f"{bp['selected_m']}")
    print(f"  True break index = {out['T_break_true']}")
    print(f"  Detected break(s) = {bp['breaks'][bp['selected_m']]}")
    print()
    print("State-dependent LP (true low slope ≈ 0.2, high slope ≈ 1.2):")
    print(out["irf"][["h", "beta_L", "beta_H"]].round(3).to_string(index=False))

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        df = out["df"]
        irf = out["irf"]
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6),
                                  gridspec_kw=dict(height_ratios=[1, 1.2]))
        # Panel 1: y series with detected breaks.
        axes[0].plot(df.index, df["y"], color="0.4", lw=0.6)
        for b in bp["breaks"][bp["selected_m"]]:
            axes[0].axvline(df.index[b], color="0.0", lw=0.8, ls="--")
        axes[0].set_title("y with detected break dates")

        # Panel 2: regime-dependent LP β_L vs β_H.
        h = irf["h"].values
        axes[1].plot(h, irf["beta_L"], color="0.0", lw=1.0,
                     label="β_L (low regime)")
        axes[1].fill_between(h, irf["lo_L"], irf["hi_L"], color="0.0",
                              alpha=0.15)
        axes[1].plot(h, irf["beta_H"], color="0.5", lw=1.0,
                     label="β_H (high regime)")
        axes[1].fill_between(h, irf["lo_H"], irf["hi_H"], color="0.5",
                              alpha=0.20)
        axes[1].axhline(0, color="0.5", lw=0.4)
        axes[1].set_xlabel("Horizon h"); axes[1].set_ylabel("β_h")
        axes[1].set_title("State-dependent LP slopes")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "regime_workflow.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'regime_workflow.png'}")


if __name__ == "__main__":
    main()
