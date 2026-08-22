"""HFI monetary-policy shock: synthetic Gertler-Karadi 2015-style pipeline.

Demonstrates:
    1. Construct GK 2015 surprises from synthetic FFR-futures price changes.
    2. Aggregate announcement-day surprises to monthly bins.
    3. Run proxy-SVAR identification using the surprise as external instrument.
    4. Plot the IRF of macro variables to a unit MP shock with bootstrap bands.
    5. Print Olea-Pflueger effective F and CI level.

No real data is shipped — fully synthetic, runs offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.hfi import aggregate_to_period, gk2015_surprise
from puremacro.var.identify.proxy import proxy_svar


def main():
    rng = np.random.default_rng(2026)
    # ---- 1. Synthetic monthly macro panel ----
    T = 240
    n = 3
    var_names = ["IP_growth", "CPI", "FFR"]
    Y = np.zeros((T, n))
    for t in range(2, T):
        Y[t] = 0.6 * Y[t - 1] - 0.1 * Y[t - 2] + 0.3 * rng.standard_normal(n)

    # ---- 2. Synthetic FOMC-day surprises ----
    n_announce = T
    rate_pre = 95.0 * np.ones(n_announce)
    rate_post = rate_pre + 0.05 * rng.standard_normal(n_announce)
    days_remaining = rng.integers(5, 28, size=n_announce)
    surprise = gk2015_surprise(rate_pre, rate_post, days_remaining,
                               days_in_month=30)
    dates = pd.date_range("2000-01-15", periods=n_announce, freq="MS")
    z = aggregate_to_period(surprise, dates, freq="M").values

    # ---- 3. Proxy-SVAR ----
    res = proxy_svar(Y, p=2, horizon=24, instrument_series=z,
                     n_boot=300, ci=0.9, seed=0)

    print(res.summary())

    # ---- 4. Plot ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    H = res.irf_point.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True)
    horizons = np.arange(H)
    for j in range(n):
        ax = axes[j]
        ax.plot(horizons, res.irf_point[:, j, 0], "b-", label="point")
        ax.fill_between(horizons, res.irf_lower[:, j, 0], res.irf_upper[:, j, 0],
                        color="b", alpha=0.2, label=f"{int(res.ci * 100)}% CI")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{var_names[j]} ↑ MP shock")
        ax.set_xlabel("horizon (months)")
    axes[0].legend(loc="best")
    fig.suptitle(f"HFI proxy-SVAR (synthetic, OP F={res.first_stage_F:.1f})")
    fig.tight_layout()
    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        fig.savefig(out_dir / "hfi_gertler_karadi.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'hfi_gertler_karadi.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
