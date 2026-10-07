"""HFI monetary-policy shock: synthetic Gertler-Karadi 2015-style pipeline.

Demonstrates:
    1. Simulate a monthly structural VAR in which a monetary-policy (MP) shock
       raises the policy rate and lowers activity and inflation.
    2. Turn each month's MP shock into a noisy FOMC announcement surprise, as
       the market sees it: a move in the current-month fed-funds futures
       contract, quoted as a *price* ``100 - implied rate``.
    3. Build the Gertler-Karadi (2015) surprise with
       ``gk2015_surprise(..., quote="price")``. The price falls when the
       implied rate rises, so the function negates the price change and a
       tightening comes out positive, as in GK eq. (19).
    4. Aggregate announcement-day surprises to monthly bins.
    5. Run proxy-SVAR identification using the surprise as external
       instrument, normalised so the shock raises the policy rate, and plot
       the IRFs with bootstrap bands against the true DGP responses.
    6. Print the Olea-Pflueger effective F and the instrument's correlation
       with the planted shock, and show that forgetting ``quote="price"``
       reverses the instrument's sign.

No real data is shipped — fully synthetic, runs offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.hfi import aggregate_to_period, gk2015_surprise
from puremacro.var.identify.proxy import proxy_svar

VAR_NAMES = ["IP_growth", "CPI_inflation", "FFR"]
POLICY_IDX = 2  # position of the policy rate in VAR_NAMES

# Structural DGP  Y_t = A1 Y_{t-1} + B0 eps_t,  eps = [demand, supply, MP].
# Column 2 of B0 is the MP shock. All variables are in percentage points, so
# a one-s.d. MP shock raises the policy rate by 0.15 pp (15 bp) and lowers IP
# growth by 0.10 pp and inflation by 0.05 pp on impact.
A1 = np.array([
    [0.50, 0.00, -0.15],
    [0.05, 0.60, -0.05],
    [0.10, 0.10, 0.85],
])
B0 = np.array([
    [0.50, 0.20, -0.10],
    [0.10, -0.30, -0.05],
    [0.15, 0.10, 0.15],
])


def _true_irf(horizon: int) -> np.ndarray:
    """DGP responses to a one-s.d. MP shock, shape (horizon + 1, n)."""
    out = np.empty((horizon + 1, A1.shape[0]))
    x = B0[:, POLICY_IDX].copy()
    for h in range(horizon + 1):
        out[h] = x
        x = A1 @ x
    return out


def run_pipeline(seed: int = 2026, T: int = 360, horizon: int = 24,
                 n_boot: int = 300, noise_to_signal: float = 1.0) -> dict:
    """Simulate the DGP, build the GK surprise and estimate the proxy-SVAR.

    Returns a dict with the ``ProxySVARResult`` (``"result"``), the monthly
    instrument ``"z"``, the planted MP shock ``"eps_mp"``, the true impact
    column ``"b_true"``, the true IRFs ``"irf_true"`` and the correlations of
    the correct and of the mis-quoted instrument with the planted shock.
    Nothing is written to disk.
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Monthly structural VAR ----
    burn = 50
    n = len(VAR_NAMES)
    eps = rng.standard_normal((T + burn, n))
    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = A1 @ Y[t - 1] + B0 @ eps[t]
    Y, eps = Y[burn:], eps[burn:]
    eps_mp = eps[:, POLICY_IDX]

    # ---- 2. One FOMC announcement per month, on a random day ----
    months = pd.period_range("2000-01", periods=T, freq="M")
    days_in_month = months.days_in_month.to_numpy()
    day = np.array([rng.integers(1, d) for d in days_in_month])   # 1 .. D-1
    dates = months.start_time + pd.to_timedelta(day - 1, unit="D")
    days_remaining = days_in_month - day + 1    # announcement day included

    # The announcement moves the expected target rate by `target` (in pp);
    # the window also picks up unrelated news, so the surprise is a noisy
    # proxy for the MP shock. The current-month contract settles on the
    # month's AVERAGE rate, so its implied rate moves by only
    # target * days_remaining / days_in_month.
    target = B0[POLICY_IDX, POLICY_IDX] * (
        eps_mp + noise_to_signal * rng.standard_normal(T)
    )
    implied_pre = 4.0 + Y[:, POLICY_IDX]                 # percent per annum
    implied_post = implied_pre + target * days_remaining / days_in_month
    price_pre = 100.0 - implied_pre                      # CME quote convention
    price_post = 100.0 - implied_post

    # ---- 3. GK (2015) surprise from PRICE quotes: positive = tightening ----
    surprise = gk2015_surprise(price_pre, price_post, days_remaining,
                               days_in_month=days_in_month, quote="price")
    misquoted = gk2015_surprise(price_pre, price_post, days_remaining,
                                days_in_month=days_in_month)   # quote="rate"

    # ---- 4. Monthly instrument (one announcement per month, so "sum" maps
    # each surprise to its own month; method="gk2015" would instead split it
    # with the next month, GK's choice for a VAR in monthly-average rates) ----
    z = aggregate_to_period(surprise, dates, freq="M").to_numpy()

    # ---- 5. Proxy-SVAR: first-stage F on the policy-rate residual, shock
    # signed to raise the policy rate on impact ----
    res = proxy_svar(Y, p=2, horizon=horizon, instrument_series=z,
                     n_boot=n_boot, ci=0.9, seed=0,
                     shock_target_idx=POLICY_IDX,
                     sign_var=POLICY_IDX, sign=+1.0)

    return {
        "result": res,
        "var_names": list(VAR_NAMES),
        "z": z,
        "eps_mp": eps_mp,
        "b_true": B0[:, POLICY_IDX].copy(),
        "irf_true": _true_irf(horizon),
        "corr_with_true_shock": float(np.corrcoef(z, eps_mp)[0, 1]),
        "corr_with_true_shock_if_misquoted": float(np.corrcoef(misquoted, eps_mp)[0, 1]),
    }


def main():
    out = run_pipeline()
    res = out["result"]

    print(res.summary())
    print(f"  corr(instrument, true MP shock)            : "
          f"{out['corr_with_true_shock']:+.3f}")
    print(f"  same, prices passed without quote='price'  : "
          f"{out['corr_with_true_shock_if_misquoted']:+.3f}")
    print("  impact responses to a +1 s.d. MP shock (estimated vs. true):")
    for j, name in enumerate(out["var_names"]):
        print(f"    {name:14s} {res.irf_point[0, j, 0]:+.3f}   {out['b_true'][j]:+.3f}")

    # ---- 6. Plot ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    n = len(out["var_names"])
    H = res.irf_point.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True)
    horizons = np.arange(H)
    for j in range(n):
        ax = axes[j]
        ax.plot(horizons, res.irf_point[:, j, 0], "b-", label="proxy-SVAR")
        ax.fill_between(horizons, res.irf_lower[:, j, 0], res.irf_upper[:, j, 0],
                        color="b", alpha=0.2, label=f"{int(res.ci * 100)}% CI")
        ax.plot(horizons, out["irf_true"][:, j], "k--", lw=1.0, label="true (DGP)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{out['var_names'][j]}: +1 s.d. MP shock")
        ax.set_xlabel("horizon (months)")
    axes[0].set_ylabel("percentage points")
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
