"""Real-time vintages and forecast revisions.

Simulate an ALFRED-style panel where each calendar date has multiple
"vintages" (publication versions), with later vintages typically
revising earlier numbers. Then track how a simple AR(1) one-quarter-
ahead forecast revises as new vintages arrive.

This is the workflow you'd use on real FRED-ALFRED data fetched via
``puremacro.fetch.fetch_fred_alfred`` — see
``examples.fetch_fred_demo`` for the live-network version.

Run:
    python -m puremacro.examples.vintage_revisions
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..vintages import as_of, align_vintages, forecast_revision


def _simulate_real_time_panel(seed: int = 0) -> pd.DataFrame:
    """Build a long DataFrame (date, vintage, value).

    The "true" series follows a random walk; each calendar period gets
    three vintages (initial / first revision / final), each closer to
    truth.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2010Q1", "2018Q4", freq="Q").to_timestamp(how="start")
    n_q = len(quarters)
    truth = np.cumsum(rng.standard_normal(n_q) * 0.5) + 100.0
    rows = []
    for i, q in enumerate(quarters):
        # Initial release: 1 quarter after the reference period.
        vintages = [
            (q + pd.DateOffset(months=4),  truth[i] + rng.standard_normal() * 0.6),  # initial
            (q + pd.DateOffset(months=10), truth[i] + rng.standard_normal() * 0.3),  # 1st revision
            (q + pd.DateOffset(months=22), truth[i] + rng.standard_normal() * 0.1),  # final
        ]
        for vint, val in vintages:
            rows.append({"date": q, "vintage": vint, "value": val})
    return pd.DataFrame(rows), truth, quarters


def _ar1_one_step(series: pd.Series) -> float:
    """One-quarter-ahead forecast from a fitted AR(1) on the series level."""
    s = series.dropna().astype(float)
    if len(s) < 5:
        return float(s.iloc[-1]) if len(s) else float("nan")
    y = s.values
    rho = np.corrcoef(y[:-1], y[1:])[0, 1]
    return float(y[-1] + rho * (y[-1] - y[-2]))


def run_demo() -> dict:
    panel, truth, quarters = _simulate_real_time_panel()
    snap_dates = pd.to_datetime([
        "2014-06-01", "2015-06-01", "2016-06-01",
        "2017-06-01", "2018-06-01",
    ])
    wide = align_vintages(panel, snap_dates)
    revisions = forecast_revision(panel, snap_dates, _ar1_one_step)
    return {
        "panel": panel, "truth": truth, "quarters": quarters,
        "wide": wide, "snap_dates": snap_dates,
        "revisions": revisions,
    }


def main() -> None:
    out = run_demo()
    print("Real-time vintage panel + forecast-revision demo")
    print(f"  Calendar periods: {len(out['quarters'])}, "
          f"vintages per period: 3 (initial / 1st-revision / final)")
    print()
    print("First 6 rows of as-of(2015-06-01) snapshot:")
    snap = as_of(out["panel"], "2015-06-01")
    print(snap.head(6).to_string())
    print()
    print("Forecast revisions across vintages (one-quarter-ahead AR(1)):")
    print(out["revisions"].round(3).to_string())
    print()
    print("Forecast revision range:",
          f"max - min = {out['revisions'].max() - out['revisions'].min():.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        wide = out["wide"]
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=False)
        for v in wide.columns:
            axes[0].plot(wide.index, wide[v], lw=0.7,
                         label=f"as of {pd.Timestamp(v).strftime('%Y-%m')}")
        axes[0].set_title("Same calendar series under different vintages")
        axes[0].set_ylabel("value")
        axes[0].legend(frameon=False, fontsize=8, loc="best")
        axes[1].plot(out["revisions"].index, out["revisions"].values,
                     marker="o", color="0.0", lw=1.0)
        axes[1].set_title("AR(1) forecast as we re-run on each vintage")
        axes[1].set_ylabel("forecast")
        axes[1].set_xlabel("vintage date")
        fig.tight_layout()
        fig.savefig(out_dir / "vintage_revisions.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'vintage_revisions.png'}")


if __name__ == "__main__":
    main()
