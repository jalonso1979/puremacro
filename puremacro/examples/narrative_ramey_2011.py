"""Replication-layer demo (workflow B): Ramey (2011) defense-news
narrative instrument → ``proxy_svar`` on a small US-style VAR →
spending-multiplier-style IRF.

The demo tries the public mirror first; if the network is unavailable
(or the mirror has moved) it falls back to a small bundled-style
synthetic series so the example still runs end-to-end.

Run:
    python -m puremacro.examples.narrative_ramey_2011
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import NarrativeInstrument
from ..narrative.replication import load_ramey_2011_defense
from ..narrative.types import NarrativeEvent


def _fallback_series() -> NarrativeInstrument:
    """Hand-coded Ramey-style defense-news events for the demo if the
    network mirror isn't reachable. NOT a publishable replication."""
    rows = [
        ("1950-Q3", +0.020, "Korean War mobilisation"),
        ("1965-Q1", +0.012, "Vietnam buildup"),
        ("1980-Q3", +0.008, "Carter defense increase"),
        ("1981-Q1", +0.010, "Reagan defense buildup"),
        ("2001-Q4", +0.007, "Post-9/11 defense increase"),
        ("2003-Q2", +0.006, "Iraq war"),
        ("2013-Q1", -0.005, "Sequester defense cut"),
    ]
    events = [
        NarrativeEvent(
            date=pd.Period(q, freq="Q").to_timestamp(),
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="both",
            subtarget="defense",
            sign=int(1 if v > 0 else -1),
            confidence=0.75,
            source_text=note,
            source_url="(synthetic fallback)",
            scoring_method="manual",
            metadata={"replication": "ramey_2011_synthetic_fallback"},
        )
        for q, v, note in rows
    ]
    return NarrativeInstrument.from_events(events, target="both",
                                              aggregation="sum")


def _simulate_us_var(T: int = 280, seed: int = 0) -> pd.DataFrame:
    """Three-variable VAR(2) on a quarterly grid: gov spending growth,
    GDP growth, hours growth. The demo runs the proxy SVAR on this."""
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.6, 0.10, 0.0],
        [0.10, 0.5, 0.10],
        [0.05, 0.10, 0.55],
    ])
    Sigma = np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]) * 0.4 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    idx = pd.date_range("1955-01-01", periods=T, freq="QS")
    return pd.DataFrame(Y, index=idx,
                          columns=["dlog_gov", "dlog_gdp", "dlog_hours"])


def run_demo() -> dict:
    try:
        inst = load_ramey_2011_defense()
        used_mirror = True
    except Exception as e:
        print(f"  [fallback] Ramey mirror unreachable ({type(e).__name__}); "
              "using synthetic fallback series.")
        inst = _fallback_series()
        used_mirror = False
    df = _simulate_us_var()

    # Align the instrument to the simulated VAR's date index.
    z = inst.quarterly.reindex(df.index, fill_value=0.0)
    n = len(df)
    Y = df.values
    return {"inst": inst, "df": df, "z": z, "Y": Y,
            "used_mirror": used_mirror}


def main() -> None:
    out = run_demo()
    inst = out["inst"]
    print("Ramey (2011) defense-news replication demo")
    print(f"  Loaded {len(inst.events)} narrative events "
          f"({'mirror' if out['used_mirror'] else 'fallback'}).")
    print(f"  Quarterly span: "
          f"{inst.quarterly.index.min().date()} to "
          f"{inst.quarterly.index.max().date()}")
    nz = inst.quarterly[inst.quarterly != 0]
    print(f"  Non-zero quarters: {len(nz)}")
    print()
    print("First 5 non-zero events:")
    print(nz.head().to_string())
    print()

    # Run proxy-SVAR on the simulated three-variable system.
    z = out["z"].values
    if (z != 0).sum() < 5:
        print("[skip] not enough non-zero observations for proxy SVAR.")
        return
    print("Running proxy SVAR on the 3-variable simulated VAR with this "
          "narrative instrument...")
    res = inst.to_proxy_svar(out["Y"], p=2, horizon=20,
                              z=z, n_boot=200, ci=0.9, seed=0)
    irf_point = res.irf_point
    print(f"  Olea-Pflueger F: {res.first_stage_F:.2f}")
    # proxy_svar returns shape (H+1, n, n): [h, response_var, shock_var]
    print(f"  IRF tensor shape: {irf_point.shape}  (H+1, n, n)")
    H = irf_point.shape[0] - 1
    print()
    print("Spending-multiplier-style picture: response of GDP (var 1) "
          "to the identified narrative shock (column 0):")
    for h in range(0, H + 1, 4):
        print(f"   h={h:>2d}q : {irf_point[h, 1, 0]:+.4f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        lo, hi = res.irf_lower, res.irf_upper
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
        names = list(out["df"].columns)
        h_arr = np.arange(H + 1)
        for i, ax in enumerate(axes):
            ax.plot(h_arr, irf_point[:, i, 0], color="0.0", lw=1.0)
            if lo is not None and hi is not None:
                ax.fill_between(h_arr, lo[:, i, 0], hi[:, i, 0],
                                  color="0.0", alpha=0.15)
            ax.axhline(0, color="0.5", lw=0.4)
            ax.set_title(names[i])
            ax.set_xlabel("h")
        fig.suptitle("Ramey-style defense-news shock IRF (proxy SVAR)")
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_ramey_2011.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'narrative_ramey_2011.png'}")


if __name__ == "__main__":
    main()
