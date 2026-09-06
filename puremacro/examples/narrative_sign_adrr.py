"""Narrative sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018) —
the October 1979 Volcker episode as an identification input.

AD-RR's flagship application: a monetary SVAR where, on top of
traditional sign restrictions (a contractionary monetary shock raises
the federal funds rate and lowers inflation and output), one imposes
*narrative* restrictions about a single famous date — the week after
October 6, 1979, when Volcker's Saturday-night FOMC announcement
tightened policy:

  (I)   the monetary shock in 1979Q4 was positive (contractionary), and
  (III) it was the *overwhelming* contributor to the unexpected rise in
        the federal funds rate that quarter.

The headline mechanism illustrated here: the narrative restrictions
sharply tighten the sign-identified set, because they discard rotations
whose "monetary shock" column cannot account for the Volcker episode.

Data: a **synthetic-but-calibrated** quarterly DGP (1965Q1-2007Q4,
T=172) — persistence and impact signs loosely match a standard
FFR / CPI-inflation / output-growth monetary VAR, and a +3.5 s.d.
monetary shock is planted in 1979Q4. Synthetic data keeps the demo
deterministic, offline, and fast (< 60 s); swap in FRED series via
``puremacro.fetch`` for a real-data version.

This is **not** a replication of the paper's numbers: AD-RR's
application uses Uhlig's (2005) six-variable monthly US dataset, which
this package cannot fetch offline, and reports its IRFs graphically.
``tests/test_replication_adrr2018.py`` therefore pins a regression
snapshot of this demo's own output, not published values.

The Type-I restriction enters as a ``NarrativeEvent`` (the package's
narrative-layer object; the adapter uses its announcement ``date`` and
``sign``), showcasing the narrative -> identification fusion.

Run:
    python -m puremacro.examples.narrative_sign_adrr
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..narrative import NarrativeEvent
from ..var.identify import NarrativeRestriction, narrative_sign_svar

VARIABLES = ["Fed funds rate", "Inflation", "Output growth"]
VOLCKER_DATE = "1979-10-06"


def _simulate(T: int = 172, seed: int = 12) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
    """Synthetic monetary VAR(1), quarterly 1965Q1-2007Q4.

    Shock 0 is the monetary shock: impact (+FFR, -inflation, -output).
    Shocks 1-2 are demand-ish and supply-ish completions with sign
    patterns that do not mimic the monetary column.
    """
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.80, 0.15, 0.05],   # FFR: persistent, leans against inflation
        [0.05, 0.60, 0.10],   # inflation
        [-0.10, 0.00, 0.30],  # output growth: hurt by lagged FFR
    ])
    B0 = np.array([
        [1.00, 0.20, 0.10],   # monetary +, demand +, supply +
        [-0.30, 0.80, -0.40], # monetary -, demand +, supply -
        [-0.50, 0.60, 0.90],  # monetary -, demand +, supply +
    ])
    dates = pd.period_range("1965Q1", periods=T, freq="Q").to_timestamp()
    t_volcker = int(np.where(dates == pd.Timestamp("1979-10-01"))[0][0])
    eps = rng.standard_normal((T, 3))
    eps[t_volcker, 0] = 3.5  # the planted Volcker tightening
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B0 @ eps[t]
    return Y, dates, t_volcker


def run_demo(H: int = 16, n_draws: int = 3000) -> dict:
    Y, dates, t_volcker = _simulate()

    # Traditional sign restrictions at h=0 and h=1 on the monetary
    # shock (column 0): +FFR, -inflation, -output.
    S = np.zeros((3, 3))
    S[:, 0] = [+1, -1, -1]
    sign_matrix = {0: S, 1: S}

    # The Volcker episode as a narrative-layer object -> Type I input.
    volcker = NarrativeEvent(
        date=VOLCKER_DATE, country="USA", magnitude=1.0, magnitude_unit="z",
        target="policy_rate", subtarget=None, sign=+1, confidence=1.0,
        source_text="FOMC, Saturday October 6, 1979: new operating "
                    "procedures; federal funds rate rises sharply.",
        source_url="https://www.federalreserve.gov/monetarypolicy/fomchistorical1979.htm",
        scoring_method="manual", kind="monetary",
    )
    narrative = [
        volcker,  # Type I: monetary shock positive in 1979Q4
        NarrativeRestriction(  # Type III: overwhelming driver of the FFR jump
            kind="hd_dominance", date=VOLCKER_DATE, shock=0, variable=0,
            dominance="overwhelming",
        ),
    ]

    plain = narrative_sign_svar(
        Y, p=1, horizon=H, sign_matrix=sign_matrix, restrictions=[],
        dates=dates, n_draws=n_draws, ci=0.68, seed=0,
    )
    narr = narrative_sign_svar(
        Y, p=1, horizon=H, sign_matrix=sign_matrix, restrictions=narrative,
        dates=dates, n_draws=n_draws, n_weight_sims=300, ci=0.68, seed=0,
    )
    return {"plain": plain, "narr": narr, "H": H, "t_volcker": t_volcker,
            "dates": dates}


def main() -> None:
    out = run_demo()
    plain, narr, H = out["plain"], out["narr"], out["H"]

    print("Narrative sign restrictions for SVARs — AD-RR (2018), AER 108(10)")
    print("Synthetic-but-calibrated monetary VAR; Volcker 1979Q4 episode planted.")
    print()
    print(narr.summary())
    print(f"  traditional-only acceptance : {plain.n_traditional_accepted}/{plain.n_draws}")
    print(f"  + narrative acceptance      : {narr.n_narrative_accepted}"
          f"/{narr.n_traditional_accepted} traditionally-accepted draws")
    print()
    print("Pointwise 68% band width for the monetary-shock column,")
    print("averaged over horizons 0..16 (the AD-RR headline: bands tighten):")
    print(f"    {'variable':<16} {'plain sign':>11} {'narrative':>11} {'shrink':>8}")
    for i, name in enumerate(VARIABLES):
        wp = float((plain.irf_upper - plain.irf_lower)[:, i, 0].mean())
        wn = float((narr.irf_upper - narr.irf_lower)[:, i, 0].mean())
        print(f"    {name:<16} {wp:>11.3f} {wn:>11.3f} {1 - wn / wp:>7.1%}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h_arr = np.arange(H + 1)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
        for i, (ax, name) in enumerate(zip(axes, VARIABLES)):
            ax.fill_between(h_arr, plain.irf_lower[:, i, 0], plain.irf_upper[:, i, 0],
                            color="0.85", label="plain sign (68%)")
            ax.fill_between(h_arr, narr.irf_lower[:, i, 0], narr.irf_upper[:, i, 0],
                            color="tab:blue", alpha=0.35,
                            label="+ narrative (68%)")
            ax.plot(h_arr, narr.irf_median[:, i, 0], color="tab:blue", lw=1.2)
            ax.plot(h_arr, plain.irf_median[:, i, 0], color="0.3", lw=1.0, ls="--")
            ax.axhline(0.0, color="0.5", lw=0.4)
            ax.set_title(name)
            ax.set_xlabel("Horizon (quarters)")
            if i == 0:
                ax.set_ylabel("Response to monetary shock")
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Monetary shock IRFs: plain sign vs AD-RR narrative sign restrictions")
        fig.tight_layout()
        fig_path = out_dir / "narrative_sign_adrr.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"\n  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
