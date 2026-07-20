"""Regenerate puremacro/paper/scorecard.png — validation-gallery coverage.

Run from the package dir:  python paper/make_scorecard_fig.py
Pyodide-safe (only puremacro.validation + matplotlib); no statsmodels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from puremacro.validation import scorecard  # noqa: E402


def main() -> None:
    df = scorecard()
    agg = df.groupby("subsystem").size().sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(agg.index, agg.values, color="0.30")
    for i, v in enumerate(agg.values):
        ax.text(v + 0.08, i, str(int(v)), va="center", fontsize=9)
    ax.set_xlabel("validation cases (all passing)")
    ax.set_title(
        f"puremacro validation gallery: {len(df)} cases across {df['subsystem'].nunique()} subsystems"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    dest = Path(__file__).resolve().parent / "scorecard.png"
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest} ({len(df)} cases, {df['subsystem'].nunique()} subsystems)")


if __name__ == "__main__":
    main()
