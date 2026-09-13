"""Regenerate paper/scorecard.png: validation-gallery checks by kind of reference.

Run from the repository root:  python paper/make_scorecard_fig.py
Pyodide-safe (only puremacro.validation + matplotlib); no statsmodels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from puremacro.validation import scorecard  # noqa: E402

# The gallery's own `mechanism` labels, grouped into the three kinds of
# reference the paper describes. "package" cases compare against stored outputs
# of statsmodels, arch, linearmodels or esda; "scipy" cases against SciPy
# computed at run time; "published" against a printed table.
KIND = {
    "package": "External reference",
    "scipy": "External reference",
    "published": "External reference",
    "analytical": "Analytical result",
    "internal": "Internal consistency",
}
ORDER = ["External reference", "Analytical result", "Internal consistency"]
# Categorical slots 1-3 of the dataviz reference palette (light mode); these
# three validate on every pair, so identity survives colour-vision deficiency.
COLOR = {
    "External reference": "#2a78d6",
    "Analytical result": "#eb6834",
    "Internal consistency": "#1baf7a",
}
SUBSYSTEM = {
    "var": "VAR / SVAR",
    "spatial": "Spatial econometrics",
    "did": "Difference-in-differences",
    "inference": "HAC and weak-IV inference",
    "narrative": "Text-based indices",
    "garch": "GARCH volatility",
    "dynpanel": "Dynamic panels",
    "state_space": "State space / Kalman",
    "spectral": "Spectral analysis",
    "lp": "Local projections",
    "forecast": "Forecast evaluation",
    "dsge": "Linear RE / DSGE",
    "vfi": "Dynamic programming",
    "cointegration": "Cointegration",
    "unit_root": "Unit roots",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff")


def main() -> None:
    df = scorecard()
    unmapped = set(df["mechanism"]) - set(KIND)
    if unmapped:
        raise SystemExit(f"unmapped mechanism labels: {sorted(unmapped)}")
    df = df.assign(kind=df["mechanism"].map(KIND))
    tab = (df.pivot_table(index="subsystem", columns="kind", values="id",
                          aggfunc="count", fill_value=0)
             .reindex(columns=ORDER, fill_value=0))
    # Largest subsystem at the top; ties broken by name so the order is stable.
    rows = sorted(tab.index, key=lambda s: (int(tab.loc[s].sum()), s))
    tab = tab.loc[rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.patch.set_facecolor(SURFACE)
    left = [0] * len(tab)
    for kind in ORDER:
        vals = [int(v) for v in tab[kind]]
        ax.barh(range(len(tab)), vals, left=left, height=0.62, color=COLOR[kind],
                edgecolor=SURFACE, linewidth=1.0, label=f"{kind} ({sum(vals)})")
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + 0.25, i, str(total), va="center", ha="left",
                fontsize=8.5, color=INK_2)

    ax.set_yticks(range(len(tab)), [SUBSYSTEM.get(s, s) for s in tab.index],
                  fontsize=9, color=INK)
    ax.set_xlim(0, max(left) * 1.1)
    ax.set_xlabel(f"validation checks ({int(df['passed'].sum())} of {len(df)} pass)",
                  fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5,
                       handlelength=1.0, handleheight=1.0, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    dest = Path(__file__).resolve().parent / "scorecard.png"
    fig.savefig(dest, dpi=300, facecolor=SURFACE)
    counts = {k: int(tab[k].sum()) for k in ORDER}
    print(f"wrote {dest} ({len(df)} cases, {df['subsystem'].nunique()} subsystems, {counts})")


if __name__ == "__main__":
    main()
