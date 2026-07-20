"""tools/build_bbui_district_panel.py — district-level BBUI panel builder.

Fetches Beige Book releases via ``iter_beige_book`` (which uses the
cached + rate-limited HTTP layer, cache root ``~/.cache/puremacro/http``)
and writes a tidy quarterly (district x period) uncertainty panel:

    date, district, value, n_docs, n_sections

Resumability: every page fetch goes through the on-disk HTTP cache, so
re-running the script — after a crash, or with a wider window — only
downloads pages not yet cached. The CSV itself is rewritten atomically
each run from the full record set.

Usage
-----
    python tools/build_bbui_district_panel.py --start 2022-01 --end 2025-12

Outputs (defaults, next to the package's other processed data):
    data/processed/bbui_district_panel.csv
    data/processed/bbui_district_panel.png   (12-district small multiples)

The index value is the raw LUI sentence-cooccurrence score by default
(fraction of sentences flagged labor-uncertain, quarterly mean across a
district's canonical sections) so that appending later windows does not
change already-built rows; pass ``--normalize zscore`` for a within-
district z-scored panel.
"""
from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from puremacro.narrative.sources import iter_beige_book  # noqa: E402
from puremacro.narrative.indices import bbui  # noqa: E402
from puremacro.narrative.indices._fed_districts import (  # noqa: E402
    DISTRICT_NAMES,
)


def _month_bounds(start: str, end: str) -> tuple[str, str]:
    """('2022-01', '2025-12') -> ('2022-01-01', '2025-12-31')."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    last = calendar.monthrange(y1, m1)[1]
    return f"{y0:04d}-{m0:02d}-01", f"{y1:04d}-{m1:02d}-{last:02d}"


def build_panel(start: str, end: str, *, section: str | None,
                normalize: str) -> tuple[pd.DataFrame, int]:
    since, until = _month_bounds(start, end)
    print(f"Fetching Beige Book releases {since} .. {until} "
          f"(cached HTTP; re-runs are incremental) ...", flush=True)
    records = list(iter_beige_book(since=since, until=until))
    print(f"  parsed {len(records)} (release, district, section) records",
          flush=True)
    if not records:
        raise SystemExit(
            "No Beige Book records parsed for the requested window. "
            "Check network access / the requested dates."
        )
    panel = bbui(records, level="district", output="tidy",
                 section=section, normalize=normalize)
    return panel, len(records)


def plot_panel(panel: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    districts = [d for d in DISTRICT_NAMES if d in set(panel["district"])]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    national = panel[panel["district"] == "National"]
    for ax, district in zip(axes.ravel(), districts):
        sub = panel[panel["district"] == district]
        if not national.empty:
            ax.plot(national["date"], national["value"], color="0.75",
                    lw=1.0, label="National")
        ax.plot(sub["date"], sub["value"], color="C0", lw=1.5,
                label=district)
        ax.set_title(district, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(districts):]:
        ax.set_visible(False)
    fig.suptitle("Beige Book Uncertainty Index (BBUI) by Fed district "
                 "— district (blue) vs National summary (grey)",
                 fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2022-01", metavar="YYYY-MM",
                    help="first release month to include (default 2022-01)")
    ap.add_argument("--end", default="2025-12", metavar="YYYY-MM",
                    help="last release month to include (default 2025-12)")
    ap.add_argument("--section", default=None,
                    help="restrict to one canonical section "
                         "(overall|labor|prices|consumer|manufacturing|"
                         "realestate); default: all sections")
    ap.add_argument("--normalize", default="raw",
                    choices=["raw", "zscore", "bbd_100"],
                    help="index normalization (default raw — stable "
                         "under later appends)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "processed"
                    / "bbui_district_panel.csv",
                    help="output CSV path")
    ap.add_argument("--fig", type=Path, default=None,
                    help="sanity-figure PNG path (default: CSV path "
                         "with .png suffix)")
    args = ap.parse_args(argv)

    panel, n_records = build_panel(args.start, args.end,
                                   section=args.section,
                                   normalize=args.normalize)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    panel.to_csv(tmp, index=False)
    tmp.replace(args.out)

    n_districts = panel["district"].nunique()
    n_periods = panel["date"].nunique()
    print(f"Wrote {args.out}")
    print(f"  panel: {n_districts} districts x {n_periods} quarters "
          f"({len(panel)} rows) from {n_records} parsed records")
    print(f"  coverage: {int((panel['n_docs'] > 0).sum())} district-quarters "
          f"with documents; mean n_docs "
          f"{panel.loc[panel['n_docs'] > 0, 'n_docs'].mean():.1f}")

    fig_path = args.fig or args.out.with_suffix(".png")
    plot_panel(panel, fig_path)
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
