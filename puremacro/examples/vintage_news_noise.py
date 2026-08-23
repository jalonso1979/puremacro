"""Are GDP revisions news or noise? A cross-country answer.

The Mankiw-Shapiro (1986) test is usually taught as a fact about the
United States, because the United States is where the vintages are easy
to get. With :func:`puremacro.fetch.vintage_panel` the same test runs
across 40-odd economies in one call, which turns "GDP revisions are
noise, not news" from an assertion into a stylised fact a class can
check.

Run::

    python -m puremacro.examples.vintage_news_noise
    python -m puremacro.examples.vintage_news_noise --countries USA DEU ESP MEX

Needs network on first run; responses are cached on disk afterwards.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from puremacro.fetch import vintage_panel


#: A spread of economies with deep archives, small enough to fetch in
#: one or two batched requests.
DEFAULT_COUNTRIES = [
    "USA", "GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT",
    "CAN", "JPN", "AUS", "MEX", "KOR", "NOR", "SWE",
]


def run(countries=None, *, hac_lags="auto", min_obs=20) -> pd.DataFrame:
    """Fetch the panel and return the per-country test table."""
    panel = vintage_panel(countries or DEFAULT_COUNTRIES, series="B1GQ")
    if panel.is_empty():
        raise RuntimeError(
            "vintage_panel returned nothing. Check network access; "
            f"reasons: {panel.metadata.get('failed')}"
        )
    return panel, panel.news_or_noise_panel(hac_lags=hac_lags,
                                            min_obs=min_obs)


def _plot(res: pd.DataFrame, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = res[res["ok"]].sort_values("beta_on_preliminary")
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(ok))))
    y = np.arange(len(ok))
    ax.errorbar(
        ok["beta_on_preliminary"], y,
        xerr=1.96 * ok["se_beta_on_preliminary"],
        fmt="o", markersize=4, capsize=2.5, linewidth=1,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1,
               label=r"$\beta=0$: news (revision unforecastable)")
    ax.set_yticks(y)
    ax.set_yticklabels(ok["country"])
    ax.set_xlabel(
        r"$\beta$ from $y_f - y_p = \alpha + \beta\, y_p + \varepsilon$"
        "\n(quarterly real GDP growth; bars are 95% HAC intervals)"
    )
    ax.set_title("News or noise? GDP revisions across countries")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"figure written to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--hac-lags", default="auto")
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--plot", default=None,
                    help="path for the cross-country figure (PNG)")
    args = ap.parse_args()

    panel, res = run(args.countries, hac_lags=args.hac_lags,
                     min_obs=args.min_obs)

    print("Providers used:", panel.metadata.get("provider_used"))
    print("Vintage-date semantics:")
    for prov, meaning in panel.vintage_semantics().items():
        print(f"  {prov}: {meaning}")
    print()

    cols = ["country", "n_obs", "beta_on_preliminary",
            "se_beta_on_preliminary", "p_beta_on_preliminary",
            "beta_on_final", "p_beta_on_final", "noise_share", "verdict"]
    ok = res[res["ok"]]
    print(ok[cols].round(4).to_string(index=False))

    skipped = res[~res["ok"]]
    if not skipped.empty:
        print("\nNot estimated:")
        print(skipped[["country", "n_obs", "note"]].to_string(index=False))

    if not ok.empty:
        print(f"\nmedian beta on preliminary: "
              f"{ok['beta_on_preliminary'].median():.4f}")
        print(f"share with beta < 0: "
              f"{(ok['beta_on_preliminary'] < 0).mean():.2f}")
        print("\nverdicts:")
        print(ok["verdict"].value_counts().to_string())
        print(
            "\nRead the pair, not one coefficient: beta on the preliminary "
            "estimate tests news, beta on the final tests noise. Under pure "
            "noise the first lies strictly inside (-1, 0) — its size is the "
            "noise share, not the verdict."
        )

    if args.plot:
        _plot(res, args.plot)


if __name__ == "__main__":
    main()
