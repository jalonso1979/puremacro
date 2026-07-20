"""Live FRED fetch + LP + markdown report — full pipeline.

Tries to download a real FRED series via the no-API-key CSV endpoint;
if the network is unavailable (offline iPad, restricted Pyodide
sandbox, FRED rate-limit, etc.) it gracefully falls back to a small
synthesized series so the demo still runs.

The point: any of the LP / VAR / GARCH / spectral / cointegration
modules can consume FRED-fetched series without leaving the notebook.

Run:
    python -m puremacro.examples.fetch_fred_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..fetch import fetch_fred
from ..lp.jorda import lp_hac
from ..reports import coef_table


_DEFAULT_SERIES = "GDPC1"   # quarterly real GDP
_DEFAULT_PROXY = "FEDFUNDS" # monthly Fed funds (will be aggregated to quarterly)


def _try_fetch(series_id: str, label: str) -> tuple[pd.Series, bool]:
    """Try the live fetch; on any failure return a simulated series + flag."""
    try:
        s = fetch_fred(series_id).dropna()
        if len(s) < 50:
            raise ValueError("too short")
        return s, True
    except Exception as e:
        # Synthesize a plausible quarterly series.
        rng = np.random.default_rng(hash(series_id) % 2**32)
        idx = pd.date_range("1980-01-01", periods=160, freq="QS")
        if "GDP" in series_id.upper():
            x = np.exp(np.cumsum(0.005 + rng.standard_normal(len(idx)) * 0.01))
            s = pd.Series(100 * x, index=idx, name=f"{series_id} (sim)")
        else:
            s = pd.Series(np.cumsum(rng.standard_normal(len(idx)) * 0.4),
                           index=idx, name=f"{series_id} (sim)")
        print(f"  [offline] {label} fetch failed ({type(e).__name__}); "
              f"using synthesized series.")
        return s, False


def run_demo() -> dict:
    gdp, gdp_live = _try_fetch(_DEFAULT_SERIES, "GDPC1")
    ff, ff_live = _try_fetch(_DEFAULT_PROXY, "FEDFUNDS")

    # Align to common quarterly grid.
    gdp_q = gdp.resample("QS").last()
    ff_q  = ff.resample("QS").mean()
    df = pd.concat([np.log(gdp_q).rename("log_gdp"),
                     ff_q.rename("ff")], axis=1).dropna()
    if len(df) < 30:
        # not enough data — use a small synthesized two-variable system.
        rng = np.random.default_rng(0)
        idx = pd.date_range("1980-01-01", periods=160, freq="QS")
        x = rng.standard_normal(len(idx)) * 0.5
        y = 0.5 * np.cumsum(x) + rng.standard_normal(len(idx)) * 0.4
        df = pd.DataFrame({"log_gdp": y, "ff": x}, index=idx)

    # Single-country LP: response of log GDP to a Fed Funds shock.
    irf = lp_hac(df, y="log_gdp", x="ff",
                  horizons=range(0, 17), n_lags=2, alpha=0.10)
    return {
        "gdp": gdp, "ff": ff,
        "gdp_live": gdp_live, "ff_live": ff_live,
        "df": df, "irf": irf,
    }


def main() -> None:
    out = run_demo()
    print(f"FRED fetch demo: GDPC1 live={out['gdp_live']}, "
          f"FEDFUNDS live={out['ff_live']}")
    print(f"  Aligned quarterly sample: {out['df'].index.min().date()} to "
          f"{out['df'].index.max().date()}, n = {len(out['df'])}")
    print()
    print("Newey-West LP: response of log_gdp to a unit shock in ff")
    print(coef_table(out["irf"]["beta"].values,
                      out["irf"]["se"].values,
                      names=[f"h={h}" for h in out["irf"]["h"].values],
                      include_t=False, include_p=False, include_ci=False,
                      digits=4))

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        irf = out["irf"]
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        ax.plot(irf["h"], irf["beta"], color="0.0", lw=1.0)
        ax.fill_between(irf["h"], irf["lo"], irf["hi"], color="0.0", alpha=0.15)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("∂ log GDP / ∂ ff_t")
        ax.set_title("LP IRF on FRED data" if out["gdp_live"] and out["ff_live"]
                      else "LP IRF on synthesized fallback data")
        fig.tight_layout()
        fig.savefig(out_dir / "fetch_fred_demo.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'fetch_fred_demo.png'}")


if __name__ == "__main__":
    main()
