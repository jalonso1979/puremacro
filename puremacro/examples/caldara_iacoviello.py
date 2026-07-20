"""Caldara-Iacoviello (2022)-style geopolitical-risk shock VAR on USA quarterly data.

Loads ``data/processed/panel_Q.parquet``, slices to the United States, and
runs a 3-variable Cholesky-identified VAR with ``[gpr_q, log_gdp_real,
short_rate]``. Ordering ``gpr_q`` first imposes that geopolitical-risk
shocks are predetermined relative to real activity and the short rate.

This is a stylised-facts replication on quarterly data, not a literal
port of Caldara-Iacoviello's monthly VAR. Expected output: a negative
GDP response peaking 2-4 quarters out and a tightening (or muted)
short-rate response.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import long_to_wide
from ..experiment import run_experiment


_DEFAULT_PANEL = "data/processed/panel_Q.parquet"


def run_caldara_iacoviello(
    panel_path: str | Path = _DEFAULT_PANEL,
    country: str = "USA",
) -> dict:
    """Run the Caldara-Iacoviello stylised replication and return the IRF dict."""
    panel_path = Path(panel_path)
    df_long = pd.read_parquet(panel_path)
    wide = long_to_wide(df_long)
    return run_experiment(
        wide,
        outcome="log_gdp_real",
        proxy="gpr_q",
        method="var_cholesky",
        country=country,
        horizons=range(0, 21),
        n_lags=4,
        controls=["short_rate"],
        n_boot=200,
        seed=0,
        alpha=0.10,
        normalize="1sd_pct",
    )


def main() -> None:
    result = run_caldara_iacoviello()
    point = result["irf_point"]["log_gdp_real"]
    abs_pt = point.abs()
    peak_h = int(abs_pt.idxmax())
    print("Caldara-Iacoviello (2022) stylised replication: USA, var_cholesky")
    print(f"  N obs (post-lag):  {result['n_obs']}")
    print(f"  Peak |response|:   {float(point.iloc[peak_h]):+.3f}% at h={peak_h}")
    print(f"  Response at h=0:   {float(point.iloc[0]):+.3f}%")
    print(f"  Response at h=20:  {float(point.iloc[-1]):+.3f}%")


if __name__ == "__main__":
    main()
