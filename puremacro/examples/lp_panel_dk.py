"""Panel local projection of GDP on uncertainty with Driscoll-Kraay SE.

Loads ``data/processed/panel_Q.parquet`` (long-form OECD-style panel),
drops the EA20 / EU27 aggregates, and runs a two-way fixed-effects panel
LP of ``log_gdp_real`` on ``unc_q`` with Driscoll-Kraay HAC standard
errors (robust to heteroskedasticity, autocorrelation, and
cross-sectional dependence).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import long_to_wide
from ..lp.panel_dk import panel_lp_dk


_DEFAULT_PANEL = "data/processed/panel_Q.parquet"
_AGGREGATE_CODES = {"EA20", "EU27", "EU27_2020"}


def run_panel_dk(
    panel_path: str | Path = _DEFAULT_PANEL,
    horizons=range(0, 9),
    n_lags: int = 2,
) -> pd.DataFrame:
    """Run the panel LP with Driscoll-Kraay SE and return the IRF table."""
    panel_path = Path(panel_path)
    df_long = pd.read_parquet(panel_path)
    wide = long_to_wide(df_long)
    codes = wide.index.get_level_values("code")
    wide = wide.loc[~codes.isin(_AGGREGATE_CODES)]
    return panel_lp_dk(
        wide,
        y="log_gdp_real",
        x="unc_q",
        horizons=horizons,
        n_lags=n_lags,
        entity_level="code",
        time_level="date",
    )


def main() -> None:
    irf = run_panel_dk()
    print("Panel LP (two-way FE, Driscoll-Kraay SE): log_gdp_real on unc_q")
    print(irf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
