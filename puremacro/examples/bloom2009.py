"""Bloom (2009)-style uncertainty-shock VAR replication on quarterly USA data.

Loads ``data/processed/panel_Q.parquet`` (a long-form OECD-style panel),
slices to the United States, and runs a 3-variable Cholesky-identified
VAR with ``[unc_q, log_gdp_real, log_gfcf_real]``. Ordering ``unc_q``
first satisfies the Bloom-style identifying assumption that uncertainty
shocks are predetermined relative to real activity within the quarter.

This is a *stylized-facts* replication on quarterly panel data, not a
literal port of Bloom's monthly US series. We expect: a negative
response of GDP in the year after the shock, a peak within 2-3 years,
and partial recovery by year 5.

Español
-------
Replicación del VAR de choque de incertidumbre al estilo de Bloom (2009) con datos
trimestrales de Estados Unidos.

Carga ``data/processed/panel_Q.parquet`` (un panel en formato largo estilo OCDE),
filtra a Estados Unidos y ejecuta un VAR de 3 variables identificado mediante la
descomposición de Cholesky con ``[unc_q, log_gdp_real, log_gfcf_real]``. Ordenar
``unc_q`` primero satisface el supuesto de identificación al estilo Bloom: los choques
de incertidumbre son predeterminados respecto a la actividad real dentro del trimestre.

Esta es una replicación de *hechos estilizados* sobre datos trimestrales en panel, no una
traducción literal de la serie mensual de Estados Unidos de Bloom. Se espera: una respuesta
negativa del PIB en el año siguiente al choque, un pico entre 2 y 3 años, y una recuperación
parcial hacia el año 5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import long_to_wide
from ..experiment import run_experiment


_DEFAULT_PANEL = "data/processed/panel_Q.parquet"


def run_bloom_replication(
    panel_path: str | Path = _DEFAULT_PANEL,
    country: str = "USA",
) -> dict:
    """Run the Bloom-2009 stylized replication and return the IRF dict."""
    panel_path = Path(panel_path)
    df_long = pd.read_parquet(panel_path)
    wide = long_to_wide(df_long)
    result = run_experiment(
        wide,
        outcome="log_gdp_real",
        proxy="unc_q",
        method="var_cholesky",
        country=country,
        horizons=range(0, 21),
        n_lags=4,
        controls=["log_gfcf_real"],
        n_boot=200,
        seed=0,
        alpha=0.10,
        normalize="1sd_pct",
    )
    return result


def _summarize(result: dict, outcome: str = "log_gdp_real") -> dict:
    """Return peak horizon, peak value, and 90% bands at peak."""
    point = result["irf_point"][outcome]
    lower = result["irf_lower"][outcome]
    upper = result["irf_upper"][outcome]
    abs_point = point.abs()
    peak_h = int(abs_point.idxmax())
    return {
        "peak_h": peak_h,
        "peak_value": float(point.iloc[peak_h]),
        "peak_lower": float(lower.iloc[peak_h]),
        "peak_upper": float(upper.iloc[peak_h]),
        "h0": float(point.iloc[0]),
        "h20": float(point.iloc[-1]),
    }


def main() -> None:
    """Execute the replication and print a short summary."""
    result = run_bloom_replication()
    summary = _summarize(result)
    print("Bloom (2009) stylized replication: USA, var_cholesky")
    print(f"  N obs (post-lag): {result['n_obs']}")
    print(f"  Peak horizon:     h = {summary['peak_h']}")
    print(f"  Peak response:    {summary['peak_value']:+.3f}% of GDP")
    print(f"  90% band at peak: [{summary['peak_lower']:+.3f}, "
          f"{summary['peak_upper']:+.3f}]")
    print(f"  Response at h=0:  {summary['h0']:+.3f}%")
    print(f"  Response at h=20: {summary['h20']:+.3f}%")

    # Optional figure rendering. Only writes if output/ already exists.
    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        from ..plot import plot_irf_single
        irf_df = pd.DataFrame({
            "h": result["irf_point"].index,
            "beta": result["irf_point"]["log_gdp_real"].values,
            "lo": result["irf_lower"]["log_gdp_real"].values,
            "hi": result["irf_upper"]["log_gdp_real"].values,
        })
        fig = plot_irf_single(
            irf_df,
            title="Uncertainty shock to GDP (USA, var_cholesky)",
            ylabel="% of GDP per 1-SD shock",
        )
        fig.savefig(out_dir / "bloom2009_irf.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'bloom2009_irf.png'}")


if __name__ == "__main__":
    main()
