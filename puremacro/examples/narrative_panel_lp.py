"""Cross-country fiscal-multiplier panel local projection using the
homogeneous narrative-IV panel as the shock series and the wedges-panel
RGDP as the outcome.

Uses:
    uncertainty_examples/puremacro/puremacro/examples/output/
        narrative_iv_panel_quarterly.csv  (date × country signed magnitudes)
    wedges_panel_BGP_clean_nu2.csv         (RGDP, RCON, RGCF, …)

The exercise: for each panel country with overlapping coverage,
estimate the LP

    100 * Δ_h log(RGDP_{c,t+h}) = α_c + τ_t + β_h * z_{c,t}
                                  + Σ controls  + ε

over horizons h = 0..16 quarters, with country & time fixed effects
and Driscoll-Kraay HAC. The narrative IV ``z`` is the signed quarterly
magnitude (% of GDP). β_h is the (cumulative) fiscal multiplier:
percent change in real GDP per percent of GDP of expansionary
narrative shock.

Run:
    python -m puremacro.examples.narrative_panel_lp
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAV_ROOT = HERE.parents[3]
NAR  = HERE / "output" / "narrative_iv_panel_quarterly.csv"
WEDGES = MAV_ROOT / "wedges_panel_BGP_clean_nu2.csv"


def _to_quarter(s):
    s = str(s)
    if "-Q" in s and len(s) == 7:
        return pd.Period(s, freq="Q").to_timestamp()
    return pd.Timestamp(s)


def _wedges_to_long(w: pd.DataFrame) -> pd.DataFrame:
    if "qdate" in w.columns:
        w = w.copy()
        # qdate may already be parseable, but be safe with both formats.
        try:
            w["t"] = pd.to_datetime(w["qdate"], errors="coerce")
        except Exception:
            w["t"] = w["qdate"].apply(_to_quarter)
    elif "date" in w.columns:
        w["t"] = pd.to_datetime(w["date"])
    return w


def main() -> None:
    for _path in (NAR, WEDGES):
        if not _path.exists():
            raise FileNotFoundError(f"Required input panel not found: {_path}")
    nar = pd.read_csv(NAR, parse_dates=["date"])
    wed = _wedges_to_long(pd.read_csv(WEDGES))

    # Wide narrative panel → long.
    z_long = (
        nar.melt(id_vars="date", var_name="code", value_name="z")
            .rename(columns={"date": "t"})
    )
    z_long["t"] = pd.to_datetime(z_long["t"])

    # Merge wedges + narrative on (code, t).
    df = wed.merge(z_long, on=["code", "t"], how="left")
    df["z"] = df["z"].fillna(0.0)

    # Use log RGDP × 100 so β_h is in percentage points.
    if "RGDP" not in df.columns:
        raise SystemExit("Wedges panel does not have RGDP column.")
    df["log_y"] = 100.0 * np.log(df["RGDP"])
    df = df.sort_values(["code", "t"]).reset_index(drop=True)

    # Build the LP one horizon at a time. Country FE via demeaning,
    # time FE via demeaning, then OLS. SE: cluster-by-country.
    horizons = list(range(0, 17))
    rows = []
    for h in horizons:
        # Δ log y_{t+h} = log y_{t+h} - log y_{t-1}
        df["lhs"] = df.groupby("code")["log_y"].shift(-h) - \
                    df.groupby("code")["log_y"].shift(1)
        # Two-way demean: country + time.
        sub = df[["code", "t", "lhs", "z"]].dropna()
        if sub.empty:
            continue
        sub = sub.copy()
        sub["lhs_d"] = sub["lhs"] - sub.groupby("code")["lhs"].transform("mean")
        sub["lhs_d"] = sub["lhs_d"] - sub.groupby("t")["lhs_d"].transform("mean")
        sub["z_d"] = sub["z"] - sub.groupby("code")["z"].transform("mean")
        sub["z_d"] = sub["z_d"] - sub.groupby("t")["z_d"].transform("mean")
        if (sub["z_d"] ** 2).sum() == 0:
            continue
        beta = float((sub["z_d"] * sub["lhs_d"]).sum()
                     / (sub["z_d"] ** 2).sum())
        # Cluster-robust SE by country (one-way).
        sub["resid"] = sub["lhs_d"] - beta * sub["z_d"]
        # CR1 sandwich
        s = sub.groupby("code").apply(
            lambda g: (g["z_d"] * g["resid"]).sum()
        )
        meat = float((s ** 2).sum())
        bread = float((sub["z_d"] ** 2).sum())
        var_b = meat / max(bread ** 2, 1e-12)
        se = float(np.sqrt(var_b))
        rows.append({"h": h, "beta": beta, "se": se,
                       "lo": beta - 1.645 * se,
                       "hi": beta + 1.645 * se,
                       "n": int(len(sub))})

    irf = pd.DataFrame(rows).set_index("h")
    print("Cross-country fiscal-multiplier panel LP")
    print(f"  Countries with non-zero z: "
          f"{int((nar.iloc[:, 1:] != 0).any(axis=0).sum())}")
    print(f"  Common (country, quarter) cells with both z and RGDP: "
          f"{(df['z'] != 0).sum():,}")
    print()
    print("β_h   IRF: ∂ log RGDP × 100 / ∂ z (%-of-GDP narrative shock)")
    print(irf[["beta", "se", "lo", "hi", "n"]].round(4).to_string())

    out_dir = HERE / "output"
    if out_dir.is_dir() and len(rows) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(irf.index, irf["beta"], color="0.0", lw=1.0)
        ax.fill_between(irf.index, irf["lo"], irf["hi"],
                          color="0.0", alpha=0.15)
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("β_h (% Δ RGDP per % GDP shock)")
        ax.set_title(
            "Cross-country fiscal-multiplier LP — narrative IV "
            f"(N={int(irf['n'].max())} obs, two-way FE, cluster SE)",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "narrative_panel_lp.png",
                     dpi=120, bbox_inches="tight")
        print()
        print(f"Figure saved: {out_dir / 'narrative_panel_lp.png'}")


if __name__ == "__main__":
    main()
