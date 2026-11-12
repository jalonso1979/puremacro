"""Permanent regression tests for the reusable own-data fiscal policy brief workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

import puremacro
from puremacro.examples import first_brief_fiscal as fb

DATA_DIR = Path(puremacro.__file__).resolve().parent / "replication" / "data"


def test_parse_quarterly_index_formats_and_rejections() -> None:
    """Quarterly parser accepts EN/ES/ISO dates and rejects gaps, duplicates, or bad tokens."""
    en_idx = fb.parse_quarterly_index(["1999Q1", "1999Q2", "1999Q3", "1999Q4"])
    es_idx = fb.parse_quarterly_index(["1999T1", "1999T2", "1999T3", "1999T4"])
    iso_idx = fb.parse_quarterly_index(["1999-01-01", "1999-04-01", "1999-07-01", "1999-10-01"])
    p_idx = fb.parse_quarterly_index(pd.period_range("1999Q1", periods=4, freq="Q"))

    assert list(en_idx) == list(es_idx) == list(iso_idx) == list(p_idx)

    # Duplicate quarter rejected
    with pytest.raises(ValueError, match="duplicate quarterly dates"):
        fb.parse_quarterly_index(["2000Q1", "2000Q2", "2000Q2", "2000Q3"])

    # Gapped quarter rejected
    with pytest.raises(ValueError, match="gap|regular"):
        fb.parse_quarterly_index(["2000Q1", "2000Q2", "2000Q4"])

    # Malformed token rejected
    with pytest.raises(ValueError, match="could not parse quarterly date"):
        fb.parse_quarterly_index(["2000Q1", "not-a-quarter"])


def test_refuses_implicit_identification_and_ambiguous_shock_config() -> None:
    """Identification is never guessed from column names; shock_mode must be explicit."""
    # Even if a frame has 'news' or 'shock', omitting instrument under external_instrument fails
    with pytest.raises(ValueError, match="requires an explicit instrument column"):
        fb.FiscalBriefConfig(outcome="gdp", policy="gov", shock_mode="external_instrument")

    # Passing an external instrument while requesting blanchard_perotti is rejected as ambiguous
    with pytest.raises(ValueError, match="instrument"):
        fb.FiscalBriefConfig(
            outcome="gdp",
            policy="gov",
            instrument="news",
            shock_mode="blanchard_perotti",
        )


def test_second_dataset_qna40_deu_bundle_and_deflator_scaling(tmp_path: Path) -> None:
    """Bundled OECD QNA snapshot (DEU) runs end-to-end with zero library edits and portable manifest."""
    qna_csv = DATA_DIR / "qna40_panel.csv"
    out_en = tmp_path / "deu_en"
    out_es = tmp_path / "deu_es"

    cfg_real = fb.FiscalBriefConfig(
        filter_col="code",
        filter_value="DEU",
        date_col="date",
        outcome="gdp_real",
        policy="cons_gov_real",
        shock_mode="blanchard_perotti",
        scaling="hp_trend",
        state=None,
        headline_horizons=(3, 7),
        horizons=tuple(range(13)),
        citation="OECD QNA snapshot (DEU, 1995Q1-2026Q2)",
        units_en="Multipliers are euros of real GDP per euro of real government consumption over the horizon.",
        units_es="Los multiplicadores son euros de PIB real por euro de consumo público real en el horizonte.",
    )

    with pytest.warns(UserWarning, match="short sample"):
        study_en = fb.run_brief(out_en, lang="en", data=qna_csv, config=cfg_real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        study_es = fb.run_brief(out_es, lang="es", data=qna_csv, config=cfg_real)

    assert set(study_en) == {"record", "paths", "manifest"}
    assert set(study_es) == {"record", "paths", "manifest"}
    expected_names = {
        "headline.md",
        "headline.csv",
        "headline.tex",
        "headline.json",
        "paths.csv",
        "multipliers.png",
        "manifest.json",
    }
    assert {p.name for p in out_en.iterdir()} == expected_names
    assert {p.name for p in out_es.iterdir()} == expected_names

    manifest_raw = (out_en / "manifest.json").read_text(encoding="utf-8")
    headline_raw = (out_en / "headline.json").read_text(encoding="utf-8")
    # Zero host filesystem paths leaked into portable artifacts
    assert str(tmp_path) not in manifest_raw
    assert str(tmp_path) not in headline_raw
    assert str(DATA_DIR) not in manifest_raw
    assert str(DATA_DIR) not in headline_raw

    manifest = json.loads(manifest_raw)
    record = json.loads(headline_raw)

    assert record["data"]["file"] == "qna40_panel.csv"
    assert record["specification"]["shock_mode"] == "blanchard_perotti"
    assert "Blanchard-Perotti" in record["specification"]["identifying_assumption"]["en"]
    assert "Blanchard y Perotti" in record["specification"]["identifying_assumption"]["es"]
    assert record["specification"]["state"] is None
    assert record["full"]["sample_guard"]["short_sample"] is True
    assert any("short sample" in w for w in manifest["warnings"])

    # Every file hash in manifest.json matches actual bytes on disk
    for fname, info_dict in manifest["artifacts"].items():
        actual_sha = hashlib.sha256((out_en / fname).read_bytes()).hexdigest()
        assert actual_sha == info_dict["sha256"]

    md_en = (out_en / "headline.md").read_text(encoding="utf-8")
    md_es = (out_es / "headline.md").read_text(encoding="utf-8")
    assert "Short-sample caution" in md_en
    assert "Advertencia de muestra corta" in md_es
    assert "Blanchard-Perotti" in md_en

    # Nominal + deflator_col produces finite multipliers and records deflator provenance
    cfg_nom = fb.FiscalBriefConfig(
        filter_col="code",
        filter_value="DEU",
        date_col="date",
        outcome="gdp",
        policy="cons_gov",
        deflator_col="gdp_defl",
        shock_mode="blanchard_perotti",
        scaling="hp_trend",
        state=None,
        headline_horizons=(3, 7),
        horizons=tuple(range(13)),
        citation="OECD QNA snapshot (DEU nominal deflated by gdp_defl)",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rec_nom, paths_nom = fb.compute_brief(data=qna_csv, config=cfg_nom)
    assert rec_nom["specification"]["deflator_col"] == "gdp_defl"
    assert paths_nom["linear"].notna().all()


def test_second_dataset_narrative_fiscal_cli_and_spanish_quarters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI and --config JSON run state-dependent external-instrument brief on Spanish YYYYTn CSV."""
    fiscal = pd.read_csv(DATA_DIR / "tax14_us_fiscal.csv")
    shocks = pd.read_csv(DATA_DIR / "tax14_narrative_tax_shocks.csv")
    okun = pd.read_csv(DATA_DIR / "okun.csv")

    fiscal["q"] = [pd.Period(str(v), freq="Q") for v in fiscal["date"]]
    shocks["q"] = [pd.Period(str(v), freq="Q") for v in shocks["date"]]
    okun["q"] = pd.to_datetime(okun["date"]).dt.to_period("Q")
    okun_q = okun.groupby("q", as_index=False)["unrate"].mean()

    merged = (
        fiscal[["q", "gdpc1", "fedspend", "gdpdef"]]
        .merge(shocks[["q", "rr_exog"]], on="q", how="inner")
        .merge(okun_q[["q", "unrate"]], on="q", how="inner")
        .sort_values("q")
        .reset_index(drop=True)
    )
    # Format dates as Spanish trimestres (YYYYTn) and scale real spending
    merged["trimestre"] = [f"{p.year}T{p.quarter}" for p in merged["q"]]
    merged["pib_real"] = merged["gdpc1"]
    merged["gasto_real"] = merged["fedspend"] / merged["gdpdef"] * 100.0
    merged["choque_narrativo"] = merged["rr_exog"]
    merged["desempleo"] = merged["unrate"]

    csv_path = tmp_path / "datos_fiscales_es.csv"
    merged[["trimestre", "pib_real", "gasto_real", "choque_narrativo", "desempleo"]].to_csv(
        csv_path, index=False
    )

    out_cli = tmp_path / "salida_cli"
    rc = fb.main([
        "--data", str(csv_path),
        "--date-col", "trimestre",
        "--outcome", "pib_real",
        "--policy", "gasto_real",
        "--instrument", "choque_narrativo",
        "--scaling", "hp_trend",
        "--state", "desempleo",
        "--state-lag", "1",
        "--state-op", ">=",
        "--state-threshold", "5.8",
        "--headline-horizons", "3,7",
        "--max-horizon", "10",
        "--exclude", "1973T4:1975T1",
        "--citation", "Romer & Romer / Mertens & Ravn narrative fiscal dataset",
        "--lang", "es",
        "--output", str(out_cli),
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Muestra completa" in printed
    assert "1973Q4-1975Q1" in printed

    rec = json.loads((out_cli / "headline.json").read_text(encoding="utf-8"))
    assert rec["specification"]["shock_mode"] == "external_instrument"
    assert rec["specification"]["instrument"] == "choque_narrativo"
    assert rec["experiment"]["exclude"] == ["1973Q4", "1975Q1"]

    # Weak cells in headline.csv carry NaN for estimate/se/lo/hi and report AR bounds
    h_csv = pd.read_csv(out_cli / "headline.csv")
    weak_rows = h_csv[h_csv["weak"] == True]  # noqa: E712
    if not weak_rows.empty:
        assert weak_rows["estimate"].isna().all()
        assert weak_rows["lo"].isna().all()
        assert weak_rows["hi"].isna().all()

    # Also verify --config JSON path execution
    cfg_json = tmp_path / "brief_config.json"
    cfg_json.write_text(
        json.dumps({
            "data": str(csv_path),
            "date_col": "trimestre",
            "outcome": "pib_real",
            "policy": "gasto_real",
            "instrument": "choque_narrativo",
            "scaling": "hp_trend",
            "state": None,
            "headline_horizons": [3, 7],
            "horizons": list(range(9)),
            "citation": "JSON config test",
        }),
        encoding="utf-8",
    )
    out_json = tmp_path / "salida_json"
    rc_json = fb.main(["--config", str(cfg_json), "--output", str(out_json)])
    assert rc_json == 0
    assert (out_json / "manifest.json").exists()


def test_poly_trend_overfitting_warning_and_potential_col_equivalence() -> None:
    """High-degree polynomial trend on <200 quarters warns; potential_col matches pre_scaled."""
    rng = np.random.default_rng(20261112)
    n = 110
    dates = pd.period_range("1995Q1", periods=n, freq="Q")
    pot = np.exp(np.linspace(7.0, 7.6, n))
    shock = rng.normal(0.0, 0.01, size=n)
    g_raw = pot * (0.20 + 0.8 * shock + rng.normal(0.0, 0.003, size=n))
    y_raw = pot * (1.00 + 1.1 * shock + rng.normal(0.0, 0.005, size=n))
    u = 5.5 + np.sin(np.linspace(0.0, 6.0 * np.pi, n)) * 1.5

    df_raw = pd.DataFrame({
        "date": [str(p) for p in dates],
        "gdp_real": y_raw,
        "gov_real": g_raw,
        "shock_raw": shock * np.roll(pot, 1),
        "pot": pot,
        "urate": u,
    })
    df_raw.loc[0, "shock_raw"] = np.nan

    # Degree-6 polynomial trend on 110 quarters emits overfitting warning
    cfg_poly6 = fb.FiscalBriefConfig(
        outcome="gdp_real",
        policy="gov_real",
        instrument="shock_raw",
        scaling="poly_trend",
        poly_degree=6,
        state=None,
        headline_horizons=(2, 4),
        horizons=tuple(range(6)),
    )
    with pytest.warns(UserWarning, match="poly_trend degree=6"):
        fb.prepare_fiscal_data(df_raw, cfg_poly6)

    # potential_col vs pre_scaled equivalence
    cfg_pot = fb.FiscalBriefConfig(
        outcome="gdp_real",
        policy="gov_real",
        instrument="shock_raw",
        potential_col="pot",
        scaling="potential_col",
        state="urate",
        state_threshold=6.4,
        state_lag=1,
        headline_horizons=(2, 4),
        horizons=tuple(range(6)),
    )
    df_pre = pd.DataFrame({
        "date": [str(p) for p in dates],
        "y_scaled": y_raw / pot,
        "g_scaled": g_raw / pot,
        "z_scaled": df_raw["shock_raw"].to_numpy(),
        "urate": u,
    })
    cfg_pre = fb.FiscalBriefConfig(
        outcome="y_scaled",
        policy="g_scaled",
        instrument="z_scaled",
        scaling="pre_scaled",
        state="urate",
        state_threshold=6.4,
        state_lag=1,
        headline_horizons=(2, 4),
        horizons=tuple(range(6)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rec_pot, paths_pot = fb.compute_brief(data=df_raw, config=cfg_pot)
        rec_pre, paths_pre = fb.compute_brief(data=df_pre, config=cfg_pre)

    np.testing.assert_allclose(paths_pot["linear"], paths_pre["linear"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(paths_pot["slack"], paths_pre["slack"], rtol=1e-12, atol=1e-12)
    assert rec_pot["full"]["sample_guard"]["short_sample"] is True
    assert rec_pot["full"]["sample_guard"]["short_regime"] is True

    # Extreme state threshold with almost zero slack quarters fails fast with actionable message
    cfg_bad_state = fb.FiscalBriefConfig(
        outcome="y_scaled",
        policy="g_scaled",
        instrument="z_scaled",
        scaling="pre_scaled",
        state="urate",
        state_threshold=99.0,
        headline_horizons=(2, 4),
        horizons=tuple(range(6)),
    )
    with pytest.raises(ValueError, match="too few usable quarters by regime"):
        fb.prepare_fiscal_data(df_pre, cfg_bad_state)
