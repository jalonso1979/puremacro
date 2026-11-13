"""Permanent regression and conformance tests for Delivery 4: Trade-to-Household Incidence Research Pilot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro
from puremacro.examples.first_brief_tariffs import (
    TradeBriefConfig,
    compute_brief,
    load_bundled,
    load_enigh_groups,
    load_groups,
    national_basket,
    run_brief,
    supported_sectors,
)


def test_trade_brief_config_validation() -> None:
    """TradeBriefConfig validates inputs and rejects invalid parameters."""
    cfg = TradeBriefConfig()
    assert cfg.scenario == "USA_tariff_on_CHN_all"
    assert cfg.importer == "USA"
    assert cfg.exporter == "CHN"
    assert cfg.sector == "all"
    assert cfg.tariff_rate == 0.10
    assert cfg.trade_elasticity == 4.0
    assert cfg.household_elasticity == 1.0
    assert cfg.use_enigh is True
    assert cfg.bilingual is True
    assert cfg.sensitivity_trade_elasticities == (2.0, 4.0, 6.0)
    assert cfg.sensitivity_household_elasticities == (0.5, 1.0, 1.5)

    # Negative tariff rate rejected
    with pytest.raises(ValueError, match="tariff_rate"):
        TradeBriefConfig(tariff_rate=-0.05)

    # Negative trade elasticity rejected
    with pytest.raises(ValueError, match="trade_elasticity"):
        TradeBriefConfig(trade_elasticity=-1.0)

    # Negative household elasticity rejected
    with pytest.raises(ValueError, match="household_elasticity"):
        TradeBriefConfig(household_elasticity=-0.5)

    # Empty or non-positive sensitivity trade elasticities rejected
    with pytest.raises(ValueError, match="sensitivity_trade_elasticities"):
        TradeBriefConfig(sensitivity_trade_elasticities=())
    with pytest.raises(ValueError, match="sensitivity_trade_elasticities"):
        TradeBriefConfig(sensitivity_trade_elasticities=(0.0, 2.0))

    # Empty or negative sensitivity household elasticities rejected
    with pytest.raises(ValueError, match="sensitivity_household_elasticities"):
        TradeBriefConfig(sensitivity_household_elasticities=())
    with pytest.raises(ValueError, match="sensitivity_household_elasticities"):
        TradeBriefConfig(sensitivity_household_elasticities=(-0.5, 1.0))

    # Non-existent data_path rejected
    with pytest.raises(FileNotFoundError, match="data_path"):
        TradeBriefConfig(data_path="non_existent_file.csv")

    # Scenario parsing
    custom_sc = TradeBriefConfig(scenario="DEU_tariff_on_USA_C")
    assert custom_sc.importer == "DEU"
    assert custom_sc.exporter == "USA"
    assert custom_sc.sector == "C"


def test_load_enigh_groups_loader() -> None:
    """load_enigh_groups loads 90,033-household empirical quintiles satisfying Engel's Law."""
    icio, calib = load_bundled()
    groups = load_enigh_groups(icio, "USA")

    # 5 income quintiles
    assert list(groups.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert "population_weight" in groups.columns
    assert "annual_spending" in groups.columns

    # Population weights sum to 1.0 (~0.20 per quintile)
    pop_wts = groups["population_weight"].to_numpy()
    np.testing.assert_allclose(pop_wts.sum(), 1.0, atol=1e-6)
    for w in pop_wts:
        assert 0.19 <= w <= 0.21

    # Annual spending rises monotonically
    spending = groups["annual_spending"].to_numpy()
    assert all(spending[i] < spending[i + 1] for i in range(len(spending) - 1))

    # Budget shares across all 20 sectors sum to 1.0 within 1e-6
    sector_cols = list(icio.sector_codes)
    row_sums = groups[sector_cols].sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    # Food / agriculture share satisfies Engel's Law: Q1 spends a larger fraction on food than Q5
    food_q1 = groups.loc["Q1", "A"] + groups.loc["Q1", "C"]
    food_q5 = groups.loc["Q5", "A"] + groups.loc["Q5", "C"]
    assert food_q1 > food_q5
    assert food_q1 > 0.40  # poorer quintile spends >40% on food/goods
    assert food_q5 < 0.35  # richer quintile spends <35% on food/goods

    # Metadata attributes
    assert groups.attrs["is_synthetic"] is False
    assert "ENIGH 2022" in groups.attrs["label"]
    assert groups.attrs["file"] == "enigh2022_household_policy.npz"
    assert len(groups.attrs["sha256"]) == 64


def test_compute_brief_bundled_enigh() -> None:
    """compute_brief executes end-to-end on bundled table + ENIGH and returns (record, paths)."""
    cfg = TradeBriefConfig(
        use_enigh=True,
        trade_elasticity=4.0,
        household_elasticity=1.0,
        sensitivity_trade_elasticities=(2.0, 4.0),
        sensitivity_household_elasticities=(0.5, 1.0),
    )
    record, paths = compute_brief(cfg)

    assert isinstance(record, dict)
    assert record["schema"] == 1
    assert record["scenario"]["importer"] == "USA"
    assert record["scenario"]["exporter"] == "CHN"
    assert record["households"]["is_synthetic"] is False
    assert len(record["households"]["rows"]) == 5

    assert isinstance(paths, pd.DataFrame)
    # 2 trade elasticities * 2 household elasticities * 6 rows (5 quintiles + all) = 24 rows
    assert len(paths) == 24
    assert set(paths["trade_elasticity"].unique()) == {2.0, 4.0}
    assert set(paths["household_elasticity"].unique()) == {0.5, 1.0}
    assert set(paths["group"].unique()) == {"Q1", "Q2", "Q3", "Q4", "Q5", "all"}
    assert np.isfinite(paths["ev_pct_proportional"]).all()
    assert np.isfinite(paths["national_ev_pct"]).all()


def test_run_brief_bundled_enigh_outputs_and_portable_manifest(tmp_path: Path) -> None:
    """run_brief writes all required Delivery 4 artifacts with zero host paths and valid SHA-256 hashes."""
    out_dir = tmp_path / "enigh_brief"
    cfg = TradeBriefConfig(
        use_enigh=True,
        trade_elasticity=4.0,
        household_elasticity=1.0,
        sensitivity_trade_elasticities=(2.0, 4.0, 6.0),
        sensitivity_household_elasticities=(0.5, 1.0, 1.5),
        bilingual=True,
    )
    study = run_brief(out_dir, config=cfg, lang="en")

    assert set(study.keys()) >= {"record", "paths", "manifest", "solution", "households"}

    # Expected artifact files
    expected_files = {
        "headline.md",
        "headline.csv",
        "headline.tex",
        "headline.json",
        "headline_es.md",
        "paths.csv",
        "first_brief.png",
        "manifest.json",
        "national.md",
        "national.csv",
        "national.tex",
        "households.md",
        "households.csv",
        "households.tex",
    }
    actual_files = {p.name for p in out_dir.iterdir()}
    assert actual_files == expected_files

    manifest_raw = (out_dir / "manifest.json").read_text(encoding="utf-8")
    headline_raw = (out_dir / "headline.json").read_text(encoding="utf-8")

    # Zero host filesystem paths leaked into artifacts
    assert str(tmp_path) not in manifest_raw
    assert str(tmp_path) not in headline_raw

    manifest = json.loads(manifest_raw)
    assert manifest["groups"]["is_synthetic"] is False
    assert manifest["groups"]["file"] == "enigh2022_household_policy.npz"
    assert manifest["table"]["file"] == "icio_oecd2023_2019_isic_g20.npz"

    # Check that every artifact SHA-256 in manifest matches actual bytes on disk
    for fname, info in manifest["artifacts"].items():
        actual_sha = hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
        assert actual_sha == info["sha256"], f"SHA mismatch on {fname}"
        assert (out_dir / fname).stat().st_size == info["bytes"]

    # Verify sensitivity bands are complete in paths.csv: 3x3 grid
    paths_df = pd.read_csv(out_dir / "paths.csv")
    assert set(paths_df["trade_elasticity"].unique()) == {2.0, 4.0, 6.0}
    assert set(paths_df["household_elasticity"].unique()) == {0.5, 1.0, 1.5}
    # 9 parameter combinations * 6 groups (Q1..Q5 + all) = 54 rows
    assert len(paths_df) == 54

    # Verify headline.md contents
    md_en = (out_dir / "headline.md").read_text(encoding="utf-8")
    assert "First policy brief: 10% tariff by USA on all modeled sectors" in md_en
    assert "Purchaser prices" in md_en
    assert "Factor income" in md_en
    assert "Fiscal transfers" in md_en
    assert "Equivalent variation" in md_en
    assert "Household groups: ENIGH 2022 income quintiles" in md_en
    assert "Sensitivity to trade and household elasticities" in md_en


def test_run_brief_user_supplied_csv_microdata(tmp_path: Path) -> None:
    """run_brief supports custom user microdata CSV and generates portable artifacts."""
    # Create synthetic microdata CSV
    rng = np.random.default_rng(42)
    n = 200
    weights = rng.uniform(10.0, 100.0, size=n)
    incomes = rng.lognormal(mean=10.5, sigma=0.6, size=n)
    spendings = incomes * rng.uniform(0.6, 0.9, size=n)
    foods = spendings * np.clip(0.6 - 0.2 * (incomes - np.min(incomes)) / (np.max(incomes) - np.min(incomes)), 0.15, 0.65)
    hh_sizes = rng.integers(1, 6, size=n)

    micro_df = pd.DataFrame({
        "weight": weights,
        "spending": spendings,
        "food": foods,
        "income": incomes,
        "hh_size": hh_sizes,
    })
    csv_file = tmp_path / "custom_microdata.csv"
    micro_df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "custom_run"
    cfg = TradeBriefConfig(
        data_path=csv_file,
        use_enigh=False,
        trade_elasticity=2.0,
        household_elasticity=1.0,
        sensitivity_trade_elasticities=(2.0, 4.0),
        sensitivity_household_elasticities=(0.5, 1.0),
    )
    study = run_brief(out_dir, config=cfg, lang="en")

    assert out_dir.exists()
    assert (out_dir / "paths.csv").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["groups"]["is_synthetic"] is False
    assert manifest["groups"]["file"] == "custom_microdata.csv"
    assert str(tmp_path) not in (out_dir / "manifest.json").read_text(encoding="utf-8")

    # Verify file checksums
    for fname, info in manifest["artifacts"].items():
        assert hashlib.sha256((out_dir / fname).read_bytes()).hexdigest() == info["sha256"]


def test_run_brief_user_supplied_group_shares_csv(tmp_path: Path) -> None:
    """run_brief supports custom user group shares CSV."""
    icio, calib = load_bundled()
    sectors = list(icio.sector_codes)
    basket = national_basket(icio, "USA")

    row1 = {"group": "bottom_half", "population_weight": 0.5, "annual_spending": 45_000.0,
            **{s: float(basket[s]) for s in sectors}}
    row2 = {"group": "top_half", "population_weight": 0.5, "annual_spending": 90_000.0,
            **{s: float(basket[s]) for s in sectors}}
    group_df = pd.DataFrame([row1, row2])
    csv_file = tmp_path / "custom_groups.csv"
    group_df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "groups_run"
    cfg = TradeBriefConfig(
        data_path=csv_file,
        use_enigh=False,
        sensitivity_trade_elasticities=(2.0,),
        sensitivity_household_elasticities=(1.0,),
    )
    study = run_brief(out_dir, config=cfg, lang="en")

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["groups"]["is_synthetic"] is False
    assert manifest["groups"]["file"] == "custom_groups.csv"
    assert (out_dir / "paths.csv").exists()


def test_bilingual_spanish_and_english_generation(tmp_path: Path) -> None:
    """run_brief generates valid bilingual outputs in Spanish and English."""
    out_dir = tmp_path / "spanish_run"
    cfg = TradeBriefConfig(
        use_enigh=True,
        bilingual=True,
        sensitivity_trade_elasticities=(2.0,),
        sensitivity_household_elasticities=(1.0,),
    )
    study = run_brief(out_dir, config=cfg, lang="es")

    md_es = (out_dir / "headline.md").read_text(encoding="utf-8")
    assert "Primer informe de política" in md_es
    assert "Precios de comprador" in md_es
    assert "Ingreso de los factores" in md_es
    assert "Variación equivalente" in md_es
    assert "Incidencia en los hogares" in md_es
    assert "Sensibilidad a elasticidades comerciales y de hogares" in md_es

    # Bilingual companion file in English exists
    md_en = (out_dir / "headline_en.md").read_text(encoding="utf-8")
    assert "First policy brief" in md_en
    assert "Purchaser prices" in md_en

    # Zero host paths
    assert str(tmp_path) not in (out_dir / "manifest.json").read_text(encoding="utf-8")


def test_load_groups_adversarial_microdata_validation(tmp_path: Path) -> None:
    """load_groups rejects invalid, negative, or degenerate user microdata CSVs with informative ValueErrors."""
    icio, _ = load_bundled()

    # Case 1: All-zero or negative spending records
    p_zero_spend = tmp_path / "zero_spend.csv"
    pd.DataFrame({
        "weight": [10.0, 20.0, 30.0],
        "spending": [0.0, -100.0, 0.0],
    }).to_csv(p_zero_spend, index=False)
    with pytest.raises(ValueError, match="no positive expenditure records"):
        load_groups(p_zero_spend, icio)

    # Case 2: Negative or non-finite sampling weights
    p_neg_wt = tmp_path / "neg_wt.csv"
    pd.DataFrame({
        "weight": [-5.0, 10.0, 20.0, 30.0, 40.0],
        "spending": [100.0, 200.0, 300.0, 400.0, 500.0],
    }).to_csv(p_neg_wt, index=False)
    with pytest.raises(ValueError, match="must contain finite, strictly positive values"):
        load_groups(p_neg_wt, icio)

    p_zero_wt = tmp_path / "zero_wt.csv"
    pd.DataFrame({
        "weight": [0.0, 10.0, 20.0, 30.0, 40.0],
        "spending": [100.0, 200.0, 300.0, 400.0, 500.0],
    }).to_csv(p_zero_wt, index=False)
    with pytest.raises(ValueError, match="must contain finite, strictly positive values"):
        load_groups(p_zero_wt, icio)

    p_nan_wt = tmp_path / "nan_wt.csv"
    pd.DataFrame({
        "weight": [np.nan, 10.0, 20.0, 30.0, 40.0],
        "spending": [100.0, 200.0, 300.0, 400.0, 500.0],
    }).to_csv(p_nan_wt, index=False)
    with pytest.raises(ValueError, match="must contain finite, strictly positive values"):
        load_groups(p_nan_wt, icio)

    # Case 3: Fewer than 5 observations
    p_few_rows = tmp_path / "few_rows.csv"
    pd.DataFrame({
        "weight": [10.0, 20.0, 30.0, 40.0],
        "spending": [100.0, 200.0, 300.0, 400.0],
    }).to_csv(p_few_rows, index=False)
    with pytest.raises(ValueError, match="at least 5 observations"):
        load_groups(p_few_rows, icio)

    # Case 4: Degenerate microdata where quintiles cannot be constructed
    p_degenerate = tmp_path / "degenerate_quintiles.csv"
    pd.DataFrame({
        "weight": [10.0, 10.0, 10.0, 10.0, 10.0],
        "spending": [100.0, 100.0, 100.0, 100.0, 100.0],
        "income": [500.0, 500.0, 500.0, 500.0, 500.0],
    }).to_csv(p_degenerate, index=False)
    with pytest.raises(ValueError, match="has no valid observations or zero weight"):
        load_groups(p_degenerate, icio)

