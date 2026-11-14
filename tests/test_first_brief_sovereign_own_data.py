"""Tests for puremacro.examples.sovereign_own_data_brief."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.examples import sovereign_own_data_brief as sb


def _custom_macro_df(
    *,
    frequency: str = "quarterly",
    n_obs: int = 64,
    with_gap: bool = False,
    with_dup: bool = False,
) -> pd.DataFrame:
    df = sb.demo_sovereign_dataframe(frequency=frequency, n_obs=max(n_obs, 20))  # type: ignore[arg-type]
    df = df.iloc[:n_obs].copy().reset_index(drop=True)
    if with_gap and len(df) > 4:
        df = df.drop(index=3).reset_index(drop=True)
    if with_dup and len(df) > 4:
        df.loc[4, "date"] = df.loc[3, "date"]
    return df


def _fast_config(**overrides: object) -> sb.SovereignBriefConfig:
    defaults: dict[str, object] = {
        "n_b": 41,
        "n_y": 15,
        "sim_periods": 6000,
        "sim_seeds": (1979, 1980),
        "sim_burn": 300,
        "dsa_horizon": 12,
        "dsa_n_sims": 100,
        "sensitivity_theta": (0.15, 0.282),
        "sensitivity_sigma_scale": (0.8, 1.2),
        "income_col": "real_gdp",
        "spread_col": "spread_pct",
    }
    defaults.update(overrides)
    return sb.SovereignBriefConfig(**defaults)  # type: ignore[arg-type]


def test_config_and_calendar_validation_rejections(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frequency"):
        sb.SovereignBriefConfig(frequency="monthly")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rate_convention"):
        sb.SovereignBriefConfig(rate_convention="continuous")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="beta"):
        sb.SovereignBriefConfig(beta=1.02)
    with pytest.raises(ValueError, match="theta"):
        sb.SovereignBriefConfig(theta=0.0)
    with pytest.raises(ValueError, match="filter_col and filter_value"):
        sb.SovereignBriefConfig(filter_col="country")

    bad_cfg = tmp_path / "bad_sov.json"
    bad_cfg.write_text(json.dumps({"not_a_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        sb.load_config(bad_cfg)

    # Gap in quarterly series
    gap_df = _custom_macro_df(frequency="quarterly", n_obs=24, with_gap=True)
    with pytest.raises(ValueError, match="Non-contiguous"):
        sb.prepare_sovereign_data(gap_df, _fast_config())

    # Duplicate period
    dup_df = _custom_macro_df(frequency="quarterly", n_obs=24, with_dup=True)
    with pytest.raises(ValueError, match="Duplicate"):
        sb.prepare_sovereign_data(dup_df, _fast_config())

    # Short hard minimum rejection (< 12 quarterly, < 8 annual)
    tiny_q = _custom_macro_df(frequency="quarterly", n_obs=10)
    with pytest.raises(ValueError, match="Insufficient sample length"):
        sb.prepare_sovereign_data(tiny_q, _fast_config())

    tiny_a = _custom_macro_df(frequency="annual", n_obs=7)
    with pytest.raises(ValueError, match="Insufficient sample length"):
        sb.prepare_sovereign_data(tiny_a, _fast_config(frequency="annual"))

    # Missing mapped column, negative debt, NaN values, negative income level
    bad_col_df = _custom_macro_df(n_obs=24).drop(columns=["pb"])
    with pytest.raises(ValueError, match="Missing mapped column"):
        sb.prepare_sovereign_data(bad_col_df, _fast_config())

    neg_debt_df = _custom_macro_df(n_obs=24)
    neg_debt_df.loc[2, "debt"] = -0.1
    with pytest.raises(ValueError, match="cannot be negative"):
        sb.prepare_sovereign_data(neg_debt_df, _fast_config())

    nan_df = _custom_macro_df(n_obs=24)
    nan_df.loc[5, "g"] = float("nan")
    with pytest.raises(ValueError, match="NaN or non-finite"):
        sb.prepare_sovereign_data(nan_df, _fast_config())

    neg_inc_df = _custom_macro_df(n_obs=24)
    neg_inc_df.loc[1, "real_gdp"] = -5.0
    with pytest.raises(ValueError, match="strictly positive"):
        sb.prepare_sovereign_data(neg_inc_df, _fast_config())


def test_explicit_units_frequency_and_rate_convention_consistency() -> None:
    df = _custom_macro_df(frequency="quarterly", n_obs=48)
    cfg_pct = _fast_config(
        rate_unit="pct",
        debt_unit="fraction",
        spread_unit="pct",
        rate_convention="apr",
        exact_compounding=False,
    )
    res_pct = sb.compute_brief(df, cfg_pct)

    # Convert columns explicitly to fraction/percent/bps & renamed headers
    df_alt = df.rename(
        columns={
            "date": "quarter_id",
            "g": "gdp_growth",
            "r": "real_rate",
            "pb": "prim_bal",
            "debt": "debt_pct",
            "spread_pct": "spread_bps",
        }
    ).copy()
    df_alt["gdp_growth"] /= 100.0
    df_alt["real_rate"] /= 100.0
    df_alt["prim_bal"] /= 100.0
    df_alt["debt_pct"] *= 100.0
    df_alt["spread_bps"] *= 100.0

    cfg_alt = _fast_config(
        date_col="quarter_id",
        growth_col="gdp_growth",
        interest_rate_col="real_rate",
        primary_balance_col="prim_bal",
        debt_col="debt_pct",
        spread_col="spread_bps",
        rate_unit="fraction",
        debt_unit="percent",
        spread_unit="bps",
        rate_convention="apr",
        exact_compounding=False,
    )
    res_alt = sb.compute_brief(df_alt, cfg_alt)

    assert res_alt.data.rho_y == pytest.approx(res_pct.data.rho_y, rel=1e-10)
    assert res_alt.data.sigma_y == pytest.approx(res_pct.data.sigma_y, rel=1e-10)
    assert res_alt.data.empirical_spread_mean == pytest.approx(
        res_pct.data.empirical_spread_mean, rel=1e-10
    )
    assert res_alt.dsa.dar_95 == pytest.approx(res_pct.dsa.dar_95, rel=1e-10)

    # Switching rate_convention="effective" and exact_compounding=True changes simulated DSA path
    cfg_eff = _fast_config(rate_convention="effective", exact_compounding=True)
    res_eff = sb.compute_brief(df, cfg_eff)
    assert res_eff.dsa.metadata["rate_convention"] == "effective"
    assert res_eff.dsa.metadata["exact_compounding"] is True
    assert not np.allclose(res_eff.dsa.baseline_path, res_pct.dsa.baseline_path)

    # Non-default pb_max is forwarded into DSA simulation even under default apr/linear convention
    cfg_cap = _fast_config(pb_max=0.2, rate_convention="apr", exact_compounding=False)
    res_cap = sb.compute_brief(df, cfg_cap)
    assert res_cap.dsa.metadata["pb_max"] == pytest.approx(0.2)
    assert res_cap.fiscal_reaction.reaction_function.pb_max == pytest.approx(0.2)
    assert not np.allclose(res_cap.dsa.baseline_path, res_pct.dsa.baseline_path)


def test_panel_filter_annual_advisory_and_short_sample_guards() -> None:
    df_a = _custom_macro_df(frequency="annual", n_obs=10)
    df_b = _custom_macro_df(frequency="annual", n_obs=14)
    df_a["iso3"] = "AAA"
    df_b["iso3"] = "BBB"
    stacked = pd.concat([df_a, df_b], ignore_index=True)

    # Filter AAA: n=10 (< SHORT_SAMPLE_OBS["annual"]=12), default quarterly structural parameters,
    # and income_col=None (growth proxy reconstruction)
    cfg = _fast_config(
        frequency="annual",
        filter_col="iso3",
        filter_value="AAA",
        income_col=None,
        spread_col=None,
        dsa_horizon=8,
    )
    res = sb.compute_brief(stacked, cfg)
    assert res.data.n_obs == 10
    assert res.config.hp_lambda == pytest.approx(6.25)

    joined_en = " ".join(res.warnings_en)
    joined_es = " ".join(res.warnings_es)
    assert "Short sample warning" in joined_en
    assert "Advertencia de muestra corta" in joined_es
    assert "Annual data frequency" in joined_en
    assert "frecuencia anual" in joined_es
    assert "No income_col mapped" in joined_en
    assert "No se mapeo income_col" in joined_es

    # Missing filter match raises cleanly
    with pytest.raises(ValueError, match="matched zero rows"):
        sb.prepare_sovereign_data(
            stacked, _fast_config(frequency="annual", filter_col="iso3", filter_value="NONE")
        )


def test_arellano_comparative_statics_and_sensitivity_grid() -> None:
    df = _custom_macro_df(frequency="quarterly", n_obs=72)
    cfg = _fast_config(
        n_b=61,
        n_y=17,
        sensitivity_theta=(0.15, 0.45),
        sensitivity_sigma_scale=(0.8, 1.3),
    )
    res = sb.compute_brief(df, cfg)
    sens = res.sensitivity
    assert len(sens) == 4
    assert bool(sens["converged"].all())

    # Higher reentry theta reduces default punishment -> lower value of market access / debt capacity
    low_theta = sens[
        np.isclose(sens["theta"], 0.15) & np.isclose(sens["sigma_scale"], 0.8)
    ].iloc[0]
    high_theta = sens[
        np.isclose(sens["theta"], 0.45) & np.isclose(sens["sigma_scale"], 0.8)
    ].iloc[0]
    assert float(high_theta["value_of_market_access"]) < float(low_theta["value_of_market_access"])
    assert float(high_theta["debt_output_pct"]) < float(low_theta["debt_output_pct"])


def test_cli_config_and_data_exports_bilingual_and_portable_manifest(tmp_path: Path) -> None:
    assert sb.main([]) == 0

    in_dir = tmp_path / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    csv_path = in_dir / "country_macro.csv"
    _custom_macro_df(frequency="quarterly", n_obs=32).to_csv(csv_path, index=False, encoding="utf-8")

    cfg_path = in_dir / "sov_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "frequency": "quarterly",
                "rate_convention": "effective",
                "exact_compounding": True,
                "income_col": "real_gdp",
                "spread_col": "spread_pct",
                "n_b": 41,
                "n_y": 15,
                "sim_periods": 6000,
                "sim_seeds": [1979, 1980],
                "sim_burn": 300,
                "dsa_horizon": 10,
                "dsa_n_sims": 80,
                "sensitivity_theta": [0.15, 0.282],
                "sensitivity_sigma_scale": [0.9, 1.1],
                "lang": "es",
                "dataset_label": "Synthetic Sovereign Test CSV",
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "outputs"
    rc = sb.main(["--config", str(cfg_path), "--data", str(csv_path), "--output", str(out_dir)])
    assert rc == 0

    expected_files = {
        "summary.md",
        "summary_en.md",
        "summary_es.md",
        "tables_en.md",
        "tables_es.md",
        "dsa_summary.csv",
        "dsa_paths.csv",
        "fiscal_reaction.csv",
        "sovereign_sensitivity.csv",
        "baseline_moments.csv",
        "figure.png",
        "record.json",
        "manifest.json",
    }
    assert expected_files.issubset({p.name for p in out_dir.iterdir()})

    sum_en = (out_dir / "summary_en.md").read_text(encoding="utf-8")
    sum_es = (out_dir / "summary_es.md").read_text(encoding="utf-8")
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == sum_es
    assert sum_en != sum_es
    assert "Short sample warning" in sum_en
    assert "Advertencia de muestra corta" in sum_es
    assert "rate_convention=effective" in sum_en and "rate_convention=effective" in sum_es

    png_bytes = (out_dir / "figure.png").read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n" and len(png_bytes) > 10_000

    # Manifest & SHA-256 verification + zero host path leakage
    manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8")
    record_text = (out_dir / "record.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    record = json.loads(record_text)

    expected_csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert manifest["inputs"]["input_file"] == "country_macro.csv"
    assert manifest["inputs"]["input_sha256"] == expected_csv_sha
    assert record["data_provenance"]["input_sha256"] == expected_csv_sha
    assert set(manifest["artifacts"].keys()) == (expected_files - {"manifest.json"})
    for rel, meta in manifest["artifacts"].items():
        assert meta["sha256"] == hashlib.sha256((out_dir / rel).read_bytes()).hexdigest()
        assert meta["bytes"] == (out_dir / rel).stat().st_size

    forbidden = [str(tmp_path), str(tmp_path.parent), str(csv_path), str(out_dir)]
    for blob in (manifest_text, record_text, sum_en, sum_es):
        for path_str in forbidden:
            assert path_str not in blob


def test_cli_demo_mode_works(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo_out"
    rc = sb.main(["--demo", "--output", str(out_dir), "--dsa-horizon", "8"])
    assert rc == 0
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "figure.png").is_file()


def test_text_io_always_names_utf8_encoding() -> None:
    src_path = Path(sb.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("read_text", "write_text", "to_csv", "read_csv"):
                kw_names = {kw.arg for kw in node.keywords}
                assert "encoding" in kw_names, f"{node.func.attr} missing explicit encoding="
