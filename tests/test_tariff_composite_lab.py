"""Tests for puremacro.examples.tariff_composite_lab."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from puremacro.examples import tariff_composite_lab as lab
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade._results import TradeCalibrationResult


def _tiny_calib() -> TradeCalibrationResult:
    """Consistent symmetric 2-country, 2-sector, 1-final-demand calibration (<50ms solve)."""
    mat = np.zeros((7, 6), dtype=float)
    mat[:4, :4] = [
        [20.0, 10.0, 8.0, 2.0],
        [10.0, 20.0, 2.0, 8.0],
        [8.0, 2.0, 20.0, 10.0],
        [2.0, 8.0, 10.0, 20.0],
    ]
    mat[4, :4] = [0.0, 0.0, 0.0, 0.0]
    mat[5, :4] = [35.0, 35.0, 35.0, 35.0]
    mat[6, :4] = [25.0, 25.0, 25.0, 25.0]
    mat[:4, 4] = [42.0, 42.0, 18.0, 18.0]
    mat[:4, 5] = [18.0, 18.0, 42.0, 42.0]
    return calibrate_trade_model(
        mat,
        ns=2,
        nc=2,
        nfd=1,
        country_codes=("DOM", "FOR"),
        sector_codes=("MAN", "SRV"),
    )


def _tiny_config(**overrides: Any) -> lab.TariffLabConfig:
    defaults: dict[str, Any] = {
        "importer": "DOM",
        "partner": "FOR",
        "rate": 0.15,
        "retaliation_rate": 0.15,
        "exempt_sectors": ("SRV",),
        "escalation_sectors": ("MAN",),
        "focus_countries": ("DOM", "FOR"),
        "household_source": "auto",
    }
    defaults.update(overrides)
    return lab.TariffLabConfig(**defaults)


def _skewed_groups_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_weight": [0.4, 0.4, 0.2],
            "annual_spending": [20000.0, 50000.0, 120000.0],
            "MAN": [0.80, 0.50, 0.20],
            "SRV": [0.20, 0.50, 0.80],
        },
        index=pd.Index(["low", "mid", "high"], name="group"),
    )


def test_config_validation_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="importer and partner"):
        lab.TariffLabConfig(importer="USA", partner="USA")
    with pytest.raises(ValueError, match="rate must be finite"):
        lab.TariffLabConfig(rate=0.0)
    with pytest.raises(ValueError, match="rate must be finite"):
        lab.TariffLabConfig(rate=float("nan"))
    with pytest.raises(ValueError, match="retaliation_rate"):
        lab.TariffLabConfig(retaliation_rate=-0.05)
    with pytest.raises(ValueError, match="on_conflict"):
        lab.TariffLabConfig(on_conflict="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="welfare_rule"):
        lab.TariffLabConfig(welfare_rule="stone_geary_bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ces_elasticity"):
        lab.TariffLabConfig(ces_elasticity=0.0)
    with pytest.raises(ValueError, match="groups_path"):
        lab.TariffLabConfig(household_source="csv", groups_path=None)

    bad_json = tmp_path / "bad_cfg.json"
    bad_json.write_text(json.dumps({"unexpected_key": 123}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        lab.load_config(bad_json)


def test_package_composition_conflict_rules_and_error_guards() -> None:
    calib = _tiny_calib()
    cfg = _tiny_config(on_conflict="last_wins", rate=0.15, escalation_rate=0.25)
    packages, conflicts = lab.build_tariff_packages(cfg, calib)

    assert set(packages.keys()) == {"unilateral", "retaliatory", "phased_exempted"}
    assert not conflicts.empty
    phased_cf = conflicts[conflicts["package"] == "phased_exempted"].set_index("sector")
    assert phased_cf.loc["MAN", "resolved_rate"] == pytest.approx(0.25)
    assert phased_cf.loc["SRV", "resolved_rate"] == pytest.approx(0.0)
    assert all(phased_cf["rule"] == "last_wins")

    # Max resolution favors escalation & retains base tariff over 0.0 carve-out
    cfg_max = _tiny_config(on_conflict="max", rate=0.15, escalation_rate=0.25)
    _, cf_max = lab.build_tariff_packages(cfg_max, calib)
    ph_max = cf_max[cf_max["package"] == "phased_exempted"].set_index("sector")
    assert ph_max.loc["MAN", "resolved_rate"] == pytest.approx(0.25)
    assert ph_max.loc["SRV", "resolved_rate"] == pytest.approx(0.15)

    # Strict error conflict mode rejects overlapping legs
    cfg_err = _tiny_config(on_conflict="error")
    with pytest.raises(ValueError, match="conflict"):
        lab.build_tariff_packages(cfg_err, calib)

    # Unknown sector / country rejection
    with pytest.raises(ValueError, match="Unknown importer"):
        lab.build_tariff_packages(_tiny_config(importer="ZZZ"), calib)
    with pytest.raises(ValueError, match="unknown sector"):
        lab.build_tariff_packages(_tiny_config(exempt_sectors=("BADSEC",)), calib)


def test_distributional_incidence_custom_csv_and_preference_rules(tmp_path: Path) -> None:
    calib = _tiny_calib()
    groups_csv = tmp_path / "groups.csv"
    _skewed_groups_df().reset_index().to_csv(groups_csv, index=False, encoding="utf-8")

    cfg = _tiny_config(
        groups_path=str(groups_csv),
        household_source="csv",
        welfare_rule="fixed_baskets",
    )
    res = lab.compute_tariff_lab(
        cfg,
        calib=calib,
        solver_opts={"method": "equilibrated_newton", "tol": 1e-7, "foreign_saving_units": "world_income"},
    )
    assert res.groups_meta["source"] == "user_csv"
    assert res.groups_meta["file"] == "groups.csv"
    assert res.groups_meta["sha256"] == hashlib.sha256(groups_csv.read_bytes()).hexdigest()

    inc = res.household_incidence
    assert set(inc["rebate_rule"].unique()) == {"equal", "proportional", "none"}

    # Check aggregate consistency of proportional rebate vs representative national EV
    compact = res.comparison.to_frame("compact")
    dom_uni_ev = float(
        compact[(compact["scenario"] == "unilateral") & (compact["country"] == "DOM")][
            "ev_pct_consumption"
        ].iloc[0]
    )
    dw_prop = res.distributional_results[("unilateral", "proportional")]
    assert float(dw_prop.aggregate["ev_pct"]) == pytest.approx(dom_uni_ev, abs=1e-3)

    # Equal per-household rebate redistributes strictly toward lower-spending households vs proportional
    sub_eq = inc[(inc["scenario"] == "unilateral") & (inc["rebate_rule"] == "equal")].set_index("group")
    sub_pr = inc[(inc["scenario"] == "unilateral") & (inc["rebate_rule"] == "proportional")].set_index("group")
    sub_no = inc[(inc["scenario"] == "unilateral") & (inc["rebate_rule"] == "none")].set_index("group")
    assert float(sub_eq.loc["low", "ev_pct"]) > float(sub_pr.loc["low", "ev_pct"])
    assert float(sub_eq.loc["high", "ev_pct"]) < float(sub_pr.loc["high", "ev_pct"])
    assert float(sub_eq.loc["low", "ev_pct"]) > float(sub_no.loc["low", "ev_pct"])

    # Cobb-Douglas preference rule executes cleanly and produces distinct EVs
    cfg_cd = _tiny_config(
        groups_path=str(groups_csv),
        household_source="csv",
        welfare_rule="cobb_douglas",
    )
    res_cd = lab.compute_tariff_lab(
        cfg_cd,
        calib=calib,
        solver_opts={"method": "equilibrated_newton", "tol": 1e-7, "foreign_saving_units": "world_income"},
    )
    assert np.all(np.isfinite(res_cd.household_incidence["ev_pct"].to_numpy(dtype=float)))


def test_adversarial_household_csv_rejections(tmp_path: Path) -> None:
    calib = _tiny_calib()
    base = _skewed_groups_df().reset_index()

    # Row share sums != 1
    bad_sum = base.copy()
    bad_sum.loc[0, "MAN"] = 0.90
    p_sum = tmp_path / "bad_sum.csv"
    bad_sum.to_csv(p_sum, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1.0"):
        lab.load_lab_groups(_tiny_config(groups_path=str(p_sum), household_source="csv"), calib)

    # Negative share
    bad_neg = base.copy()
    bad_neg.loc[0, "MAN"] = -0.10
    bad_neg.loc[0, "SRV"] = 1.10
    p_neg = tmp_path / "bad_neg.csv"
    bad_neg.to_csv(p_neg, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="negative"):
        lab.load_lab_groups(_tiny_config(groups_path=str(p_neg), household_source="csv"), calib)

    # Extra unexpected column
    bad_col = base.copy()
    bad_col["extra"] = 1.0
    p_col = tmp_path / "bad_col.csv"
    bad_col.to_csv(p_col, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected columns"):
        lab.load_lab_groups(_tiny_config(groups_path=str(p_col), household_source="csv"), calib)

    # Zero population weight
    bad_pop = base.copy()
    bad_pop.loc[1, "population_weight"] = 0.0
    p_pop = tmp_path / "bad_pop.csv"
    bad_pop.to_csv(p_pop, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="strictly positive"):
        lab.load_lab_groups(_tiny_config(groups_path=str(p_pop), household_source="csv"), calib)

    # Missing file
    with pytest.raises(FileNotFoundError):
        lab.load_lab_groups(
            _tiny_config(groups_path=str(tmp_path / "missing.csv"), household_source="csv"),
            calib,
        )


def test_load_config_with_custom_packages_and_relative_groups(tmp_path: Path) -> None:
    calib = _tiny_calib()
    groups_csv = tmp_path / "hh.csv"
    _skewed_groups_df().reset_index().to_csv(groups_csv, index=False, encoding="utf-8")

    cfg_path = tmp_path / "custom_lab.json"
    cfg_payload = {
        "importer": "DOM",
        "partner": "FOR",
        "focus_countries": ["DOM", "FOR"],
        "groups_path": "hh.csv",
        "lang": "es",
        "packages": [
            {
                "name": "staged_custom",
                "on_conflict": "last_wins",
                "legs": [
                    {
                        "importer": "DOM",
                        "name": "broad_10",
                        "tariffs": [{"exporter": "FOR", "sector": "*", "rate": 0.10}],
                    },
                    {
                        "importer": "DOM",
                        "name": "srv_zero",
                        "tariffs": {"FOR:SRV": 0.0},
                    },
                ],
            }
        ],
    }
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
    loaded = lab.load_config(cfg_path)
    assert loaded.household_source == "csv"
    assert loaded.groups_path is not None and Path(loaded.groups_path).name == "hh.csv"

    out_dir = tmp_path / "out_custom"
    res, paths = lab.run_tariff_lab(
        loaded,
        out_dir,
        calib=calib,
        solver_opts={"method": "equilibrated_newton", "tol": 1e-7, "foreign_saving_units": "world_income"},
    )
    assert set(res.packages.keys()) == {"staged_custom"}
    assert res.conflicts.iloc[0]["resolved_rate"] == pytest.approx(0.0)
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == (
        out_dir / "summary_es.md"
    ).read_text(encoding="utf-8")
    assert "manifest.json" in paths


def test_cli_bundled_icio_end_to_end_bilingual_and_portable_manifest(tmp_path: Path) -> None:
    assert lab.main([]) == 0

    out_dir = tmp_path / "icio_run"
    rc = lab.main(
        [
            "--output",
            str(out_dir),
            "--importer",
            "MEX",
            "--partner",
            "USA",
            "--rate",
            "0.15",
            "--retaliation-rate",
            "0.10",
            "--countries",
            "MEX,USA,CHN",
            "--enigh",
            "--lang",
            "es",
        ]
    )
    assert rc == 0

    expected_files = {
        "summary.md",
        "summary_en.md",
        "summary_es.md",
        "tables_en.md",
        "tables_es.md",
        "national_comparison.csv",
        "scenario_summary.csv",
        "payoff_matrix.csv",
        "robustness_summary.csv",
        "tariff_conflicts.csv",
        "household_incidence.csv",
        "comparison.png",
        "record.json",
        "manifest.json",
    }
    assert expected_files.issubset({p.name for p in out_dir.iterdir()})

    # Bilingual parity & PNG signature check
    sum_en = (out_dir / "summary_en.md").read_text(encoding="utf-8")
    sum_es = (out_dir / "summary_es.md").read_text(encoding="utf-8")
    assert (out_dir / "summary.md").read_text(encoding="utf-8") == sum_es
    assert sum_en != sum_es
    assert "Executive Summary" in sum_en
    assert "Resumen Ejecutivo" in sum_es
    assert "enigh_2022" in sum_en and "enigh_2022" in sum_es

    png_bytes = (out_dir / "comparison.png").read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n" and len(png_bytes) > 10_000

    # Manifest SHA-256 verification and zero host path leakage
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    record_text = (out_dir / "record.json").read_text(encoding="utf-8")
    manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert manifest["schema"] == 1
    assert manifest["inputs"]["groups_file"] == "enigh2022_household_policy.npz"
    assert isinstance(manifest["inputs"]["groups_sha256"], str)
    assert len(manifest["inputs"]["groups_sha256"]) == 64
    assert set(manifest["artifacts"].keys()) == (expected_files - {"manifest.json"})
    for rel, info in manifest["artifacts"].items():
        actual_sha = hashlib.sha256((out_dir / rel).read_bytes()).hexdigest()
        assert info["sha256"] == actual_sha
        assert info["bytes"] == (out_dir / rel).stat().st_size

    forbidden_substrings = [str(tmp_path), str(tmp_path.parent), str(out_dir)]
    for blob in (manifest_text, record_text, sum_en, sum_es):
        for s in forbidden_substrings:
            assert s not in blob


def test_text_io_always_names_utf8_encoding() -> None:
    src_path = Path(lab.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("read_text", "write_text", "to_csv", "read_csv"):
                kw_names = {kw.arg for kw in node.keywords}
                assert "encoding" in kw_names, f"{node.func.attr} missing explicit encoding="
