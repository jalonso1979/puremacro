"""Exact published specification, independent software evidence and controls."""
import json

import numpy as np
import pandas as pd
import pytest

from puremacro.replication.romer_romer_2010 import (
    estimate_rr2010_baseline, load_rr2010_data, load_rr2010_reference,
)


@pytest.fixture(scope="module")
def result():
    return estimate_rr2010_baseline()


def test_original_data_are_complete_authenticated_and_offline():
    frame = load_rr2010_data()
    assert frame.index.equals(pd.period_range("1947Q1", "2007Q4", freq="Q"))
    assert tuple(frame.columns) == ("gdp", "nomgdp", "defic", "longr")
    assert frame.attrs["archive_sha256"] == "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c"
    assert frame.attrs["is_synthetic"] is False
    assert np.isfinite(frame.to_numpy()).all()


def test_full_software_parity_is_separate_from_published_rounding(result):
    reference = load_rr2010_reference()
    assert reference["metadata"]["runtime_imported_puremacro"] is False
    assert reference["metadata"]["software"] == "statsmodels"
    for name in ("coefficients", "coefficient_covariance", "irf", "irf_covariance",
                 "standard_errors", "t_statistics"):
        np.testing.assert_allclose(getattr(result, name), reference[name], rtol=1e-10, atol=1e-11)
    assert result.nobs == reference["nobs"] == 232
    assert result.df_resid == reference["df_resid"] == 218


def test_published_peak_passes_but_t_statistic_retains_strict_rounding_miss(result):
    # Romer--Romer (2010), p.781, Figure4 / equation6: rounded paper targets.
    assert np.argmin(result.irf) == 10
    assert abs(result.irf[10] - (-3.08)) < .005
    assert f"{result.irf[10]:.2f}" == "-3.08"
    assert f"{result.t_statistics[10]:.2f}" == "-3.52"
    gap = abs(result.t_statistics[10] - (-3.53)) - .005
    assert 3.9e-6 < gap < 4.0e-6
    comparison = load_rr2010_reference()["metadata"]["published_comparison"]
    assert comparison["h10_response"]["passed"] is True
    assert comparison["h10_t_statistic"]["passed"] is False
    assert comparison["trough_horizon"]["passed"] is True
    audit = result.metadata["original_program_databank_crosscheck"]
    assert abs(audit["h10_t_statistic"] - (-3.53)) > .005
    np.testing.assert_allclose(audit["h10_t_statistic"], result.t_statistics[10], atol=1e-12)


def test_complete_covariance_is_carried_into_structural_targets(result):
    targets = result.to_moment_targets()
    np.testing.assert_array_equal(targets.values, result.irf)
    np.testing.assert_allclose(targets.covariance, result.irf_covariance, rtol=1e-14, atol=1e-14)
    assert np.linalg.eigvalsh(targets.covariance).min() > 0
    assert np.any(np.abs(targets.covariance - np.diag(np.diag(targets.covariance))) > .01)
    assert result.metadata["sample_start"] == "1950Q1"
    assert result.metadata["sample_end"] == "2007Q4"
    assert result.metadata["shock_size"] == 1.
    assert result.metadata["coefficient_labels"] == ["constant"] + [f"tax_lag_{i}" for i in range(13)]
    assert targets.metadata["covariance_type"] == "conventional homoskedastic OLS, SSR/(n-k)"


def test_sum_of_marginal_variances_is_not_cumulative_covariance(result):
    diagonal_only = np.cumsum(np.diag(result.coefficient_covariance)[1:])
    assert not np.allclose(diagonal_only, np.diag(result.irf_covariance), rtol=.01, atol=.01)
    transform = np.zeros((13, 14))
    for h in range(13):
        transform[h, 1:h+2] = 1.
    np.testing.assert_allclose(result.irf_covariance, transform@result.coefficient_covariance@transform.T)


@pytest.mark.parametrize("mutation", ["missing_quarter", "reverse", "duplicate", "monthly",
                                       "missing_column", "nan", "infinite", "zero_gdp", "negative_nomgdp"])
def test_malformed_input_never_silently_changes_sample(mutation):
    frame = load_rr2010_data()
    if mutation == "missing_quarter":
        frame = frame.iloc[1:]
    elif mutation == "reverse":
        frame = frame.iloc[::-1]
    elif mutation == "duplicate":
        frame.index = frame.index[:-1].append(frame.index[-2:-1])
    elif mutation == "monthly":
        frame.index = pd.period_range("1947-01", periods=len(frame), freq="M")
    elif mutation == "missing_column":
        frame = frame.drop(columns="longr")
    else:
        name = "nomgdp" if mutation == "negative_nomgdp" else "gdp"
        frame.iloc[40, frame.columns.get_loc(name)] = {
            "nan": np.nan, "infinite": np.inf, "zero_gdp": 0., "negative_nomgdp": -1.}[mutation]
    with pytest.raises(ValueError):
        estimate_rr2010_baseline(frame)


def test_tax_sign_and_scale_are_effective_negative_controls(result):
    frame = load_rr2010_data()
    for multiplier in (-1., 100.):
        changed = frame.copy()
        changed[["defic", "longr"]] *= multiplier
        fitted = estimate_rr2010_baseline(changed)
        np.testing.assert_allclose(fitted.irf, result.irf/multiplier, rtol=1e-10, atol=1e-11)
        assert abs(fitted.irf[10] - (-3.08)) > .005
        assert fitted.metadata["input_origin"] == "caller-supplied sensitivity data"


def test_coefficient_and_covariance_mutations_fail_external_comparison(result):
    reference = load_rr2010_reference()
    wrong_irf = result.irf.copy()
    wrong_irf[10] += .01
    assert abs(wrong_irf[10] - (-3.08)) > .005
    wrong_covariance = result.irf_covariance/result.nobs
    assert not np.allclose(wrong_covariance, reference["irf_covariance"], rtol=1e-10, atol=1e-11)


def test_reference_and_data_checksum_corruption_fail_closed(monkeypatch, tmp_path):
    from importlib import resources
    from puremacro.replication import romer_romer_2010 as module
    source = resources.files("puremacro.replication.data")
    for name in ("rr2010_original.csv", "rr2010_original_metadata.json", "rr2010_statsmodels_reference.json"):
        (tmp_path / name).write_bytes(source.joinpath(name).read_bytes())
    monkeypatch.setattr(module.resources, "files", lambda package: tmp_path)
    for name, load in (("rr2010_original.csv", load_rr2010_data),
                       ("rr2010_statsmodels_reference.json", load_rr2010_reference)):
        original = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(original+b" ")
        with pytest.raises(ValueError, match="checksum mismatch"):
            load()
        (tmp_path / name).write_bytes(original)


def test_rewriting_reference_and_its_metadata_does_not_reauthenticate(monkeypatch, tmp_path):
    import hashlib
    from importlib import resources
    from puremacro.replication import romer_romer_2010 as module
    source = resources.files("puremacro.replication.data")
    name = "rr2010_statsmodels_reference.json"
    content = source.joinpath(name).read_bytes()+b" "
    metadata = json.loads(source.joinpath("rr2010_original_metadata.json").read_bytes())
    metadata["reference_sha256"] = hashlib.sha256(content).hexdigest()
    (tmp_path / name).write_bytes(content)
    (tmp_path / "rr2010_original_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(module.resources, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_rr2010_reference()


def test_result_exports_preserve_units_and_json_ready_metadata(result):
    frame = result.to_frame()
    assert tuple(frame.columns) == ("h", "response", "se", "t", "lower_1se", "upper_1se")
    np.testing.assert_allclose(frame["upper_1se"]-frame["response"], result.standard_errors)
    json.dumps(result.metadata, allow_nan=False)
    with pytest.raises(ValueError):
        result.irf[0] = 42.


def test_reference_exporter_has_no_puremacro_imports():
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "tools/reference_validation/export_rr2010.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "statsmodels.regression.linear_model" in imported
    assert all(name != "puremacro" and not name.startswith("puremacro.") for name in imported)
