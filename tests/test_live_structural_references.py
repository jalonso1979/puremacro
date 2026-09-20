"""Small frozen external references; no MATLAB or archive download in ordinary CI."""
from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from tools.reference_validation.validate_dynare import FIXTURES as DYNARE_FIXTURES, compare_case
from tools.reference_validation.validate_oecd import load_fixture, validate_aggregate
from puremacro.trade._oecd_icio import _parse_frame, condense_final_demand, read_native
from puremacro.trade.data import package_mrio_to_calibration_result

CASES = sorted(json.loads((DYNARE_FIXTURES / "manifest.json").read_text())["cases"])


@pytest.mark.parametrize("case", CASES)
def test_live_dynare_tensors_risk_and_common_innovation_moments(case):
    result = compare_case(case)
    assert result["passed"], result


@pytest.fixture(scope="module")
def raw_oecd():
    return load_fixture()


def frame_from_raw(raw):
    nodes = [f"{c}_{s}" for c in raw.countries for s in raw.sectors]
    fd = [f"{c}_{f}" for c in raw.countries for f in raw.fd_categories]
    frame = pd.DataFrame(0., index=nodes+["TLS", "VA", "OUT"], columns=nodes+fd+["OUT"])
    frame.loc[nodes, nodes] = raw.Z
    frame.loc[nodes, fd] = raw.F
    frame.loc["TLS", nodes] = raw.TLS
    frame.loc["TLS", fd] = raw.taxes_less_subsidies_fd
    frame.loc["VA", nodes] = raw.VA
    frame.loc[nodes, "OUT"] = raw.gross_output
    frame.loc["OUT", nodes] = raw.gross_output
    return frame


def parse(frame, raw):
    return _parse_frame(frame, countries=raw.countries, sectors=raw.sectors, year=2019)


def test_real_oecd_aggregation_counterfactual_and_currency_units(raw_oecd):
    result = validate_aggregate(raw_oecd)
    assert result["passed"], result


def test_native_label_reader_permutation_and_reported_output(raw_oecd):
    frame = frame_from_raw(raw_oecd)
    raw = parse(frame.iloc[::-1, ::-1], raw_oecd)
    for name in ("Z", "F", "VA", "TLS", "gross_output", "taxes_less_subsidies_fd"):
        np.testing.assert_array_equal(getattr(raw, name), getattr(raw_oecd, name))
    # Source imbalances must be measured before repairs, not silently overwritten.
    assert raw.metadata["provider_accounting"]["max_abs_sales_minus_outlays"] > 1


@pytest.mark.parametrize("mutation", ["missing_row", "unknown_column", "duplicate_row", "nonfinite", "wrong_output"])
def test_native_reader_rejects_malformed_schema(raw_oecd, mutation):
    frame = frame_from_raw(raw_oecd)
    if mutation == "missing_row":
        frame = frame.iloc[1:]
    elif mutation == "unknown_column":
        frame = frame.rename(columns={frame.columns[0]: "XXX_MYSTERY"})
    elif mutation == "duplicate_row":
        frame.index = [frame.index[1]] + list(frame.index[1:])
    elif mutation == "nonfinite":
        frame.iloc[0, 0] = np.nan
    else:
        frame.iloc[0, -1] += 1
    with pytest.raises(ValueError):
        parse(frame, raw_oecd)


def test_oecd_fd_mapping_preserves_signed_inventory_and_taxes(raw_oecd):
    # A negative stock change is a legitimate signed input, not missing data.
    F = raw_oecd.F.copy().reshape(raw_oecd.M, raw_oecd.C, raw_oecd.K_F)
    F[0, 0, 4] = -123.
    altered = replace(raw_oecd, final_demand_matrix=F.reshape(raw_oecd.M, -1))
    compact = condense_final_demand(altered)
    np.testing.assert_allclose(compact.F.reshape(raw_oecd.M, raw_oecd.C, 3)[:, :, 1], F[:, :, 3]+F[:, :, 4])
    np.testing.assert_allclose(compact.F.sum(axis=1), altered.F.sum(axis=1))
    np.testing.assert_allclose(compact.taxes_less_subsidies_fd.sum(), raw_oecd.taxes_less_subsidies_fd.sum())
    assert compact.fd_categories == ["C", "I", "Cx"]


def test_provider_output_discrepancies_reconciled_and_disclosed(raw_oecd):
    raw = condense_final_demand(raw_oecd)
    calib = package_mrio_to_calibration_result(raw)
    np.testing.assert_allclose(calib.ytot.ravel(order="F"), raw.Z.sum(1)+raw.F.sum(1))
    assert calib.metadata["adjustments"]["Y"]["changed_entries"] == raw.M
    assert calib.metadata["adjustments"]["Z"]["changed_entries"] == 0
    assert calib.metadata["adjustments"]["F"]["changed_entries"] == 0
    assert calib.metadata["output_reconciliation"] == "transaction row sales"


def test_native_zip_selects_year_by_name_and_validates_archive(monkeypatch, tmp_path, raw_oecd):
    # Exercise the same schema on the empirical aggregation with small rosters.
    from puremacro.trade import data
    monkeypatch.setattr(data, "OECD_77_COUNTRIES", tuple(raw_oecd.countries))
    monkeypatch.setattr(data, "OECD_45_SECTORS", tuple(raw_oecd.sectors))
    table = frame_from_raw(raw_oecd).to_csv(index_label="V1").encode()
    archive = tmp_path / "2016-2020_SML.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("release/2019_SML.csv", table)
        z.writestr("release/2020_SML.csv", b"not selected")
    result = read_native(archive, 2019)
    np.testing.assert_allclose(result.Z, raw_oecd.Z)
    assert result.metadata["archive_member"] == "release/2019_SML.csv"
    assert result.metadata["archive_sha256"] != result.metadata["sha256"]
    with pytest.raises(ValueError, match="exactly one"):
        read_native(archive, 2018)
    calib = data.load_oecd_icio_granular(year=2019, file_path=archive)
    assert calib.n_final_demand == 3
    with pytest.raises(ValueError, match="1995"):
        read_native(archive, 2022)


def test_native_csv_rejects_duplicate_header_before_pandas_mangling(tmp_path, raw_oecd):
    frame = frame_from_raw(raw_oecd)
    cols = list(frame.columns); cols[1] = cols[0]; frame.columns = cols
    path = tmp_path / "2019_SML.csv"
    frame.to_csv(path, index_label="V1")
    with pytest.raises(ValueError, match="duplicate column"):
        read_native(path, 2019)
