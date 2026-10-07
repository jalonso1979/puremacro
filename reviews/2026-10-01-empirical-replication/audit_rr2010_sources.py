"""Read-only independent RR2010 source and numerical audit.

Run with a local, original author archive, for example:
  .venv/bin/python reviews/2026-10-01-empirical-replication/audit_rr2010_sources.py \
      /tmp/puremacro-rr2010-DataSet.zip

The workbook is read directly as XML, without the converter/openpyxl. The
RATS binary layout is documented by the official Gretl implementation:
https://github.com/gretl-project/gretl/blob/master/lib/src/foreign_db.c
This script does not execute RATS or import puremacro.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


EXPECTED_SHA = "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c"
ROOT = "DataSet/RATS Programs and Databanks/"
NAMES = ("GDP", "NOMGDP", "DEFIC", "LONGR")


def workbook_inputs(payload):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    workbook = zipfile.ZipFile(io.BytesIO(payload))
    strings = ["".join(item.itertext()) for item in
               ET.fromstring(workbook.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    source = {}
    for sheet, mapping in ((1, {"D": "DEFIC", "E": "LONGR"}),
                           (2, {"C": "NOMGDP", "D": "GDP"})):
        root = ET.fromstring(workbook.read(f"xl/worksheets/sheet{sheet}.xml"))
        for row in root.findall("m:sheetData/m:row", ns):
            cells = {}
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                if value is None:
                    continue
                text = strings[int(value.text)] if cell.get("t") == "s" else value.text
                cells[re.sub(r"\d+$", "", cell.get("r"))] = text
            match = re.fullmatch(r"(\d{4})-(I|II|III|IV)", cells.get("A", "").strip())
            if match is None:
                continue
            year = int(match[1])
            if year < 1947:
                continue
            quarter = {"I": 1, "II": 2, "III": 3, "IV": 4}[match[2]]
            key = (year, quarter)
            for column, name in mapping.items():
                source.setdefault(name, {})[key] = float(cells[column])
    dates = [(year, quarter) for year in range(1947, 2008) for quarter in range(1, 5)]
    assert all(sorted(source[name]) == dates for name in NAMES)
    return {name: np.array([source[name][date] for date in dates]) for name in NAMES}


def rats_inputs(archive):
    """Decode named linked data records; all archive inputs are hash pinned."""
    source, details = {}, {}
    for member in ("GDP.DED", "NOMGDP.DED", "TAXNR.DED"):
        raw = archive.read(ROOT + member)
        assert raw[:4] == b"RAT4" and len(raw) % 256 == 0
        directory = struct.unpack_from("<i", raw, 30)[0]
        visited = set()
        while directory:
            assert 0 < directory <= len(raw) // 256 and directory not in visited
            visited.add(directory)
            offset = (directory - 1) * 256
            name = raw[offset + 16:offset + 32].decode("ascii").strip("\0 ")
            if name in NAMES:
                first = struct.unpack_from("<i", raw, offset + 12)[0]
                frequency = struct.unpack_from("<i", raw, offset + 44)[0]
                year, month = struct.unpack_from("<hh", raw, offset + 50)
                nobs = struct.unpack_from("<i", raw, offset + 56)[0]
                assert frequency == 4 and month == 1 and year in (1945, 1947)
                assert nobs == (2008 - year) * 4
                values, blocks = [], set()
                block = first
                while block and len(values) < nobs:
                    assert 0 < block <= len(raw) // 256 and block not in blocks
                    blocks.add(block)
                    position = (block - 1) * 256
                    values.extend(struct.unpack_from("<31d", raw, position + 8))
                    block = struct.unpack_from("<i", raw, position + 4)[0]
                assert len(values) >= nobs
                source[name] = np.asarray(values[:nobs])[(1947 - year) * 4:]
                details[name] = {"member": ROOT + member, "year": year, "frequency": frequency,
                                 "source_observations": nobs, "selected_observations": len(source[name]),
                                 "member_sha256": hashlib.sha256(raw).hexdigest()}
            directory = struct.unpack_from("<i", raw, offset + 4)[0]
    assert set(source) == set(NAMES)
    return source, details


def regress(source, method):
    tax = 100 * source["DEFIC"] / source["NOMGDP"] + 100 * source["LONGR"] / source["NOMGDP"]
    response = 100 * np.diff(np.log(source["GDP"]))[11:]
    design = np.array([[1., *[tax[t - h] for h in range(13)]] for t in range(12, 244)])
    q, r = np.linalg.qr(design, mode="reduced")
    coefficients = (np.linalg.solve(r, q.T @ response) if method == "qr" else
                    np.linalg.lstsq(design, response, rcond=None)[0])
    residual = response - design @ coefficients
    inverse_r = np.linalg.solve(r, np.eye(14))
    covariance = (residual @ residual / 218) * inverse_r @ inverse_r.T
    irf = np.cumsum(coefficients[1:])
    joint = np.array([[covariance[1:h + 2, 1:k + 2].sum() for k in range(13)] for h in range(13)])
    se = np.sqrt(np.diag(joint))
    return {"coefficients": coefficients, "coefficient_covariance": covariance,
            "irf": irf, "irf_covariance": joint, "standard_errors": se, "t_statistics": irf / se}


def main():
    archive_bytes = Path(sys.argv[1]).read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == EXPECTED_SHA
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    workbook = archive.read("DataSet/Romer&RomerFiscalData.xlsx")
    xml = workbook_inputs(workbook)
    rats, details = rats_inputs(archive)
    fits = {"workbook_svd": regress(xml, "svd"), "workbook_qr": regress(xml, "qr"),
            "rats_svd": regress(rats, "svd"), "rats_qr": regress(rats, "qr")}
    baseline = fits["workbook_svd"]
    data_root = Path(__file__).resolve().parents[2] / "puremacro/replication/data"
    with (data_root / "rr2010_original.csv").open() as stream:
        packaged_rows = list(csv.DictReader(stream))
    packaged = {name: np.array([float(row[name.lower()]) for row in packaged_rows]) for name in NAMES}
    reference = json.loads((data_root / "rr2010_statsmodels_reference.json").read_text())
    result = {
        "archive_sha256": EXPECTED_SHA,
        "archive_source": "https://eml.berkeley.edu/~dromer/papers/DataSet.zip",
        "workbook_sha256": hashlib.sha256(workbook).hexdigest(),
        "rats_format_reference": "https://github.com/gretl-project/gretl/blob/master/lib/src/foreign_db.c",
        "source_databanks": details,
        "source_max_absolute_differences": {name: float(np.max(np.abs(xml[name] - rats[name]))) for name in NAMES},
        "source_nonidentical_float_counts": {name: int(np.count_nonzero(xml[name] != rats[name])) for name in NAMES},
        "packaged_csv_max_abs_difference_from_raw_workbook": {
            name: float(np.max(np.abs(packaged[name] - xml[name]))) for name in NAMES},
        "frozen_statsmodels_max_abs_difference_from_workbook_svd": {
            metric: float(np.max(np.abs(np.asarray(reference[metric]) - baseline[metric]))) for metric in baseline},
        "nobs": 232, "regressors": 14, "residual_df": 218,
        "sample": "1950Q1--2007Q4; primitive inputs 1947Q1--2007Q4",
        "specification": "100*dlog(GDP) on constant and 100*(DEFIC+LONGR)/NOMGDP at lags0--12",
        "results": {name: {"response_h10": float(fit["irf"][10]),
                           "se_h10": float(fit["standard_errors"][10]),
                           "t_h10": float(fit["t_statistics"][10]),
                           "trough_horizon": int(np.argmin(fit["irf"])),
                           "max_abs_difference_vs_workbook_svd": {
                               metric: float(np.max(np.abs(fit[metric] - baseline[metric]))) for metric in baseline}}
                    for name, fit in fits.items()},
        "published_response": -3.08, "published_t": -3.53,
        "rounding_half_width": .005,
        "t_excess_beyond_rounding_interval": float(abs(baseline["t_statistics"][10] + 3.53) - .005),
        "conclusion": "Response and trough reproduce published rounding; t statistic narrowly fails. No tolerance widening. Original RATS databanks and XLSX yield the same discrepancy; SVD/QR numerical differences are orders of magnitude smaller.",
        "limitations": "RATS executable was not run. This establishes equivalence of original source inputs and independent linear algebra, not why the printed paper differs."
    }
    path = Path(__file__).with_name("rr2010_source_audit.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
