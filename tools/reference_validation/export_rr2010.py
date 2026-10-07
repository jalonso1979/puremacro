"""Convert the authentic RR2010 archive and export an independent OLS oracle.

Development only; requires pandas/openpyxl/statsmodels, never imports puremacro.
Run from outside the repository with an independently provisioned interpreter:

    python /path/to/tools/reference_validation/export_rr2010.py \
        --archive /tmp/DataSet.zip --output /path/to/puremacro/replication/data

No network request or original author program is executed. The inspected RATS
program's no-controls block is implemented using statsmodels OLS with QR and
conventional SSR/(n-k) covariance. Full coefficients and joint IRF covariance
are exported separately from the paper's rounded h=10 targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import struct
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.regression.linear_model import OLS


ARCHIVE_URL = "https://eml.berkeley.edu/~dromer/papers/DataSet.zip"
ARCHIVE_SHA256 = "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c"
WORKBOOK_MEMBER = "DataSet/Romer&RomerFiscalData.xlsx"
PROGRAM_MEMBER = "DataSet/RATS Programs and Databanks/EXOGERNR.RAT"
PAPER_URL = "https://eml.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def extract(workbook):
    """Join the two authenticated sheets on their original Roman-quarter dates."""
    sheets = pd.read_excel(io.BytesIO(workbook), sheet_name=None, header=None)
    full_dates = pd.period_range("1945Q1", "2007Q4", freq="Q-DEC")
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    tables = []
    extraction = {"sheets": [], "join": "one-to-one original quarterly date labels"}
    for sheet, row, columns in (("Tax Measures", 11, ["DEFICNR", "LONGRNR"]),
                                ("Other Variables", 18, ["GDP", "NOMGDP"])):
        raw = sheets[sheet]
        headers = [str(value).strip() for value in raw.iloc[row]]
        if any(headers.count(column) != 1 for column in columns):
            raise ValueError(f"Original {sheet} header does not match the inspected layout")
        selected = raw.iloc[row+1:].dropna(how="all")
        dates = []
        for label in selected.iloc[:, 0]:
            match = re.fullmatch(r"(\d{4})-(I|II|III|IV)", str(label).strip())
            if match is None:
                raise ValueError(f"Malformed original quarter label: {label!r}")
            dates.append(f"{match[1]}Q{roman[match[2]]}")
        index = pd.PeriodIndex(dates, freq="Q-DEC", name="quarter")
        if not index.equals(full_dates):
            raise ValueError(f"{sheet} dates are not the complete original 1945Q1--2007Q4 grid")
        table = selected.iloc[:, [headers.index(column) for column in columns]].copy()
        table.columns = columns
        table.index = index
        tables.append(table)
        extraction["sheets"].append({"sheet": sheet, "header_row_1based": row+1,
                                     "date_column": "A", "source_columns": columns,
                                     "source_rows": len(table)})
    merged = tables[0].join(tables[1], how="inner", validate="one_to_one")
    selected = merged.loc["1947Q1":"2007Q4"]
    expected = pd.period_range("1947Q1", "2007Q4", freq="Q-DEC")
    if not selected.index.equals(expected):
        raise ValueError("Merged source does not contain the exact 244-quarter input grid")
    output = selected[["GDP", "NOMGDP", "DEFICNR", "LONGRNR"]].astype(float)
    output.columns = ["gdp", "nomgdp", "defic", "longr"]
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Original workbook contains nonfinite selected values")
    output.insert(0, "quarter", output.index.astype(str))
    extraction.update({"rows": len(output), "selected_start": str(output.index[0]),
                       "selected_end": str(output.index[-1]),
                       "column_mapping": {"GDP": "gdp", "NOMGDP": "nomgdp",
                                          "DEFICNR": "defic", "LONGRNR": "longr"}})
    return output, extraction


def oracle(frame):
    """Independent array construction, statsmodels fit and explicit lag sums."""
    gdp, nominal = frame["gdp"].to_numpy(), frame["nomgdp"].to_numpy()
    deficit, long_run = frame["defic"].to_numpy(), frame["longr"].to_numpy()
    # These transformations follow EXOGERNR.RAT literally. Pre-sample lags
    # remain available because we construct all 244 quarters before slicing.
    tax = 100*deficit/nominal + 100*long_run/nominal
    outcome = np.array([100*(np.log(gdp[t])-np.log(gdp[t-1])) for t in range(12, 244)])
    design = np.array([[1.] + [tax[t-lag] for lag in range(13)] for t in range(12, 244)])
    fitted = OLS(outcome, design, hasconst=True).fit(method="qr", cov_type="nonrobust")
    beta = np.asarray(fitted.params)
    covariance = np.asarray(fitted.cov_params())
    response = np.array([sum(beta[1:h+2]) for h in range(13)])
    joint = np.array([[sum(covariance[i, j] for i in range(1, h+2)
                          for j in range(1, k+2)) for k in range(13)] for h in range(13)])
    se = np.sqrt(np.diag(joint))
    if fitted.nobs != 232 or fitted.df_resid != 218:
        raise ValueError("Independent fit disagrees with the original sample/degrees of freedom")
    if not all(np.isfinite(v).all() for v in (beta, covariance, response, joint, se)):
        raise ValueError("Independent fit produced nonfinite evidence")
    return {"coefficients": beta.tolist(), "coefficient_covariance": covariance.tolist(),
            "irf": response.tolist(), "irf_covariance": joint.tolist(),
            "standard_errors": se.tolist(), "t_statistics": (response/se).tolist(),
            "nobs": int(fitted.nobs), "df_resid": int(fitted.df_resid)}


def audit_databanks(archive, workbook_frame):
    """Cross-check the pinned archive's RATS4 inputs, including their last bits.

    This narrowly scoped reader validates the inspected archive layout; it is
    not a general RATS file reader. Data records contain two 32-bit links and
    31 IEEE float64 values. Directory records identify series and start dates.
    """
    output = workbook_frame.copy()
    records = []
    for stem, variable, column, year, nobs in (("GDP", "GDP", "gdp", 1947, 244),
            ("NOMGDP", "NOMGDP", "nomgdp", 1947, 244),
            ("TAXNR", "DEFIC", "defic", 1945, 252),
            ("TAXNR", "LONGR", "longr", 1945, 252)):
        member = f"DataSet/RATS Programs and Databanks/{stem}.DED"
        content = archive.read(member)
        if content[:4] != b"RAT4" or len(content) % 256:
            raise ValueError("Unexpected authenticated RATS databank layout")
        headers = [content[offset:offset+256] for offset in range(0, len(content), 256)
                   if content[offset+16:offset+32].decode("ascii", "replace").strip() == variable]
        if len(headers) != 1:
            raise ValueError(f"Expected exactly one RATS directory entry for {variable}")
        header = headers[0]
        if (struct.unpack_from("<H", header, 44)[0] != 4
                or struct.unpack_from("<H", header, 50)[0] != year
                or struct.unpack_from("<H", header, 52)[0] != 1
                or struct.unpack_from("<I", header, 56)[0] != nobs):
            raise ValueError(f"Unexpected RATS dates or frequency for {variable}")
        pointer = struct.unpack_from("<I", header, 12)[0]
        values, visited = [], set()
        previous = 0
        while pointer:
            if pointer in visited or not 1 <= pointer <= len(content)//256:
                raise ValueError("Invalid RATS data block chain")
            visited.add(pointer)
            block = content[(pointer-1)*256:pointer*256]
            back, forward, *numbers = struct.unpack("<II31d", block)
            if back != previous:
                raise ValueError("Invalid RATS data block backlink")
            previous, pointer = pointer, forward
            values.extend(numbers)
        if not nobs <= len(values) < nobs+31:
            raise ValueError("Invalid RATS data block count")
        selected = np.asarray(values[:nobs])[-244:]
        difference = float(np.max(np.abs(selected-workbook_frame[column].to_numpy())))
        if not np.isfinite(selected).all() or difference > 1e-11:
            raise ValueError(f"Workbook differs materially from RATS {variable}")
        output[column] = selected
        records.append({"member": member, "sha256": digest(content), "variable": variable,
                        "start": f"{year}Q1", "nobs": nobs,
                        "maximum_absolute_workbook_difference": difference})
    result = oracle(output)
    return {"series": records, "h10_response": result["irf"][10],
            "h10_t_statistic": result["t_statistics"][10],
            "finding": "RATS databank and workbook differ only at floating-point last bits; the published t-statistic rounding discrepancy persists"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(name == "puremacro" or name.startswith("puremacro.") for name in sys.modules):
        raise RuntimeError("Independent exporter must not import puremacro")
    archive_bytes = args.archive.read_bytes()
    if digest(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("Archive differs from the inspected original author download")
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        workbook = archive.read(WORKBOOK_MEMBER)
        program = archive.read(PROGRAM_MEMBER)
    frame, extraction = extract(workbook)
    csv = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    # Re-read exactly the stored decimal representation, never a lower-precision
    # staging table. The runtime and reference receive the same authored data.
    parsed = pd.read_csv(io.BytesIO(csv), float_precision="round_trip")
    reference = oracle(parsed)
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        databank_audit = audit_databanks(archive, parsed)
    published_comparison = {
        "rounding_half_width": .005,
        "h10_response": {"observed": reference["irf"][10], "published": -3.08,
                         "passed": bool(abs(reference["irf"][10]+3.08) < .005)},
        "h10_t_statistic": {"observed": reference["t_statistics"][10], "published": -3.53,
                            "passed": bool(abs(reference["t_statistics"][10]+3.53) < .005)},
        "trough_horizon": {"observed": int(np.argmin(reference["irf"])), "published": 10,
                           "passed": bool(np.argmin(reference["irf"]) == 10)},
        "interpretation": "Point estimate and trough replicate; the t-statistic lies just outside literal two-decimal rounding, retained as a discrepancy without a tolerance adjustment"}
    common = {"archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
              "workbook_member": WORKBOOK_MEMBER, "workbook_sha256": digest(workbook),
              "program_member": PROGRAM_MEMBER, "program_sha256": digest(program),
              "csv_sha256": digest(csv), "paper_url": PAPER_URL,
              "sample_start": "1950Q1", "sample_end": "2007Q4",
              "covariance_type": "conventional OLS, SSR/(n-k)",
              "coefficient_order": ["constant"]+[f"tax_lag_{lag}" for lag in range(13)],
              "irf_horizons": list(range(13)), "shock_size": 1.,
              "shock_unit": "percentage points of nominal GDP", "response_unit": "percent real GDP"}
    reference["metadata"] = {**common, "software": "statsmodels", "version": statsmodels.__version__,
                             "python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "fit_method": "OLS.fit(method='qr', cov_type='nonrobust')",
                             "runtime_imported_puremacro": False,
                             "repository_on_sys_path": str(Path(__file__).resolve().parents[2]) in sys.path,
                             "exporter_sha256": digest(Path(__file__).read_bytes()),
                             "generated_at": datetime.now(timezone.utc).isoformat(),
                             "published_comparison": published_comparison,
                             "original_program_databank_crosscheck": databank_audit,
                             "reference_role": "independent software reproduction of inspected author RATS no-controls block; not a live RATS run"}
    reference_bytes = (json.dumps(reference, indent=2, allow_nan=False)+"\n").encode()
    metadata = {**common, "source": "Romer and Romer (2010), original author replication archive",
                "is_synthetic": False, "extraction": extraction,
                "reference_sha256": digest(reference_bytes),
                "published_comparison": published_comparison,
                "original_program_databank_crosscheck": databank_audit,
                "author_regression_excerpt": "SMPL 1950:1 2007:4\nDOFOR TAXTYPE = EXOGER\nLINREG PCGDP1 /\n# CONSTANT TAXTYPE{0 TO 12}",
                "original_program_location": "The complete original program is retained in the authenticated author archive at program_member",
                "original_data_start": "1947Q1", "original_data_end": "2007Q4",
                "vintage": "Authors' original data; real GDP chain-type quantity index downloaded 2008-02-17",
                "transformations": ["Preserve original workbook numeric values at float64 round-trip precision",
                                    "Use nonretroactive deficit and long-run tax changes only",
                                    "Exogenous shock=100*DEFICNR/NOMGDP+100*LONGRNR/NOMGDP",
                                    "Dependent variable=100*quarterly log change in GDP",
                                    "Estimate constant+tax lags0--12 on 1950Q1--2007Q4;232obs,14regressors,218 residual df",
                                    "Cumulative response=sum lag coefficients; full covariance via all lag-covariance entries"],
                "published_targets": {"location": "p.781, equation (6) / Figure 4", "horizon": 10,
                                      "response": -3.08, "t_statistic": -3.53, "rounding_half_width": .005},
                "limitations": ["Conventional OLS inference reproduces original study rather than imposing HAC",
                                "Identification inherits narrative-shock exogeneity; replication is not independent proof of exogeneity",
                                "The paper's -3.53 t-statistic narrowly fails strict rounding; this is recorded rather than tolerance-adjusted",
                                "Frozen statsmodels reference uses original data and a separate estimator, not puremacro-generated numbers"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"rr2010_original.csv").write_bytes(csv)
    (args.output/"rr2010_statsmodels_reference.json").write_bytes(reference_bytes)
    (args.output/"rr2010_original_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"rows": len(frame), "nobs": reference["nobs"], "df_resid": reference["df_resid"],
                      "h10_response": reference["irf"][10], "h10_t": reference["t_statistics"][10],
                      "csv_sha256": metadata["csv_sha256"], "reference_sha256": metadata["reference_sha256"],
                      "extraction": extraction}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
