"""tools/build_icio_77c_11s.py -- the 77-country x 11-sector ICIO fixture from an OECD file.

puremacro's trade model runs on an (850, 1078) table: 77 countries x 11 composite
sectors of intermediate flows, 77 x 3 final-demand columns (C, I, Cx) and three
value rows (net taxes, labour, capital). The legacy fixture ``icio_77c_11s.npz``
is a bit-exact copy of the MATLAB ``data_77c_11s.mat`` that ``Agregar_11s.m``
built from a corrupted ``data_2020_SML.csv`` export (tokens with three or four
decimals lost their decimal point). This script is a Python port of that
MATLAB aggregation, verified against the legacy fixture (``verify-legacy``),
and builds the replacement fixture from a clean OECD ICIO 2023-edition release
(``build``).

The port exposes two indexing defects of ``Agregar_11s.m`` that are baked into
the legacy table in addition to the decimal corruption. In the block that
reassembles the table around the rest-of-world (ROW) country, the loop that
should fill ROW's final-demand sales to the other 76 countries (``data7fd``)
writes into the intermediate block ``data7`` instead, so those sales land in
the intermediate columns of the first countries while ROW's final-demand block
stays empty; and the loop that should fill the tax/value-added cells of ROW's
own final-demand columns (``data5Tfd``) writes into ROW's flow rows
(``data5fd``). ``faithful_legacy_bugs=True`` reproduces both so the port can be
checked against the legacy fixture; ``build`` uses the corrected aggregation.

Usage (from the repository root)::

    python tools/build_icio_77c_11s.py build --source /path/to/2020_SML.csv
    python tools/build_icio_77c_11s.py verify-legacy --source /path/to/data_2020_SML.csv
    python tools/build_icio_77c_11s.py describe [--fixture PATH]

``build`` writes ``puremacro/trade/_datafiles/icio_77c_11s_oecd2020.npz`` and
``MANIFEST_OECD2020.json`` (array digest, source MD5, recipe, totals). The source
file's MD5 must be a registered clean release (``puremacro.trade.mrio.OECD_ICIO_MD5``);
the corrupted export is refused except by ``verify-legacy``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puremacro.trade.mrio import (  # noqa: E402
    AGREGAR_11_CONCORDANCE,
    OECD_ICIO_MD5,
    OECD_KNOWN_CORRUPTED_MD5,
)

DATAFILES = ROOT / "puremacro" / "trade" / "_datafiles"
LEGACY_FIXTURE = DATAFILES / "icio_77c_11s.npz"
CLEAN_FIXTURE = DATAFILES / "icio_77c_11s_oecd2020.npz"
CLEAN_MANIFEST = DATAFILES / "MANIFEST_OECD2020.json"

NC, NS_RAW, NFD_RAW, NFD = 77, 45, 6, 3


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_oecd_sml(path: Path, *, allow_corrupted: bool = False) -> tuple[np.ndarray, dict]:
    """Read a ``{year}_SML.csv`` (labelled, 3,468 x 3,928) or the legacy headerless export.

    Returns the numeric matrix (label column dropped) and a provenance dict.
    """
    import pandas as pd

    md5 = file_md5(path)
    status = "clean"
    year = next((y for y, d in OECD_ICIO_MD5.items() if d == md5), None)
    if md5 in OECD_KNOWN_CORRUPTED_MD5:
        status = "corrupted"
        if not allow_corrupted:
            raise SystemExit(
                f"{path.name}: MD5 {md5} is the corrupted export "
                f"({OECD_KNOWN_CORRUPTED_MD5[md5]}); refusing to build a fixture from it."
            )
    elif year is None:
        status = "unregistered"
        print(f"warning: {path.name} MD5 {md5} is not a registered OECD ICIO 2023-edition file",
              file=sys.stderr)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    if head.startswith("V1") or head.startswith('"V1"') or head.startswith(","):
        df = pd.read_csv(path, header=0, index_col=0)
        raw = df.to_numpy(dtype=np.float64)
        labels = [str(c) for c in df.columns]
        countries = list(dict.fromkeys(c.split("_")[0] for c in labels[: NC * NS_RAW]))
    else:  # MATLAB load() style: headerless, tab-separated numbers
        raw = pd.read_csv(path, sep="\t", header=None, dtype=np.float64).to_numpy()
        countries = []
    if raw.shape[0] < NC * NS_RAW + 2 or raw.shape[1] < NC * NS_RAW + NFD_RAW * NC:
        raise SystemExit(f"{path.name}: unexpected shape {raw.shape}")
    prov = {
        "source_file": path.name, "source_md5": md5, "source_status": status,
        "source_year": year, "source_shape": list(raw.shape), "countries_in_file": countries,
    }
    return raw, prov


def aggregate_77c_11s(raw: np.ndarray, *, faithful_legacy_bugs: bool = False) -> np.ndarray:
    """Port of ``Agregar_11s.m``: (3468, 3928) native table -> (850, 1078) fixture."""
    nc, ns, nfd0, nfd = NC, NS_RAW, NFD_RAW, NFD
    n_ind = nc * ns
    data = raw[: n_ind + 2, : n_ind + nfd0 * nc].copy()   # flows + TLS + VA rows; 6 FD per country
    # 1. Six final-demand categories -> C (HFCE+NPISH+GGFC), I (GFCF+INVNT), Cx (DPABR).
    fd = np.zeros((n_ind + 2, nfd * nc))
    for k in range(nc):
        b = n_ind + nfd0 * k
        fd[:, nfd * k + 0] = data[:, b:b + 3].sum(axis=1)
        fd[:, nfd * k + 1] = data[:, b + 3:b + 5].sum(axis=1)
        fd[:, nfd * k + 2] = data[:, b + 5]
    data = np.hstack([data[:, :n_ind], fd])
    # 2. Merge industries 44 and 45 (T and U sections) -> 44 industries, rows then columns.
    rows = [np.vstack([data[k * 45:k * 45 + 43], data[k * 45 + 43:k * 45 + 45].sum(axis=0, keepdims=True)])
            for k in range(nc)]
    rows.append(data[n_ind:n_ind + 2])
    data2 = np.vstack(rows)
    cols = [np.hstack([data2[:, k * 45:k * 45 + 43], data2[:, k * 45 + 43:k * 45 + 45].sum(axis=1, keepdims=True)])
            for k in range(nc)]
    cols.append(data2[:, n_ind:n_ind + nfd * nc])
    data = np.hstack(cols)
    ns = 44
    n_ind = nc * ns
    # 3. The MATLAB "ue" reassembly. With ue_no = 1..76 and ue = 77 it is the identity, except
    #    for the two indexing defects reproduced here on request.
    if faithful_legacy_bugs:
        r0 = 76 * ns
        inter, fdb = data[:n_ind, :n_ind], data[:n_ind, n_ind:]
        tvi, tvf = data[n_ind:n_ind + 2, :n_ind], data[n_ind:n_ind + 2, n_ind:]
        data7 = inter[r0:r0 + ns, :76 * ns].copy()
        for k2 in range(76):
            for ij in range(nfd):
                data7[:, ij + nfd * k2] += fdb[r0:r0 + ns, ij + nfd * k2]   # ROW->others FD into flows
        data7fd = np.zeros((ns, 76 * nfd))                                  # ROW->others FD left empty
        data5fd = fdb[r0:r0 + ns, 76 * nfd:77 * nfd].copy()
        for ii in range(2):
            for ij in range(nfd):
                data5fd[ii, ij] += tvf[ii, 76 * nfd + ij]                   # ROW FD TLS/VA into flow rows
        data5Tfd = np.zeros((2, nfd))                                       # ROW FD TLS/VA cells left empty
        top = np.hstack([inter[:r0, :76 * ns], inter[:r0, r0:r0 + ns], fdb[:r0, :76 * nfd], fdb[:r0, 76 * nfd:]])
        mid = np.hstack([data7, inter[r0:r0 + ns, r0:r0 + ns], data7fd, data5fd])
        bot = np.hstack([tvi[:, :76 * ns], tvi[:, r0:r0 + ns], tvf[:, :76 * nfd], data5Tfd])
        data = np.vstack([top, mid, bot])
    # 4. 44 industries -> 11 Agregar groups, rows then columns.
    n_s = AGREGAR_11_CONCORDANCE[:ns]
    mns = max(AGREGAR_11_CONCORDANCE)
    data2 = np.zeros((mns * nc, data.shape[1]))
    for k in range(nc):
        for i in range(ns):
            data2[n_s[i] - 1 + mns * k] += data[i + ns * k]
    data2 = np.vstack([data2, data[n_ind:n_ind + 2]])
    data3 = np.zeros((mns * nc + 2, mns * nc))
    data4 = np.zeros((mns * nc + 2, nfd * nc))
    for k in range(nc):
        for i in range(ns):
            data3[:, n_s[i] - 1 + mns * k] += data2[:, i + ns * k]
        for i in range(nfd):
            data4[:, i + nfd * k] += data2[:, n_ind + i + nfd * k]
    data = np.hstack([data3, data4])
    ns = mns
    n_ind = nc * ns
    # 5. Net taxes of the intermediate columns as the residual sales - purchases - VA.
    sales = data[:n_ind].sum(axis=1)
    purchases = data[:n_ind, :n_ind].sum(axis=0)
    va = data[n_ind + 1, :n_ind]
    data[n_ind, :n_ind] = sales - purchases - va
    # 6. Value added split 2/3 labour, 1/3 capital.
    alpha = 1.0 / 3.0
    return np.vstack([data[:n_ind + 1], (1.0 - alpha) * data[n_ind + 1], alpha * data[n_ind + 1]])


def native_totals(raw: np.ndarray) -> dict:
    """World totals of the native 45-sector file (the aggregation must conserve them)."""
    n_ind = NC * NS_RAW
    va = float(raw[n_ind + 1, :n_ind].sum())
    tls = float(raw[n_ind, :n_ind].sum())
    output = float(raw[:n_ind, : n_ind + NFD_RAW * NC].sum())
    return {"world_value_added": va, "world_net_taxes_intermediate": tls, "world_gross_output": output}


def fixture_totals(M: np.ndarray) -> dict:
    n_ind = NC * 11
    va = M[n_ind + 1, :n_ind] + M[n_ind + 2, :n_ind]
    sales = M[:n_ind].sum(axis=1)
    return {
        "world_value_added": float(va.sum()),
        "world_net_taxes_intermediate": float(M[n_ind, :n_ind].sum()),
        "world_gross_output": float(sales.sum()),
        "negative_value_added_cells": int((va < 0).sum()),
        "zero_sales_nodes": int((sales == 0).sum()),
        "negative_intermediate_cells": int((M[:n_ind, :n_ind] < 0).sum()),
        "negative_final_demand_cells": int((M[:n_ind, n_ind:] < 0).sum()),
        "negative_net_tax_cells": int((M[n_ind, :n_ind] < 0).sum()),
        "min_value_added": float(va.min()),
    }


def array_sha256(M: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()).hexdigest()


def cmd_build(args: argparse.Namespace) -> int:
    src = Path(args.source)
    raw, prov = read_oecd_sml(src)
    M = aggregate_77c_11s(raw, faithful_legacy_bugs=False)
    out = Path(args.out)
    np.savez_compressed(out, data=M)
    manifest = {
        "file": out.name, "array": "data", "shape": list(M.shape), "dtype": "float64",
        "sha256_array": array_sha256(M), "sha256_file": hashlib.sha256(out.read_bytes()).hexdigest(),
        "built": _dt.date.today().isoformat(), "builder": "tools/build_icio_77c_11s.py build",
        "source": "OECD Inter-Country Input-Output (ICIO) tables, 2023 edition, "
                  f"{prov['source_year']} table ({prov['source_file']})",
        **prov,
        "recipe": [
            "six final-demand categories -> C (HFCE+NPISH+GGFC), I (GFCF+INVNT), Cx (DPABR)",
            "industries 44 and 45 (ISIC T, U) merged",
            "44 industries -> 11 groups with puremacro.trade.mrio.AGREGAR_11_CONCORDANCE",
            "net taxes of intermediate columns = sales - purchases - value added (residual)",
            "value added split 2/3 labour (row 849), 1/3 capital (row 850)",
        ],
        "legacy_defects_not_reproduced": [
            "Agregar_11s.m wrote ROW's final-demand sales to the other countries into intermediate "
            "columns (data7fd loop writes into data7) and left the ROW final-demand block empty",
            "Agregar_11s.m wrote the tax/VA cells of ROW's final-demand columns into ROW's flow rows "
            "(data5Tfd loop writes into data5fd) and left those cells empty",
        ],
        "source_totals": native_totals(raw),
        "fixture_totals": fixture_totals(M),
        "country_order": "identical to puremacro.trade.data.CANONICAL_COUNTRY_CODES (checked at build)",
        "publisher": "Organisation for Economic Co-operation and Development (OECD)",
        "source_url": "https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm",
        "license": "OECD terms of use -- reuse and redistribution permitted with attribution to the OECD",
        "attribution": "Work using this matrix should credit the OECD Inter-Country Input-Output tables. "
                       "The OECD is not affiliated with puremacro and does not endorse it.",
        "not_covered_by": "puremacro's MIT licence covers its code, not this third-party data. "
                          "See SOURCES.md in this directory.",
    }
    from puremacro.trade.data import CANONICAL_COUNTRY_CODES
    if prov["countries_in_file"] and list(CANONICAL_COUNTRY_CODES) != prov["countries_in_file"]:
        raise SystemExit("country order in the source file differs from CANONICAL_COUNTRY_CODES")
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) and {args.manifest}")
    for k, v in manifest["fixture_totals"].items():
        print(f"  {k}: {v}")
    st = manifest["source_totals"]
    ft = manifest["fixture_totals"]
    print(f"  conservation: VA {ft['world_value_added'] - st['world_value_added']:+.3e}, "
          f"output {ft['world_gross_output'] - st['world_gross_output']:+.3e}")
    return 0


def cmd_verify_legacy(args: argparse.Namespace) -> int:
    src = Path(args.source)
    raw, prov = read_oecd_sml(src, allow_corrupted=True)
    with np.load(args.fixture) as z:
        legacy = np.asarray(z["data"], dtype=np.float64)
    faithful = aggregate_77c_11s(raw, faithful_legacy_bugs=True)
    corrected = aggregate_77c_11s(raw, faithful_legacy_bugs=False)
    d = np.abs(faithful - legacy)
    ok = np.allclose(faithful, legacy, rtol=1e-12, atol=1e-6)
    print(f"source {src.name} md5={prov['source_md5']} ({prov['source_status']})")
    print(f"faithful port vs {Path(args.fixture).name}: max|diff|={d.max():.3e}, "
          f"bit-equal cells={np.mean(faithful == legacy):.6f}, allclose(rtol=1e-12, atol=1e-6)={ok}")
    dd = np.abs(corrected - faithful)
    print(f"legacy defects on this table: {int((dd > 0).sum())} cells differ between the corrected "
          f"and the faithful aggregation; {np.abs(corrected - faithful).sum() / 2:.4e} moved")
    return 0 if ok else 1


def cmd_describe(args: argparse.Namespace) -> int:
    with np.load(args.fixture) as z:
        M = np.asarray(z["data"], dtype=np.float64)
    print(f"{args.fixture}: shape={M.shape} sha256_array={array_sha256(M)}")
    for k, v in fixture_totals(M).items():
        print(f"  {k}: {v}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the clean fixture from an OECD {year}_SML.csv")
    b.add_argument("--source", required=True)
    b.add_argument("--out", default=str(CLEAN_FIXTURE))
    b.add_argument("--manifest", default=str(CLEAN_MANIFEST))
    v = sub.add_parser("verify-legacy", help="reproduce the legacy fixture from the corrupted export")
    v.add_argument("--source", required=True)
    v.add_argument("--fixture", default=str(LEGACY_FIXTURE))
    d = sub.add_parser("describe", help="print the totals of a fixture")
    d.add_argument("--fixture", default=str(CLEAN_FIXTURE))
    args = p.parse_args(argv)
    return {"build": cmd_build, "verify-legacy": cmd_verify_legacy, "describe": cmd_describe}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
