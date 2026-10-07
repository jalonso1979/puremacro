"""Native multi-regional input-output tables with provenance, readers, aggregation and coarse tariffs.

This module is the ingestion layer for published MRIO databases at their
native resolution. It records where every number came from (SHA-256 and MD5
of the source bytes, archive member CRCs, edition, units) and every
transformation applied afterwards, so that a calibration can be traced back
to the published file. A file is authenticated only when its digest matches a
registered or caller-supplied value; anything else is read with a
``RuntimeWarning`` and recorded as ``status="unknown_edition"``. It ports, with puremacro conventions, the readers and
the balanced-table contract of the IO research workspace
(``dynamic_model/native_data.py``, ``headlinePaper/rebuild/calibrate_databases_2019.py``
and ``headlinePaper/rebuild/vendor/puremacro/trade/corrected/{data,tariffs}.py``).

Shared data contract
--------------------
Cells are country-major: cell ``i = country_index * S + sector_index``. An
:class:`MRIOTable` carries

* ``Z`` ``(M, M)`` basic-price intermediate deliveries (rows sell, columns buy),
  as a dense array or a ``scipy.sparse.csr_matrix`` (EXIOBASE is read sparse);
* ``F`` ``(M, N, K)`` final deliveries by seller cell, destination country and
  category, with ``fd_codes`` a subset of ``("C", "G", "X", "V", "VAL")`` in
  that order: C = household + NPISH + government consumption, G = gross fixed
  capital formation, X = residents' purchases abroad, V = signed inventory
  changes, VAL = valuables;
* ``VA``, ``TLS``, ``output`` ``(M,)`` and ``TFD`` ``(N, K)``;
* optional factor detail (``production_taxes``, ``labor_compensation``,
  ``operating_surplus``) and a ``merchandise_mask`` over sectors;
* ``metadata`` with ``sources`` (a tuple of :class:`SourceRecord`) and
  ``transformations`` (a tuple of strings, empty for a raw table).

Conventions that differ across sources (quoted from the IO documentation)
-------------------------------------------------------------------------
* OECD ICIO: "Resident consumption with a separate DPABR basket; consumption
  includes government"; "INVNT is signed inventory change; no stock is
  observed"; "ICIO VA has no observed labor/capital split".
* FIGARO: "Published territorial consumption, including government; no
  invented tourism redistribution"; "P5M combines inventory changes and
  valuables; the source does not separate them"; "Row total derived from
  published transactions; no separate output column".
* EXIOBASE: "Published EXIOBASE consumption, including government; no separate
  OECD DPABR basket"; "Inventory changes and valuables are separate flows; no
  stock is observed"; "This 2019 release is a nowcast; no stronger measurement
  claim is implied".
* "active_mask means positive reported output; it is not an economic
  feasibility certificate." "No loader downloads missing data or generates a
  substitute fixture."

What is not here
----------------
No GTAP reader: the only HAR parser that decodes real GTAP files in the IO
workspace is the vendored HARPY package, which is GPL-3 licensed and cannot be
redistributed under puremacro's MIT license; the pure-Python reader in
``headlinePaper/load_gtap11.py`` only round-trips its own writer's output; and
the GTAP data are licensed. See ``docs/trade_mrio.md``.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
import zipfile
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._results import TradeCalibrationResult
from .data import CANONICAL_SECTOR_CODES, RAW_45_SECTOR_CODES, RawIOData

__all__ = [
    "AGG11_SECTOR_CODES",
    "AGG11_SECTOR_NAMES",
    "AGREGAR_11_CONCORDANCE",
    "CONCORDANCES",
    "CoarseTariffResult",
    "Concordance",
    "EXIOBASE_382_2019_IXI_SHA256",
    "EXIOBASE_ISO3",
    "FIGARO_2026ED_2019_SHA256",
    "FIGARO_ISO3",
    "FINAL_USE_CATEGORIES",
    "ISIC_SECTION_11",
    "ISIC_SECTION_11_CODES",
    "ISIC_SECTION_11_NAMES",
    "MRIOAccountingReport",
    "MRIOBuildReport",
    "MRIOIntegrityError",
    "MRIOTable",
    "OECD_ICIO_MD5",
    "OECD_KNOWN_CORRUPTED_MD5",
    "SourceRecord",
    "aggregate_mrio",
    "check_oecd_source",
    "coarse_tariff_rates",
    "goods_mask",
    "identify_source",
    "read_exiobase_native",
    "read_figaro_native",
    "read_oecd_native",
    "regularize_table",
    "to_calibration_matrix",
    "verify_zip_member",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MRIOIntegrityError(ValueError):
    """A source file, checksum, layout or accounting gate is not acceptable.

    Raised for a known-corrupted or mismatched checksum, labels that do not
    follow the official layout, non-finite or negative flows where the source
    forbids them, and failed acceptance gates in :func:`regularize_table` and
    :func:`aggregate_mrio`. It is a ``ValueError`` because the object at fault
    is the input data, not the numerical method.
    """


# ---------------------------------------------------------------------------
# Registries and constants
# ---------------------------------------------------------------------------

FINAL_USE_CATEGORIES: tuple[str, ...] = ("C", "G", "X", "V", "VAL")
"""Canonical final-use categories in storage order.

C = household, NPISH and government consumption; G = gross fixed capital
formation; X = residents' purchases abroad (OECD DPABR); V = signed inventory
changes; VAL = valuables. A table may carry any subset in this order.
"""

FINAL_USE_NAMES: dict[str, str] = {
    "C": "Consumption (households, NPISH and government)",
    "G": "Gross fixed capital formation",
    "X": "Residents' purchases abroad",
    "V": "Changes in inventories (signed)",
    "VAL": "Changes in valuables",
}

OECD_ICIO_MD5: dict[int, str] = {
    2019: "28cba31491177955445051d459053744",
    2020: "d3e0f4979d85d6c0bb7cf4c43e324287",
}
"""MD5 digests of the accepted ``{year}_SML.csv`` files of the OECD ICIO 2023 edition."""

OECD_ICIO_SHA256: dict[int, str] = {
    2019: "985e3141c3da4ad56ccc9dd0721eadf540b363d6929fdbc6d8fbf5417433ac0e",
}
"""SHA-256 of the accepted 2019 file (the puremacro OECD fixture manifest records the same digest)."""

OECD_KNOWN_CORRUPTED_MD5: dict[str, str] = {
    "d1b887aaafa54ab3f28fde78fcd21cdf": (
        "data_2020_SML.csv with lost decimal points: tokens with 3-4 decimals lost their "
        "decimal point and tokens below 0.001 became 0"
    ),
}
"""MD5 digests of OECD files that must never be read. The one entry is the file the legacy
77x11 pipeline (and therefore puremacro's bundled ``icio_77c_11s.npz``) was built from."""

FIGARO_2026ED_2019_SHA256 = "b8adf2b50e69c6bf728d5660b0cd33acee7139baaa12cf048210bd73f3be1b72"
"""SHA-256 of ``matrix_eu-ic-io_ind-by-ind_26ed_2019.csv`` (Eurostat FIGARO 2026 edition, 2019 table)."""

EXIOBASE_382_2019_IXI_SHA256 = "79cd85e2ef14666afe54cc99e60ba58263c520d1b6b4d4ee37875733c2dbb497"
"""SHA-256 of ``IOT_2019_ixi.zip`` (EXIOBASE 3.8.2, Zenodo record 5589597)."""

EXIOBASE_382_2019_IXI_MD5 = "8fd3eba498b194602bf75ffe99910990"
"""MD5 of ``IOT_2019_ixi.zip`` (EXIOBASE 3.8.2, Zenodo record 5589597)."""

OECD_SOURCE_URL = "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html"
FIGARO_SOURCE_URL = "https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables/database"
EXIOBASE_SOURCE_URL = "https://zenodo.org/records/5589597"

OECD_FD_CODES: tuple[str, ...] = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")
"""Final-use columns of the OECD SML layout, in file order."""

FIGARO_FD_CODES: tuple[str, ...] = ("P3_S13", "P3_S14", "P3_S15", "P51G", "P5M")
"""Final-use columns per country of the Eurostat FIGARO industry-by-industry CSV, in file order."""

FIGARO_TRAILING_ROWS: tuple[str, ...] = (
    "W2_D21X31", "W2_OP_RES", "W2_OP_NRES", "W2_D1", "W2_D29X39", "W2_B2A3G",
)
"""Trailing rows of the FIGARO CSV: net product taxes, residents' purchases abroad, purchases by
non-residents, compensation of employees, other net production taxes, gross operating surplus."""

FIGARO_NATIVE_64_SECTORS: tuple[str, ...] = (
    "A01", "A02", "A03", "B", "C10T12", "C13T15", "C16", "C17", "C18", "C19", "C20", "C21",
    "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_32", "C33", "D35",
    "E36", "E37T39", "F", "G45", "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58",
    "J59_60", "J61", "J62_63", "K64", "K65", "K66", "L", "M69_70", "M71", "M72", "M73",
    "M74_75", "N77", "N78", "N79", "N80T82", "O84", "P85", "Q86", "Q87_88", "R90T92", "R93",
    "S94", "S95", "S96", "T", "U",
)
"""The 64 NACE Rev. 2 industry codes exactly as spelled in the FIGARO 2026-edition file."""

EXIOBASE_FINAL_NAMES: tuple[str, ...] = (
    "Final consumption expenditure by households",
    "Final consumption expenditure by non-profit organisations serving households (NPISH)",
    "Final consumption expenditure by government",
    "Gross fixed capital formation",
    "Changes in inventories",
    "Changes in valuables",
    "Exports: Total (fob)",
)
"""Final-use categories of EXIOBASE 3 ``Y.txt`` per region, in file order (y01, y02.a, y02.b, y04, y05.a, y05.b, y06)."""

EXIOBASE_FACTOR_NAMES: tuple[str, ...] = (
    "Taxes less subsidies on products purchased: Total",
    "Other net taxes on production",
    "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
    "Operating surplus: Consumption of fixed capital",
    "Operating surplus: Rents on land",
    "Operating surplus: Royalties on resources",
    "Operating surplus: Remaining net operating surplus",
)
"""The nine monetary rows at the top of EXIOBASE ``satellite/F.txt`` (value added and product taxes)."""

_ISO3_PAIRS = (
    "AL:ALB AR:ARG AT:AUT AU:AUS BE:BEL BG:BGR BR:BRA CA:CAN CH:CHE CN:CHN "
    "CY:CYP CZ:CZE DE:DEU DK:DNK EE:EST ES:ESP FI:FIN FR:FRA GB:GBR GR:GRC "
    "HR:HRV HU:HUN ID:IDN IE:IRL IN:IND IT:ITA JP:JPN KR:KOR LT:LTU LU:LUX "
    "LV:LVA ME:MNE MK:MKD MT:MLT MX:MEX NL:NLD NO:NOR PL:POL PT:PRT RO:ROU "
    "RS:SRB RU:RUS SA:SAU SE:SWE SI:SVN SK:SVK TR:TUR US:USA ZA:ZAF TW:TWN"
).split()
_ISO3_2LETTER: dict[str, str] = dict(pair.split(":") for pair in _ISO3_PAIRS)

FIGARO_ISO3: dict[str, str] = {**_ISO3_2LETTER, "FIGW1": "ROW"}
"""FIGARO two-letter region codes to ISO-3 (``FIGW1``, the rest of the world, becomes ``ROW``)."""

EXIOBASE_ISO3: dict[str, str] = {
    **_ISO3_2LETTER,
    "WA": "ROW_ASIA_PACIFIC", "WL": "ROW_AMERICA", "WE": "ROW_EUROPE",
    "WF": "ROW_AFRICA", "WM": "ROW_MIDDLE_EAST",
}
"""EXIOBASE two-letter region codes to ISO-3; the five rest-of-world regions keep distinct names."""

AGREGAR_11_CONCORDANCE: tuple[int, ...] = (
    (1,) * 6 + (2,) * 16 + (3, 3, 4, 5) + (6,) * 5 + (7,) * 4 + (8, 8, 9, 9, 9, 10, 10, 11, 11, 11)
)
"""Group (1-based) of each of the 45 OECD ICIO industries in the 11-sector "Agregar" map used by
the original MATLAB pipeline. Group 1 is agriculture, mining AND food; group 3 is utilities."""

AGG11_SECTOR_CODES: tuple[str, ...] = (
    "AGRI_MIN_FOOD", "MANUF_EXFOOD", "UTILITIES", "CONSTRUCTION", "TRADE", "TRANSPORT",
    "ACCOM_ICT", "FIN_REALESTATE", "PROF_ADMIN_PUBADM", "EDU_HEALTH", "ARTS_OTHER",
)
"""Truthful codes of the eleven Agregar groups (the bundled ``CANONICAL_SECTOR_CODES`` name the
same positions AGRI, MINQ, MANU, ... which does not describe their content)."""

AGG11_SECTOR_NAMES: dict[str, str] = {
    "AGRI_MIN_FOOD": "Agriculture, mining and food (A01_02-C10T12)",
    "MANUF_EXFOOD": "Manufacturing excluding food (C13T15-C31T33)",
    "UTILITIES": "Utilities (D, E)",
    "CONSTRUCTION": "Construction (F)",
    "TRADE": "Wholesale and retail trade (G)",
    "TRANSPORT": "Transport (H49-H53)",
    "ACCOM_ICT": "Accommodation and ICT (I, J58T60-J62_63)",
    "FIN_REALESTATE": "Finance and real estate (K, L)",
    "PROF_ADMIN_PUBADM": "Professional, administrative and public administration (M, N, O)",
    "EDU_HEALTH": "Education and health (P, Q)",
    "ARTS_OTHER": "Arts and other services (R, S, T)",
}

ISIC_SECTION_11: tuple[int, ...] = (
    (1, 1) + (2, 2, 2) + (3,) * 17 + (4, 4, 5, 6) + (7,) * 5 + (11,) + (8, 8, 8) + (9, 9) + (11, 11)
    + (10, 10, 10) + (11, 11, 11)
)
"""Robustness map of the 45 OECD industries: A | B | C | D+E | F | G | H | J | K+L | O+P+Q | I+M+N+R+S+T."""

ISIC_SECTION_11_CODES: tuple[str, ...] = ("A", "B", "C", "DE", "F", "G", "H", "J", "KL", "OPQ", "REST")

ISIC_SECTION_11_NAMES: dict[str, str] = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "DE": "Utilities",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transport and storage",
    "J": "Information and communication",
    "KL": "Finance and real estate",
    "OPQ": "Public administration, education and health",
    "REST": "Accommodation, professional, administrative, arts and other services",
}

_OECD45_GOODS = frozenset(RAW_45_SECTOR_CODES[:22])
_AGG11_GOODS = frozenset({"AGRI_MIN_FOOD", "MANUF_EXFOOD"})
_ISIC11_GOODS = frozenset({"A", "B", "C"})
_CANONICAL11_LABEL_GOODS = frozenset({"AGRI", "MINQ", "MANU"})
_EXIOBASE_CODE = re.compile(r"^i\d{2}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _frozen(a: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _frozen_csr(Z: Any) -> sparse.csr_matrix:
    out = sparse.csr_matrix(Z, dtype=float, copy=True)
    out.sum_duplicates()
    out.eliminate_zeros()
    for arr in (out.data, out.indices, out.indptr):
        arr.flags.writeable = False
    return out


def _rowsum(Z: Any) -> np.ndarray:
    return np.asarray(Z.sum(axis=1)).ravel()


def _colsum(Z: Any) -> np.ndarray:
    return np.asarray(Z.sum(axis=0)).ravel()


def _dense(Z: Any) -> np.ndarray:
    return Z.toarray() if sparse.issparse(Z) else np.asarray(Z, dtype=float)


def _country_indicator(n: int, s: int) -> sparse.csr_matrix:
    m = n * s
    return sparse.csr_matrix((np.ones(m), (np.repeat(np.arange(n), s), np.arange(m))), shape=(n, m))


def _bilateral(Z: Any, F: np.ndarray, n: int, s: int) -> np.ndarray:
    """Country-by-country basic-price flows over all uses (rows export, columns import)."""
    if sparse.issparse(Z):
        P = _country_indicator(n, s)
        B = np.asarray((P @ Z @ P.T).todense(), dtype=float)
    else:
        B = np.asarray(Z, dtype=float).reshape(n, s, n, s).sum(axis=(1, 3))
    return B + F.sum(axis=2).reshape(n, s, n).sum(axis=1)


def _net_exports_raw(Z: Any, F: np.ndarray, n: int, s: int) -> tuple[np.ndarray, float]:
    B = _bilateral(Z, F, n, s)
    xn = B.sum(axis=1) - B.sum(axis=0)
    return xn, float(xn.sum())


def _hash_stream(fh: Any) -> tuple[str, str, int, int]:
    sha, md5, crc, n = hashlib.sha256(), hashlib.md5(), 0, 0
    for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
        sha.update(block)
        md5.update(block)
        crc = zlib.crc32(block, crc)
        n += len(block)
    return sha.hexdigest(), md5.hexdigest(), crc, n


def _two_column(rows: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field")


def _codes(seq: Sequence[str], what: str) -> tuple[str, ...]:
    codes = tuple(str(c) for c in seq)
    if not codes:
        raise ValueError(f"{what} registry cannot be empty")
    if len(set(codes)) != len(codes):
        raise ValueError(f"{what} registry must have unique labels")
    return codes


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRecord:
    """Identity of one source byte stream: path, size and SHA-256 / MD5 digests.

    ``archive_member`` and ``crc32`` are set when the bytes were a ZIP member
    or an extracted file checked against the archive's central directory
    (size and CRC-32). ``status`` is one of ``"hashed"`` (no expectation
    supplied), ``"verified"`` (matched the expected or registered digest, or
    the archive entry of an authenticated archive), ``"whitelisted"`` (an
    accepted OECD edition) or ``"unknown_edition"`` (hashed but not
    authenticated; a ``RuntimeWarning`` was issued).
    """

    path: str
    bytes: int
    sha256: str
    md5: str
    archive_member: str | None = None
    crc32: int | None = None
    status: str = "hashed"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation."""
        return {
            "path": self.path, "bytes": self.bytes, "sha256": self.sha256, "md5": self.md5,
            "archive_member": self.archive_member, "crc32": self.crc32,
            "status": self.status, "note": self.note,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column table of the record's fields."""
        return _two_column(list(self.to_dict().items()))

    def summary(self) -> str:
        """One line: file, status and the leading SHA-256 digits."""
        member = f" ({self.archive_member})" if self.archive_member else ""
        return f"{Path(self.path).name}{member}: {self.status}, {self.bytes} bytes, sha256 {self.sha256[:12]}"

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_dataframe`."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_dataframe`."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_dataframe`."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def identify_source(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
) -> SourceRecord:
    """Hash the actual bytes of a source file and refuse an absent or changed file.

    Digests are streamed in 4 MiB blocks. When an expectation is supplied and
    the digest differs, :class:`MRIOIntegrityError` is raised; nothing is
    downloaded or substituted.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Authenticated native source required: {path}")
    with path.open("rb") as fh:
        sha, md5, _, n = _hash_stream(fh)
    status = "hashed"
    if expected_sha256 is not None:
        if sha != str(expected_sha256).lower():
            raise MRIOIntegrityError(f"Source SHA-256 mismatch: {path} has {sha}, expected {expected_sha256}")
        status = "verified"
    if expected_md5 is not None:
        if md5 != str(expected_md5).lower():
            raise MRIOIntegrityError(f"Source MD5 mismatch: {path} has {md5}, expected {expected_md5}")
        status = "verified"
    return SourceRecord(path=str(path.resolve()), bytes=n, sha256=sha, md5=md5, status=status)


def verify_zip_member(
    archive: str | Path,
    member: str,
    extracted_path: str | Path | None = None,
) -> SourceRecord:
    """Hash a ZIP member and check its size and CRC-32 against the archive directory.

    With ``extracted_path`` the already extracted file is hashed and compared
    with the archive entry (the IO check for previously extracted EXIOBASE
    members); otherwise the member is streamed from the archive itself.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(member)
        except KeyError as exc:
            raise MRIOIntegrityError(f"{archive} has no member {member!r}") from exc
        if extracted_path is None:
            with bundle.open(info) as fh:
                sha, md5, crc, n = _hash_stream(fh)
            location = f"{archive.resolve()}::{member}"
        else:
            p = Path(extracted_path)
            if not p.is_file():
                raise FileNotFoundError(f"Extracted member not found: {p}")
            with p.open("rb") as fh:
                sha, md5, crc, n = _hash_stream(fh)
            location = str(p.resolve())
        if n != info.file_size:
            raise MRIOIntegrityError(f"{location}: {n} bytes, archive entry has {info.file_size}")
        if crc != info.CRC:
            raise MRIOIntegrityError(f"{location}: CRC-32 {crc} differs from the archive entry {info.CRC}")
    return SourceRecord(path=location, bytes=n, sha256=sha, md5=md5, archive_member=member,
                        crc32=int(info.CRC), status="verified")


def check_oecd_source(
    path: str | Path,
    *,
    year: int | None = None,
    whitelist: Mapping[int, str] | None = None,
    corrupted: Mapping[str, str] | None = None,
) -> SourceRecord:
    """Hash an OECD ICIO CSV (or the year member of a ZIP) against the edition registries.

    A digest in ``corrupted`` (default :data:`OECD_KNOWN_CORRUPTED_MD5`) raises
    :class:`MRIOIntegrityError` with the registered explanation. A digest in
    ``whitelist`` (default :data:`OECD_ICIO_MD5`) returns a record with status
    ``"whitelisted"``; when ``year`` is given it must be that file's year. Any
    other digest returns status ``"unknown_edition"`` and issues a
    ``RuntimeWarning``: the file is readable, but nothing here vouches for it.
    """
    whitelist = OECD_ICIO_MD5 if whitelist is None else dict(whitelist)
    corrupted = OECD_KNOWN_CORRUPTED_MD5 if corrupted is None else dict(corrupted)
    path = Path(path)
    if path.suffix.lower() == ".zip":
        if year is None:
            raise ValueError("year is required to select the CSV member of an OECD ZIP archive")
        with zipfile.ZipFile(path) as bundle:
            names = [n for n in bundle.namelist() if Path(n).name == f"{int(year)}_SML.csv"]
        if len(names) != 1:
            raise MRIOIntegrityError(f"Expected exactly one {int(year)}_SML.csv member in {path}")
        record = verify_zip_member(path, names[0])
    else:
        record = identify_source(path)
    if record.md5 in corrupted:
        raise MRIOIntegrityError(f"{path} is {corrupted[record.md5]} (md5 {record.md5}); it is refused")
    years = sorted(int(y) for y, digest in whitelist.items() if str(digest).lower() == record.md5)
    if years:
        if year is not None and int(year) not in years:
            raise MRIOIntegrityError(
                f"{path}: md5 {record.md5} is the whitelisted {years[0]} table, not the requested year {year}"
            )
        return replace(record, status="whitelisted", note=f"OECD ICIO 2023 edition, {years[0]}_SML.csv")
    warnings.warn(
        f"{path}: md5 {record.md5} is not a whitelisted OECD ICIO 2023-edition table; the file is read "
        "as an unauthenticated edition",
        RuntimeWarning, stacklevel=2,
    )
    return replace(record, status="unknown_edition", note="md5 not in the OECD whitelist")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MRIOAccountingReport:
    """Independent raw-account diagnostics of a table; nothing is rebalanced.

    Row gaps compare ``sum_j Z_ij + sum_{n,k} F_ink`` with ``output_i``; column
    gaps compare ``sum_i Z_ij + VA_j + TLS_j`` with ``output_j``. Scaled gaps
    divide by ``max(|output|, 1)`` (diagnostic units only). Counts of negative
    entries are reported, never clipped.
    """

    dataset: str
    n_countries: int
    n_sectors: int
    n_cells: int
    Z_nnz: int
    active_cells: int
    zero_output_cells: int
    negative_output_cells: int
    negative_consumption_entries: int
    negative_investment_entries: int
    negative_inventory_entries: int
    nonpositive_active_factor_VA: int
    unavailable_active_capital_shares: int
    row_output_max_absolute_gap: float
    row_output_max_scaled_gap: float
    column_output_max_absolute_gap: float
    column_output_max_scaled_gap: float
    row_column_max_absolute_gap: float
    world_reported_value_added: float
    world_product_taxes: float
    world_production_taxes: float
    final_category_totals: dict[str, float] = field(default_factory=dict)
    factor_decomposition_max_absolute_gap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation."""
        out = {k: v for k, v in self.__dict__.items() if k != "metadata"}
        out["final_category_totals"] = dict(self.final_category_totals)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column table of every diagnostic."""
        rows = [(k, v) for k, v in self.to_dict().items() if k != "final_category_totals"]
        rows += [(f"total_{c}", v) for c, v in self.final_category_totals.items()]
        return _two_column(rows)

    def summary(self) -> str:
        """One line: dimensions, active cells and the largest scaled identity gaps."""
        return (f"{self.dataset}: {self.n_countries} countries x {self.n_sectors} sectors, "
                f"{self.active_cells} active cells, {self.zero_output_cells} zero-output cells; "
                f"max scaled row gap {self.row_output_max_scaled_gap:.2e}, "
                f"max scaled column gap {self.column_output_max_scaled_gap:.2e}; "
                f"negative inventory entries {self.negative_inventory_entries}.")

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_dataframe`."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_dataframe`."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_dataframe`."""
        return df_to_typst(self.to_dataframe(), **kwargs)


@dataclass(frozen=True)
class MRIOBuildReport:
    """Provenance, transformations and gate values of one table build.

    Readers attach a report with the source digests and any rounding repairs;
    :func:`regularize_table` fills the phantom, floor, residual-tax and
    trade-balance fields; :func:`aggregate_mrio` records the concordance and
    the aggregated identity residual. ``gates`` maps each acceptance gate to
    ``{"value", "tolerance", "passed"}`` (plus ``"cells"``, the first offending
    cell labels, and ``"n_cells"`` where a gate is about individual cells) and
    ``passed`` is their conjunction; :attr:`failed_gates` names the failures.
    ``phantoms`` lists the zero-output cells eligible for a phantom (empty
    cells with output exactly zero); ``skipped_phantoms`` lists the inactive
    cells that were NOT given one because they have negative output or
    non-zero entries (only possible under ``strict=False``).
    """

    source: str
    sha256: str | None
    md5: str | None
    year: int | None
    edition: str
    units: str
    n_countries: int
    n_sectors: int
    row_identity_median_rel: float = 0.0
    n_zero_output: int = 0
    phantoms: tuple[str, ...] = ()
    floored: tuple[tuple[str, float, float], ...] = ()
    dtls_accounting_abs: float = 0.0
    dtls_floor_abs: float = 0.0
    dtls_accounting_max_share_of_output: float = 0.0
    world_abs_tls: float = 0.0
    xn0_raw_sum: float = 0.0
    world_gdp: float = 0.0
    inactive: str = "none"
    negative_fd_cells: dict[str, int] = field(default_factory=dict)
    concordance: str | None = None
    aggregated_identity_rel: float | None = None
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = True
    spectral_radius: tuple[float, float, float] | None = None
    rounding_adjustments: tuple[dict[str, Any], ...] = ()
    transformations: tuple[str, ...] = ()
    skipped_phantoms: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the gates whose ``passed`` flag is False."""
        return tuple(k for k, v in self.gates.items() if not v.get("passed", True))

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation."""
        return {
            "source": self.source, "sha256": self.sha256, "md5": self.md5, "year": self.year,
            "edition": self.edition, "units": self.units, "n_countries": self.n_countries,
            "n_sectors": self.n_sectors, "row_identity_median_rel": self.row_identity_median_rel,
            "n_zero_output": self.n_zero_output, "phantoms": list(self.phantoms),
            "skipped_phantoms": list(self.skipped_phantoms),
            "floored": [list(x) for x in self.floored],
            "dtls_accounting_abs": self.dtls_accounting_abs, "dtls_floor_abs": self.dtls_floor_abs,
            "dtls_accounting_max_share_of_output": self.dtls_accounting_max_share_of_output,
            "world_abs_tls": self.world_abs_tls, "xn0_raw_sum": self.xn0_raw_sum,
            "world_gdp": self.world_gdp, "inactive": self.inactive,
            "negative_fd_cells": dict(self.negative_fd_cells), "concordance": self.concordance,
            "aggregated_identity_rel": self.aggregated_identity_rel,
            "gates": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                      for k, v in self.gates.items()},
            "passed": self.passed, "failed_gates": list(self.failed_gates),
            "spectral_radius": list(self.spectral_radius) if self.spectral_radius else None,
            "rounding_adjustments": [dict(x) for x in self.rounding_adjustments],
            "transformations": list(self.transformations),
        }

    def to_dataframe(self, which: str = "summary") -> pd.DataFrame:
        """``"summary"``: two-column field table; ``"gates"``: one row per gate; ``"floored"``: floored cells."""
        if which == "summary":
            d = self.to_dict()
            rows = [(k, v) for k, v in d.items()
                    if k not in ("phantoms", "floored", "gates", "rounding_adjustments", "transformations",
                                 "negative_fd_cells", "spectral_radius", "skipped_phantoms", "failed_gates")]
            rows.append(("n_phantoms", len(self.phantoms)))
            rows.append(("n_skipped_phantoms", len(self.skipped_phantoms)))
            rows.append(("failed_gates", ", ".join(self.failed_gates)))
            rows.append(("n_floored", len(self.floored)))
            rows.append(("n_rounding_adjustments", len(self.rounding_adjustments)))
            rows += [(f"negative_{c}_cells", v) for c, v in self.negative_fd_cells.items()]
            if self.spectral_radius is not None:
                rows += list(zip(("spectral_radius", "collatz_wielandt_lower", "collatz_wielandt_upper"),
                                 self.spectral_radius))
            return _two_column(rows)
        if which == "gates":
            return pd.DataFrame(
                [(k, v["value"], v["tolerance"], v["passed"]) for k, v in self.gates.items()],
                columns=["Gate", "Value", "Tolerance", "Passed"],
            ).set_index("Gate")
        if which == "floored":
            return pd.DataFrame(list(self.floored), columns=["Cell", "VA_before", "VA_after"]).set_index("Cell")
        raise ValueError("which must be 'summary', 'gates' or 'floored'")

    def summary(self) -> str:
        """One line: dimensions, phantoms, floors, the residual-tax share and the gate verdict."""
        failed = f" (failed: {', '.join(self.failed_gates)})" if self.failed_gates else ""
        return (f"{self.edition} ({self.year}, {self.units}): {self.n_countries} countries x "
                f"{self.n_sectors} sectors; phantoms={len(self.phantoms)}; VA floors={len(self.floored)}; "
                f"median row identity={self.row_identity_median_rel:.2e}; "
                f"residual TLS max share of output={self.dtls_accounting_max_share_of_output:.2e}; "
                f"concordance={self.concordance}; gates passed={self.passed}{failed}")

    def to_markdown(self, which: str = "summary", **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_dataframe`."""
        return df_to_markdown(self.to_dataframe(which), **kwargs)

    def to_latex(self, which: str = "summary", **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_dataframe`."""
        return df_to_latex(self.to_dataframe(which), **kwargs)

    def to_typst(self, which: str = "summary", **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_dataframe`."""
        return df_to_typst(self.to_dataframe(which), **kwargs)


# ---------------------------------------------------------------------------
# Concordances and goods masks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concordance:
    """A many-to-one map from fine codes to coarse codes (sectors or regions).

    ``groups[k]`` is the 1-based coarse group of ``fine_codes[k]``. The 0/1
    matrix ``C = matrix`` has shape ``(S_fine, G)`` with ``C[k, groups[k]-1] = 1``,
    and ``projection(N)`` is ``P = I_N kron C^T`` of shape ``(N G, N S)``, so an
    exact aggregation is ``Z_G = P Z P^T``. Build instances with
    :meth:`build`, :meth:`from_mapping` or :meth:`identity`; every group must
    be non-empty.
    """

    name: str
    fine_codes: tuple[str, ...]
    coarse_codes: tuple[str, ...]
    groups: tuple[int, ...]
    coarse_names: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        name: str,
        fine_codes: Sequence[str],
        groups: Sequence[int],
        coarse_codes: Sequence[str] | None = None,
        coarse_names: Mapping[str, str] | None = None,
    ) -> "Concordance":
        """Validate 1-based group ids and construct the concordance."""
        fine = _codes(fine_codes, "fine")
        g = np.asarray(list(groups), dtype=int).ravel()
        if g.size != len(fine):
            raise ValueError(f"concordance has {g.size} entries for {len(fine)} fine codes")
        if g.size and g.min() < 1:
            raise ValueError("concordance group ids are 1-based")
        n_groups = int(g.max()) if g.size else 0
        if coarse_codes is None:
            coarse = tuple(f"G{k:02d}" for k in range(1, n_groups + 1))
        else:
            coarse = _codes(coarse_codes, "coarse")
        if len(coarse) != n_groups:
            raise ValueError(f"{len(coarse)} coarse codes for {n_groups} groups")
        counts = np.bincount(g - 1, minlength=n_groups)
        if np.any(counts == 0):
            empty = [coarse[k] for k in np.flatnonzero(counts == 0)]
            raise ValueError(f"concordance leaves coarse groups empty: {empty}")
        names = {str(k): str(v) for k, v in (coarse_names or {}).items()}
        return cls(name=str(name), fine_codes=fine, coarse_codes=coarse,
                   groups=tuple(int(x) for x in g), coarse_names=names)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        mapping: Mapping[str, str],
        *,
        fine_codes: Sequence[str] | None = None,
        coarse_codes: Sequence[str] | None = None,
        coarse_names: Mapping[str, str] | None = None,
    ) -> "Concordance":
        """Build from ``{fine_code: coarse_code}``; coarse order follows first appearance unless given."""
        fine = _codes(fine_codes if fine_codes is not None else list(mapping), "fine")
        missing = [c for c in fine if c not in mapping]
        if missing:
            raise ValueError(f"mapping lacks fine codes {missing[:5]}")
        coarse = (_codes(coarse_codes, "coarse") if coarse_codes is not None
                  else tuple(dict.fromkeys(str(mapping[c]) for c in fine)))
        unknown = sorted({str(mapping[c]) for c in fine} - set(coarse))
        if unknown:
            raise ValueError(f"mapping targets codes outside coarse_codes: {unknown[:5]}")
        groups = [coarse.index(str(mapping[c])) + 1 for c in fine]
        return cls.build(name, fine, groups, coarse, coarse_names)

    @classmethod
    def identity(cls, codes: Sequence[str], name: str = "identity") -> "Concordance":
        """The trivial concordance (every code is its own group)."""
        fine = _codes(codes, "fine")
        return cls.build(name, fine, range(1, len(fine) + 1), fine)

    @property
    def n_fine(self) -> int:
        """Number of fine codes ``S``."""
        return len(self.fine_codes)

    @property
    def n_coarse(self) -> int:
        """Number of coarse groups ``G``."""
        return len(self.coarse_codes)

    @property
    def matrix(self) -> np.ndarray:
        """0/1 matrix ``(S_fine, G)``."""
        C = np.zeros((self.n_fine, self.n_coarse))
        C[np.arange(self.n_fine), np.asarray(self.groups) - 1] = 1.0
        return C

    def projection(self, n_countries: int, *, sparse_matrix: bool = False) -> Any:
        """``I_N kron C^T`` of shape ``(N G, N S)``; sparse CSR when requested."""
        n = int(n_countries)
        if n < 1:
            raise ValueError("n_countries must be positive")
        if sparse_matrix:
            return sparse.kron(sparse.identity(n, format="csr"), sparse.csr_matrix(self.matrix.T), format="csr")
        return np.kron(np.eye(n), self.matrix.T)

    def members(self, coarse_code: str) -> tuple[str, ...]:
        """Fine codes that map into ``coarse_code``."""
        k = self.coarse_codes.index(coarse_code) + 1
        return tuple(c for c, g in zip(self.fine_codes, self.groups) if g == k)

    def group_of(self, fine_code: str) -> str:
        """Coarse code of a fine code."""
        return self.coarse_codes[self.groups[self.fine_codes.index(fine_code)] - 1]

    def compose(self, other: "Concordance") -> "Concordance":
        """``other o self``: map fine codes through this concordance, then through ``other``."""
        if tuple(other.fine_codes) != tuple(self.coarse_codes):
            raise ValueError("other.fine_codes must equal self.coarse_codes")
        groups = [other.groups[g - 1] for g in self.groups]
        return Concordance.build(f"{other.name} o {self.name}", self.fine_codes, groups,
                                 other.coarse_codes, other.coarse_names)

    def coarse_mask(self, fine_mask: Sequence[bool], rule: str = "all") -> np.ndarray:
        """Aggregate a boolean sector mask: ``"all"`` (every member), ``"any"`` or ``"majority"``."""
        mask = np.asarray(fine_mask, dtype=bool)
        if mask.shape != (self.n_fine,):
            raise ValueError("fine_mask must have one entry per fine code")
        C = self.matrix
        hits, sizes = mask.astype(float) @ C, C.sum(axis=0)
        if rule == "all":
            return hits == sizes
        if rule == "any":
            return hits > 0
        if rule == "majority":
            return hits * 2 > sizes
        raise ValueError("rule must be 'all', 'any' or 'majority'")

    def to_dataframe(self) -> pd.DataFrame:
        """One row per fine code with its coarse code and name."""
        return pd.DataFrame({
            "fine": self.fine_codes,
            "coarse": [self.coarse_codes[g - 1] for g in self.groups],
            "coarse_name": [self.coarse_names.get(self.coarse_codes[g - 1], "") for g in self.groups],
        }).set_index("fine")

    def summary(self) -> str:
        """One line: name, dimensions and the group sizes."""
        sizes = np.bincount(np.asarray(self.groups) - 1, minlength=self.n_coarse)
        return (f"concordance {self.name!r}: {self.n_fine} fine codes -> {self.n_coarse} groups "
                f"(sizes {', '.join(str(int(x)) for x in sizes)})")

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_dataframe`."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_dataframe`."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_dataframe`."""
        return df_to_typst(self.to_dataframe(), **kwargs)


CONCORDANCES: dict[str, Concordance] = {
    "agregar": Concordance.build("agregar", RAW_45_SECTOR_CODES, AGREGAR_11_CONCORDANCE,
                                 AGG11_SECTOR_CODES, AGG11_SECTOR_NAMES),
    "isic_section": Concordance.build("isic_section", RAW_45_SECTOR_CODES, ISIC_SECTION_11,
                                      ISIC_SECTION_11_CODES, ISIC_SECTION_11_NAMES),
}
"""Named 45-to-11 maps of the OECD ICIO industries: ``"agregar"`` (the bundled layout) and ``"isic_section"``."""


def _detect_classification(codes: tuple[str, ...]) -> str:
    s = set(codes)
    if s <= set(RAW_45_SECTOR_CODES):
        return "oecd45"
    if s <= set(AGG11_SECTOR_CODES):
        return "agregar11"
    if s <= set(ISIC_SECTION_11_CODES):
        return "isic_section11"
    if s <= set(FIGARO_NATIVE_64_SECTORS):
        return "figaro64"
    if all(_EXIOBASE_CODE.match(c) for c in codes):
        return "exiobase163"
    if s <= set(CANONICAL_SECTOR_CODES):
        raise ValueError(
            "CANONICAL_SECTOR_CODES label the eleven Agregar groups by position, not by content "
            "(position 1 is agriculture, mining and food; position 3 is utilities). Pass "
            "classification='agregar11' for the true groups of the bundled table or "
            "'canonical11' to read the labels literally."
        )
    raise ValueError(f"unknown sector registry; cannot identify goods for {sorted(s)[:5]} ...")


def goods_mask(sector_codes: Sequence[str], classification: str | None = None) -> np.ndarray:
    """Boolean mask of merchandise sectors (ISIC sections A-C) for a known registry.

    ``classification`` is detected from the codes when omitted: ``"oecd45"``
    (OECD ICIO industries: A, B and C codes), ``"agregar11"`` (groups
    AGRI_MIN_FOOD and MANUF_EXFOOD), ``"isic_section11"`` (A, B, C),
    ``"figaro64"`` (NACE A-C except C33, repair and installation, which "is
    not a merchandise shipment"), ``"exiobase163"`` (industries i01-i37 except
    i01.w manure treatment). ``"canonical11"`` reads puremacro's bundled labels
    literally (AGRI, MINQ, MANU) and must be requested explicitly because those
    labels do not describe the Agregar groups they sit on.

    The mask is decided label by label, so any order or subset of a registry
    gives the same answer per code. Under ``"agregar11"`` a bundled
    ``CANONICAL_SECTOR_CODES`` label stands for the Agregar group at its
    position in that registry (AGRI and MINQ, positions 1 and 2, are the two
    goods groups). A code outside the named registry raises ``ValueError``.
    """
    codes = tuple(str(c) for c in sector_codes)
    kind = classification or _detect_classification(codes)

    def known(registry: Any, name: str) -> None:
        unknown = [c for c in codes if c not in registry]
        if unknown:
            raise ValueError(f"codes outside the {name} registry: {unknown[:5]}")

    if kind == "oecd45":
        known(set(RAW_45_SECTOR_CODES), "oecd45")
        return np.array([c in _OECD45_GOODS for c in codes], dtype=bool)
    if kind == "agregar11":
        position = {c: k for k, c in enumerate(CANONICAL_SECTOR_CODES)}
        known(set(AGG11_SECTOR_CODES) | set(position), "agregar11")
        return np.array([position[c] < 2 if c in position else c in _AGG11_GOODS for c in codes], dtype=bool)
    if kind == "isic_section11":
        known(set(ISIC_SECTION_11_CODES), "isic_section11")
        return np.array([c in _ISIC11_GOODS for c in codes], dtype=bool)
    if kind == "figaro64":
        known(set(FIGARO_NATIVE_64_SECTORS), "figaro64")
        return np.array([c.startswith(("A", "B", "C")) and c != "C33" for c in codes], dtype=bool)
    if kind == "exiobase163":
        bad = [c for c in codes if not _EXIOBASE_CODE.match(c)]
        if bad:
            raise ValueError(f"not EXIOBASE industry codes: {bad[:5]}")
        return np.array([int(c[1:3]) <= 37 and not c.startswith("i01.w") for c in codes], dtype=bool)
    if kind == "canonical11":
        known(set(CANONICAL_SECTOR_CODES), "canonical11")
        return np.array([c in _CANONICAL11_LABEL_GOODS for c in codes], dtype=bool)
    raise ValueError(f"unknown classification {kind!r}")


def _resolve_concordance(
    spec: Any,
    *,
    fine_codes: tuple[str, ...],
    coarse_codes: Sequence[str] | None = None,
    what: str = "sector",
) -> Concordance:
    if isinstance(spec, Concordance):
        conc = spec
    elif isinstance(spec, str):
        if spec not in CONCORDANCES:
            raise ValueError(f"unknown concordance {spec!r}; known: {sorted(CONCORDANCES)}")
        conc = CONCORDANCES[spec]
    elif isinstance(spec, Mapping):
        conc = Concordance.from_mapping("custom", spec, fine_codes=fine_codes, coarse_codes=coarse_codes)
    else:
        conc = Concordance.build("custom", fine_codes, list(spec), coarse_codes)
    if tuple(conc.fine_codes) != tuple(fine_codes):
        raise ValueError(f"concordance {conc.name!r} is defined on other {what} codes than the table")
    return conc


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MRIOTable:
    """A multi-regional input-output table at native resolution with provenance.

    Construct with :meth:`from_arrays` (validating), :meth:`from_raw`,
    :meth:`from_trade_calibration` or a reader. All arrays are read-only
    copies; ``Z`` may be dense or CSR. A raw table has
    ``metadata["transformations"] == ()``; :func:`regularize_table` and
    :func:`aggregate_mrio` return new tables that append to that record.

    Attributes
    ----------
    Z : ndarray or csr_matrix, shape (M, M)
        Intermediate deliveries at basic prices; rows sell, columns buy.
    F : ndarray, shape (M, N, K)
        Final deliveries by seller cell, destination country and category.
    VA, TLS, output : ndarray, shape (M,)
        Value added (factor income plus production taxes as recorded), net
        taxes less subsidies on the buyer's intermediate purchases, and gross
        output (provider column where the source has one, else row totals).
    TFD : ndarray, shape (N, K)
        Net taxes less subsidies on final purchases.
    fd_codes : tuple of str
        Subset of :data:`FINAL_USE_CATEGORIES` in canonical order.
    production_taxes, labor_compensation, operating_surplus : ndarray or None
        Factor detail where the source has it (FIGARO, EXIOBASE); ``None`` for
        the OECD table, whose VA row has no observed split.
    merchandise_mask : ndarray of bool or None
        Goods sectors (ISIC A-C) when the registry is known.
    reference_country : str or None
        Country that absorbs the floating-point round-off of the world sum of
        trade balances in :meth:`net_exports` (default: the last country).
    """

    Z: Any
    F: np.ndarray
    VA: np.ndarray
    TLS: np.ndarray
    TFD: np.ndarray
    output: np.ndarray
    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    fd_codes: tuple[str, ...] = ("C", "G", "X", "V")
    production_taxes: np.ndarray | None = None
    labor_compensation: np.ndarray | None = None
    operating_surplus: np.ndarray | None = None
    merchandise_mask: np.ndarray | None = None
    reference_country: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        Z: Any,
        F: Any,
        VA: Any,
        TLS: Any,
        TFD: Any,
        *,
        country_codes: Sequence[str],
        sector_codes: Sequence[str],
        fd_codes: Sequence[str] = ("C", "G", "X", "V"),
        output: Any | None = None,
        production_taxes: Any | None = None,
        labor_compensation: Any | None = None,
        operating_surplus: Any | None = None,
        merchandise_mask: Any | None = None,
        reference_country: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict: bool = True,
    ) -> "MRIOTable":
        """Validate shapes, finiteness and sign conventions and build a table.

        With ``strict=True`` (the native-source contract) ``Z`` must be
        nonnegative and the G and X categories must be nonnegative; negative
        consumption, inventories, valuables, value added and taxes are real
        source data and are always kept. ``strict=False`` only checks shapes
        and finiteness. ``output`` defaults to the row totals.
        """
        ccodes = _codes(country_codes, "country")
        scodes = _codes(sector_codes, "sector")
        fcodes = tuple(str(c) for c in fd_codes)
        if not fcodes or len(set(fcodes)) != len(fcodes) or any(c not in FINAL_USE_CATEGORIES for c in fcodes):
            raise ValueError(f"fd_codes must be a non-empty subset of {FINAL_USE_CATEGORIES}")
        order = [FINAL_USE_CATEGORIES.index(c) for c in fcodes]
        if order != sorted(order):
            raise ValueError(f"fd_codes must follow the canonical order {FINAL_USE_CATEGORIES}")
        n, s, k = len(ccodes), len(scodes), len(fcodes)
        m = n * s
        if sparse.issparse(Z):
            Zf = _frozen_csr(Z)
            if Zf.shape != (m, m):
                raise ValueError(f"Z shape {Zf.shape} does not match {n} countries x {s} sectors")
            if not np.isfinite(Zf.data).all():
                raise MRIOIntegrityError("Z contains NaN or Inf")
            if strict and Zf.data.size and Zf.data.min() < 0:
                raise MRIOIntegrityError("intermediate deliveries must be nonnegative")
        else:
            Zf = _frozen(Z)
            if Zf.shape != (m, m):
                raise ValueError(f"Z shape {Zf.shape} does not match {n} countries x {s} sectors")
            if not np.isfinite(Zf).all():
                raise MRIOIntegrityError("Z contains NaN or Inf")
            if strict and Zf.size and Zf.min() < 0:
                raise MRIOIntegrityError("intermediate deliveries must be nonnegative")
        Ff = _frozen(F)
        if Ff.shape != (m, n, k):
            raise ValueError(f"F shape {Ff.shape} must be ({m}, {n}, {k})")
        arrays = {"VA": _frozen(VA).reshape(-1), "TLS": _frozen(TLS).reshape(-1)}
        for name, arr in arrays.items():
            if arr.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},)")
        TFDf = _frozen(TFD)
        if TFDf.shape != (n, k):
            raise ValueError(f"TFD shape {TFDf.shape} must be ({n}, {k})")
        for name, arr in (("F", Ff), ("VA", arrays["VA"]), ("TLS", arrays["TLS"]), ("TFD", TFDf)):
            if not np.isfinite(arr).all():
                raise MRIOIntegrityError(f"{name} contains NaN or Inf")
        if output is None:
            out = _frozen(_rowsum(Zf) + Ff.sum(axis=(1, 2)))
        else:
            out = _frozen(output).reshape(-1)
            if out.shape != (m,):
                raise ValueError(f"output must have shape ({m},)")
            if not np.isfinite(out).all():
                raise MRIOIntegrityError("output contains NaN or Inf")
        if strict:
            for code, label in (("G", "fixed investment"), ("X", "residents' purchases abroad")):
                if code in fcodes and Ff[:, :, fcodes.index(code)].min(initial=0.0) < 0:
                    raise MRIOIntegrityError(f"negative {label} requires strict=False and a separately specified model")
        optional: dict[str, np.ndarray | None] = {}
        for name, arr in (("production_taxes", production_taxes), ("labor_compensation", labor_compensation),
                          ("operating_surplus", operating_surplus)):
            if arr is None:
                optional[name] = None
                continue
            a = _frozen(arr).reshape(-1)
            if a.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},)")
            if not np.isfinite(a).all():
                raise MRIOIntegrityError(f"{name} contains NaN or Inf")
            optional[name] = a
        if (optional["labor_compensation"] is None) != (optional["operating_surplus"] is None):
            raise ValueError("labor_compensation and operating_surplus must be supplied together")
        if merchandise_mask is None:
            mask = None
        else:
            mask = np.asarray(merchandise_mask)
            if mask.shape != (s,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("merchandise_mask must be a boolean vector with one entry per sector")
            mask = _frozen(mask, dtype=bool)
        ref = ccodes[-1] if reference_country is None else str(reference_country)
        if ref not in ccodes:
            raise ValueError(f"reference_country {ref!r} is not in the country registry")
        meta = dict(metadata or {})
        meta.setdefault("sources", ())
        meta.setdefault("transformations", ())
        meta["sources"] = tuple(meta["sources"])
        meta["transformations"] = tuple(meta["transformations"])
        return cls(Z=Zf, F=Ff, VA=arrays["VA"], TLS=arrays["TLS"], TFD=TFDf, output=out,
                   country_codes=ccodes, sector_codes=scodes, fd_codes=fcodes,
                   production_taxes=optional["production_taxes"],
                   labor_compensation=optional["labor_compensation"],
                   operating_surplus=optional["operating_surplus"],
                   merchandise_mask=mask, reference_country=ref, metadata=meta)

    @classmethod
    def from_raw(
        cls,
        raw: RawIOData,
        *,
        negative_investment: str = "raise",
        fd_mapping: Mapping[str, str] | None = None,
        reference_country: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        merchandise_mask: Any | None = None,
    ) -> "MRIOTable":
        """Bridge from :class:`puremacro.trade.data.RawIOData`.

        Recognized final-use layouts: the six OECD codes (HFCE, NPISH, GGFC,
        GFCF, INVNT, DPABR -> C, G, X, V), puremacro's ``("C", "I", "Cx")``
        (for the bundled layout I contains signed inventories), or any subset
        of :data:`FINAL_USE_CATEGORIES`. Other layouts need ``fd_mapping``
        ``{native code: category}``; categories mapped from several codes are
        summed. In every layout a negative G cell follows
        ``negative_investment``: ``"raise"`` (default) or ``"to_inventory"``,
        which moves the negative part into V (added to an existing V
        category, or a new one when the layout has none). Value added given
        as ``(K_VA, M)`` rows is summed; two rows are read as labour and
        capital.
        """
        m, n = raw.M, raw.C
        cats = [str(c) for c in raw.fd_categories]
        F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(m, n, len(cats))
        tfd = raw.taxes_less_subsidies_fd
        TFD = (np.zeros((n, len(cats))) if tfd is None else np.asarray(tfd, dtype=float).reshape(n, len(cats)))
        components = {c: F[:, :, i] for i, c in enumerate(cats)}
        tax_components = {c: TFD[:, i] for i, c in enumerate(cats)}
        notes: list[str] = []
        if fd_mapping is not None:
            mapping = {str(k): str(v) for k, v in fd_mapping.items()}
        elif tuple(cats) == OECD_FD_CODES:
            mapping = {"HFCE": "C", "NPISH": "C", "GGFC": "C", "GFCF": "G", "INVNT": "V", "DPABR": "X"}
        elif tuple(cats) == ("C", "I", "Cx"):
            mapping = {"C": "C", "I": "G", "Cx": "X"}
        elif all(c in FINAL_USE_CATEGORIES for c in cats):
            mapping = {c: c for c in cats}
        else:
            raise ValueError(f"unrecognized final-use categories {cats}; pass fd_mapping")
        missing = [c for c in cats if c not in mapping]
        if missing:
            raise ValueError(f"fd_mapping lacks categories {missing}")
        targets = tuple(c for c in FINAL_USE_CATEGORIES if c in set(mapping.values()))
        Fn = np.zeros((m, n, len(targets)))
        TFDn = np.zeros((n, len(targets)))
        for i, c in enumerate(cats):
            j = targets.index(mapping[c])
            Fn[:, :, j] += F[:, :, i]
            TFDn[:, j] += TFD[:, i]
        if "G" in targets and Fn[:, :, targets.index("G")].min(initial=0.0) < 0:
            if negative_investment == "raise":
                raise MRIOIntegrityError(
                    "the investment category has negative cells (inventory drawdowns merged into it, or "
                    "negative fixed investment in the source); pass negative_investment='to_inventory' to "
                    "move them into the V category"
                )
            if negative_investment != "to_inventory":
                raise ValueError("negative_investment must be 'raise' or 'to_inventory'")
            g = targets.index("G")
            neg = np.minimum(Fn[:, :, g], 0.0)
            if "V" in targets:
                # V already holds inventories (for example a provider's Cx):
                # add the negative investment part to it.
                new_targets = targets
                F2, T2 = Fn.copy(), TFDn.copy()
            else:
                new_targets = tuple(c for c in FINAL_USE_CATEGORIES if c in set(targets) | {"V"})
                F2 = np.zeros((m, n, len(new_targets)))
                T2 = np.zeros((n, len(new_targets)))
                for j, c in enumerate(targets):
                    F2[:, :, new_targets.index(c)] = Fn[:, :, j]
                    T2[:, new_targets.index(c)] = TFDn[:, j]
            F2[:, :, new_targets.index("G")] -= neg
            F2[:, :, new_targets.index("V")] += neg
            notes.append(f"{int((neg < 0).sum())} negative investment cells moved into V")
            Fn, TFDn, targets = F2, T2, new_targets
        VA = np.asarray(raw.value_added, dtype=float)
        labor = capital = None
        if VA.ndim == 2:
            if VA.shape[0] == 2:
                labor, capital = VA[0].copy(), VA[1].copy()
            VA = VA.sum(axis=0)
        extra = dict(metadata or {})
        extra_transformations = tuple(extra.pop("transformations", ()))
        meta = {
            "dataset": str(raw.source), "year": int(raw.year), "units": str(raw.unit),
            "source_final_categories": tuple(cats), "final_use_mapping": dict(mapping),
            "final_use_components": {c: _frozen(v) for c, v in components.items()},
            "final_tax_components": {c: _frozen(v) for c, v in tax_components.items()},
            "raw_metadata": dict(raw.metadata),
        }
        meta.update(extra)
        meta["transformations"] = extra_transformations + tuple(notes)
        return cls.from_arrays(
            raw.intermediate_matrix, Fn, VA, raw.taxes_less_subsidies, TFDn,
            country_codes=raw.countries, sector_codes=raw.sectors, fd_codes=targets,
            output=raw.gross_output, labor_compensation=labor, operating_surplus=capital,
            merchandise_mask=merchandise_mask, reference_country=reference_country,
            metadata=meta, strict=True,
        )

    @classmethod
    def from_trade_calibration(
        cls,
        calib: TradeCalibrationResult,
        *,
        negative_investment: str = "raise",
        cx_category: str | None = None,
    ) -> "MRIOTable":
        """Bridge from puremacro's calibrated transaction table ``calib.data_calibra``.

        Rows ``0..M-1`` are ``[Z | F]`` with three final-use columns per
        country (C, I, Cx -> C, G, X), row ``M`` holds TLS and the final-use
        taxes, rows ``M+1`` and ``M+2`` labour and capital (the bundled 77x11
        table splits value added 2/3-1/3 by assumption). For the bundled and
        OECD layouts G contains signed inventories because the legacy layout
        merges INVNT into GFCF; the bundled table has three negative cells
        there. For the FIGARO, WIOD, Eora and EXIOBASE layouts I is GFCF only
        (``P51G``/``GFCF``) and inventories sit in Cx. Negative G cells follow
        ``negative_investment="raise"`` (default) or ``"to_inventory"``, which
        adds them to V. Production taxes are not observed (``None``).

        Cx becomes X (tariff-exempt residents' purchases abroad) only for the
        bundled layout (no ``calib.metadata["final_use_mapping"]``) or the
        OECD ``DPABR`` mapping. The FIGARO, WIOD and Eora loaders' Cx (signed
        changes in inventories and valuables) becomes V; EXIOBASE's Cx (which
        includes its export column) or an unrecognised mapping raises
        ``ValueError`` unless ``cx_category`` (``"X"``, ``"V"`` or ``"VAL"``)
        is given. The decision is recorded in ``metadata["final_use_bridge"]``.
        """
        from puremacro.trade.dynamic.accounts import _calibration_cx_category

        if not isinstance(calib, TradeCalibrationResult):
            raise TypeError("calib must be a TradeCalibrationResult")
        if calib.data_calibra is None:
            raise ValueError("calib.data_calibra is required")
        n, s, k = int(calib.n_countries), int(calib.n_sectors), int(calib.n_final_demand)
        if k != 3:
            raise ValueError("from_trade_calibration expects the three legacy final-use columns (C, I, Cx)")
        m = n * s
        D = np.asarray(calib.data_calibra, dtype=float)
        if D.shape != (m + 3, m + k * n):
            raise ValueError(f"data_calibra shape {D.shape} does not match ({m + 3}, {m + k * n})")
        ccodes = tuple(calib.country_codes) or tuple(f"C{i:02d}" for i in range(n))
        scodes = tuple(calib.sector_codes) or tuple(f"S{i:02d}" for i in range(s))
        cx_target, bridge_record = _calibration_cx_category(
            calib, cx_category, caller="MRIOTable.from_trade_calibration")
        source_mapping = bridge_record["final_use_mapping"]
        labor, capital = D[m + 1, :m], D[m + 2, :m]
        raw = RawIOData(
            countries=list(ccodes), sectors=list(scodes), intermediate_matrix=D[:m, :m],
            final_demand_matrix=D[:m, m:], value_added=np.vstack([labor, capital]),
            taxes_less_subsidies=D[m, :m], gross_output=D[:, :m].sum(axis=0),
            fd_categories=["C", "I", "Cx"], year=int(calib.metadata.get("year", 0) or 0),
            source=str(calib.metadata.get("source", "trade_calibration")),
            unit=str(calib.metadata.get("unit", "calibration value units")),
            taxes_less_subsidies_fd=D[m, m:], metadata={},
        )
        try:
            mask = goods_mask(scodes)
        except ValueError:
            mask = None
        meta = {
            "edition": "puremacro TradeCalibrationResult.data_calibra",
            "is_synthetic": bool(calib.metadata.get("is_synthetic", False)),
            "labor_share_assumption": calib.metadata.get("labor_share_assumption"),
            "consumption_convention": ("Legacy C, I, Cx layout: I merges GFCF and signed INVNT"
                                       if source_mapping is None else
                                       f"C, I, Cx layout condensed from {source_mapping}; Cx -> {cx_target}"),
            "final_use_bridge": bridge_record,
            "factor_share_provenance": {
                "kind": "calibration_rows",
                "note": "labour and capital rows of data_calibra; the bundled table uses a 2/3-1/3 split by assumption",
            },
            "production_taxes_observed": False,
            "sector_label_note": (None if mask is not None else
                                  "bundled CANONICAL_SECTOR_CODES are positional labels of the Agregar groups; "
                                  "merchandise_mask left unset, use goods_mask(codes, 'agregar11')"),
            "transformations": ("bridged from TradeCalibrationResult.data_calibra",),
        }
        return cls.from_raw(raw, negative_investment=negative_investment, metadata=meta, merchandise_mask=mask,
                            fd_mapping=None if cx_target == "X" else {"C": "C", "I": "G", "Cx": cx_target})

    # -- basic properties ---------------------------------------------------

    @property
    def n_countries(self) -> int:
        """Number of countries ``N``."""
        return len(self.country_codes)

    @property
    def n_sectors(self) -> int:
        """Number of sectors ``S``."""
        return len(self.sector_codes)

    @property
    def n_cells(self) -> int:
        """Number of country-sector cells ``M = N S``."""
        return self.n_countries * self.n_sectors

    @property
    def n_categories(self) -> int:
        """Number of final-use categories ``K``."""
        return len(self.fd_codes)

    @property
    def is_sparse(self) -> bool:
        """True when ``Z`` is stored as CSR."""
        return bool(sparse.issparse(self.Z))

    @property
    def Z_dense(self) -> np.ndarray:
        """``Z`` as a dense array (a conversion when the table is sparse)."""
        return _dense(self.Z)

    @property
    def cell_labels(self) -> tuple[str, ...]:
        """``{country}_{sector}`` labels in country-major cell order."""
        return tuple(f"{c}_{s}" for c in self.country_codes for s in self.sector_codes)

    @property
    def country_of_cell(self) -> np.ndarray:
        """Country index of every cell, ``(M,)``."""
        return np.repeat(np.arange(self.n_countries), self.n_sectors)

    @property
    def active_mask(self) -> np.ndarray:
        """Positive reported output; not an economic feasibility certificate."""
        return self.output > 0

    @property
    def final_demand(self) -> np.ndarray:
        """Total final deliveries by seller cell and destination, ``(M, N)``."""
        return self.F.sum(axis=2)

    @property
    def factor_VA(self) -> np.ndarray:
        """Labour plus operating surplus when observed; otherwise VA minus observed production taxes."""
        if self.labor_compensation is not None:
            return self.labor_compensation + self.operating_surplus
        if self.production_taxes is not None:
            return self.VA - self.production_taxes
        return np.array(self.VA, copy=True)

    @property
    def capital_share_available(self) -> np.ndarray:
        """Cells whose observed labour and surplus are nonnegative with positive factor value added."""
        if self.operating_surplus is None:
            return np.zeros(self.n_cells, dtype=bool)
        return (self.factor_VA > 0) & (self.labor_compensation >= 0) & (self.operating_surplus >= 0)

    @property
    def capital_share(self) -> np.ndarray:
        """Operating surplus over factor VA where available, NaN elsewhere (never forced into (0, 1))."""
        out = np.full(self.n_cells, np.nan)
        if self.operating_surplus is not None:
            np.divide(self.operating_surplus, self.factor_VA, out=out, where=self.capital_share_available)
        return out

    @property
    def goods(self) -> np.ndarray:
        """Merchandise mask over sectors; raises when the registry is not classified."""
        if self.merchandise_mask is None:
            raise ValueError("merchandise_mask is not set for this table; see goods_mask(...)")
        return self.merchandise_mask

    @property
    def dataset(self) -> str:
        """Dataset name from ``metadata['dataset']`` (``'arrays'`` when unset)."""
        return str(self.metadata.get("dataset", "arrays"))

    @property
    def year(self) -> int | None:
        """Reference year from the metadata, or None."""
        return self.metadata.get("year")

    @property
    def units(self) -> str:
        """Monetary units from the metadata."""
        return str(self.metadata.get("units", ""))

    @property
    def edition(self) -> str:
        """Source edition from the metadata."""
        return str(self.metadata.get("edition", ""))

    @property
    def sources(self) -> tuple[SourceRecord, ...]:
        """The :class:`SourceRecord` of every source byte stream read."""
        return tuple(self.metadata.get("sources", ()))

    @property
    def transformations(self) -> tuple[str, ...]:
        """Every transformation applied since the source read, in order."""
        return tuple(self.metadata.get("transformations", ()))

    @property
    def is_raw(self) -> bool:
        """True when no transformation has been applied since the source read."""
        return len(self.transformations) == 0

    @property
    def build_report(self) -> MRIOBuildReport | None:
        """The latest :class:`MRIOBuildReport` (reader, regularization or aggregation), or None."""
        return self.metadata.get("build_report")

    def category_index(self, code: str) -> int:
        """Position of a final-use category in ``fd_codes``; ``ValueError`` when absent."""
        try:
            return self.fd_codes.index(code)
        except ValueError as exc:
            raise ValueError(f"final-use category {code!r} is not in {self.fd_codes}") from exc

    def country_index(self, code: str) -> int:
        """Position of a country code; ``ValueError`` when absent."""
        try:
            return self.country_codes.index(code)
        except ValueError as exc:
            raise ValueError(f"country {code!r} is not in the table") from exc

    def sector_index(self, code: str) -> int:
        """Position of a sector code; ``ValueError`` when absent."""
        try:
            return self.sector_codes.index(code)
        except ValueError as exc:
            raise ValueError(f"sector {code!r} is not in the table") from exc

    def final_use(self, code: str) -> np.ndarray:
        """Deliveries of one category, ``(M, N)``; zeros when the table does not carry it."""
        if code not in self.fd_codes:
            return np.zeros((self.n_cells, self.n_countries))
        return self.F[:, :, self.category_index(code)]

    def with_metadata(self, **updates: Any) -> "MRIOTable":
        """Copy of the table with metadata keys replaced (arrays are shared)."""
        return replace(self, metadata={**self.metadata, **updates})

    # -- accounting ---------------------------------------------------------

    def row_sales(self) -> np.ndarray:
        """Row totals ``sum_j Z_ij + sum_{n,k} F_ink``, ``(M,)``."""
        return _rowsum(self.Z) + self.F.sum(axis=(1, 2))

    def column_outlays(self) -> np.ndarray:
        """Column totals ``sum_i Z_ij + VA_j + TLS_j``, ``(M,)``."""
        return _colsum(self.Z) + self.VA + self.TLS

    def bilateral_flows(self) -> np.ndarray:
        """Country-by-country basic-price flows over all uses, ``(N, N)``."""
        return _bilateral(self.Z, self.F, self.n_countries, self.n_sectors)

    def net_exports(self) -> np.ndarray:
        """Fob trade balances over all uses; the world round-off goes to ``reference_country``."""
        xn, _ = _net_exports_raw(self.Z, self.F, self.n_countries, self.n_sectors)
        xn[self.country_index(self.reference_country or self.country_codes[-1])] -= xn.sum()
        return xn

    def accounting_report(self) -> MRIOAccountingReport:
        """Recompute row and column identities independently; nothing is rebalanced."""
        row, col = self.row_sales(), self.column_outlays()
        scale = np.maximum(np.abs(self.output), 1.0)
        active = self.active_mask
        fva = self.factor_VA
        unavailable = int((~self.capital_share_available & active).sum())
        gap = None
        if self.labor_compensation is not None:
            prod = self.production_taxes if self.production_taxes is not None else 0.0
            gap = float(np.max(np.abs(self.VA - fva - prod), initial=0.0))
        nnz = int(self.Z.nnz) if self.is_sparse else int(np.count_nonzero(self.Z))
        return MRIOAccountingReport(
            dataset=self.dataset, n_countries=self.n_countries, n_sectors=self.n_sectors,
            n_cells=self.n_cells, Z_nnz=nnz, active_cells=int(active.sum()),
            zero_output_cells=int((self.output == 0).sum()),
            negative_output_cells=int((self.output < 0).sum()),
            negative_consumption_entries=int((self.final_use("C") < 0).sum()),
            negative_investment_entries=int((self.final_use("G") < 0).sum()),
            negative_inventory_entries=int((self.final_use("V") < 0).sum()),
            nonpositive_active_factor_VA=int(((fva <= 0) & active).sum()),
            unavailable_active_capital_shares=unavailable,
            row_output_max_absolute_gap=float(np.max(np.abs(row - self.output), initial=0.0)),
            row_output_max_scaled_gap=float(np.max(np.abs(row - self.output) / scale, initial=0.0)),
            column_output_max_absolute_gap=float(np.max(np.abs(col - self.output), initial=0.0)),
            column_output_max_scaled_gap=float(np.max(np.abs(col - self.output) / scale, initial=0.0)),
            row_column_max_absolute_gap=float(np.max(np.abs(row - col), initial=0.0)),
            world_reported_value_added=float(self.VA.sum()),
            world_product_taxes=float(self.TLS.sum() + self.TFD.sum()),
            world_production_taxes=float(self.production_taxes.sum()) if self.production_taxes is not None else 0.0,
            final_category_totals={c: float(self.F[:, :, i].sum()) for i, c in enumerate(self.fd_codes)},
            factor_decomposition_max_absolute_gap=gap,
        )

    # -- bridges ------------------------------------------------------------

    def to_raw(self, final_uses: str = "cix") -> RawIOData:
        """Bridge to :class:`puremacro.trade.data.RawIOData` (dense ``Z``).

        ``final_uses="cix"`` maps C, I = G + V + VAL, Cx = X so that
        ``package_mrio_to_calibration_result`` and the legacy solver apply the
        foreign-balance closure to investment; ``"native"`` keeps the table's
        own categories (the legacy calibration then treats index 1 as
        investment, whatever it is).
        """
        m, n = self.n_cells, self.n_countries
        if final_uses == "cix":
            cats = ["C", "I", "Cx"]
            F = np.stack([self.final_use("C"), self.final_use("G") + self.final_use("V") + self.final_use("VAL"),
                          self.final_use("X")], axis=2)
            tfd = np.zeros((n, 3))
            for code, j in (("C", 0), ("G", 1), ("V", 1), ("VAL", 1), ("X", 2)):
                if code in self.fd_codes:
                    tfd[:, j] += self.TFD[:, self.category_index(code)]
            mapping = {"C": ["C"], "I": [c for c in ("G", "V", "VAL") if c in self.fd_codes], "Cx": ["X"]}
        elif final_uses == "native":
            cats, F, tfd = list(self.fd_codes), np.array(self.F, copy=True), np.array(self.TFD, copy=True)
            mapping = {c: [c] for c in cats}
        else:
            raise ValueError("final_uses must be 'cix' or 'native'")
        return RawIOData(
            countries=list(self.country_codes), sectors=list(self.sector_codes),
            intermediate_matrix=np.array(self.Z_dense, copy=True), final_demand_matrix=F.reshape(m, n * len(cats)),
            value_added=np.array(self.VA, copy=True), taxes_less_subsidies=np.array(self.TLS, copy=True),
            gross_output=np.array(self.output, copy=True), fd_categories=cats,
            year=int(self.year or 0), source=self.dataset, unit=self.units,
            taxes_less_subsidies_fd=tfd.ravel(),
            metadata={"is_synthetic": bool(self.metadata.get("is_synthetic", False)),
                      "schema": "MRIOTable country-major cells",
                      "edition": self.edition, "final_demand_mapping": mapping,
                      "sources": [r.to_dict() for r in self.sources],
                      "transformations": list(self.transformations),
                      "inventory_treatment": ("signed V (and VAL) added to G; no clipping" if final_uses == "cix"
                                              else "native categories kept")},
        )

    # -- presentation -------------------------------------------------------

    def to_dataframe(self, which: str = "summary") -> pd.DataFrame:
        """``"summary"`` (provenance and dimensions), ``"countries"`` (per-country accounts),
        ``"categories"`` (final-use totals), ``"sources"`` (digests)."""
        if which == "summary":
            rep = self.accounting_report()
            rows = [
                ("dataset", self.dataset), ("year", self.year), ("edition", self.edition),
                ("units", self.units), ("countries", self.n_countries), ("sectors", self.n_sectors),
                ("cells", self.n_cells), ("final_use_categories", ", ".join(self.fd_codes)),
                ("storage", "csr" if self.is_sparse else "dense"), ("active_cells", rep.active_cells),
                ("zero_output_cells", rep.zero_output_cells), ("world_value_added", rep.world_reported_value_added),
                ("world_output", float(self.output.sum())), ("n_sources", len(self.sources)),
                ("transformations", len(self.transformations)),
                ("reference_country", self.reference_country),
            ]
            return _two_column(rows)
        if which == "countries":
            n, s = self.n_countries, self.n_sectors
            xn = self.net_exports()
            data = {"output": self.output.reshape(n, s).sum(axis=1), "value_added": self.VA.reshape(n, s).sum(axis=1),
                    "TLS": self.TLS.reshape(n, s).sum(axis=1)}
            for i, c in enumerate(self.fd_codes):
                data[f"{c}_expenditure"] = self.F[:, :, i].sum(axis=0) + self.TFD[:, i]
            data["net_exports"] = xn
            return pd.DataFrame(data, index=pd.Index(self.country_codes, name="country"))
        if which == "categories":
            rows = [(c, FINAL_USE_NAMES[c], float(self.F[:, :, i].sum()), float(self.TFD[:, i].sum()))
                    for i, c in enumerate(self.fd_codes)]
            return pd.DataFrame(rows, columns=["Category", "Meaning", "Basic_price_total", "Net_taxes"]).set_index("Category")
        if which == "sources":
            return pd.DataFrame([r.to_dict() for r in self.sources] or [{"path": None, "bytes": None, "sha256": None,
                                                                          "md5": None, "archive_member": None,
                                                                          "crc32": None, "status": None, "note": None}])
        raise ValueError("which must be 'summary', 'countries', 'categories' or 'sources'")

    def summary(self) -> str:
        """One line: dataset, dimensions, categories, units, source digest and the number of transformations."""
        rep = self.accounting_report()
        digest = self.sources[0].sha256[:12] if self.sources else "none"
        return (f"MRIO table {self.dataset} {self.year or ''}: {self.n_countries} countries x {self.n_sectors} "
                f"sectors, categories {'/'.join(self.fd_codes)}, {rep.active_cells} active cells; "
                f"{self.units}; source sha256 {digest}; transformations={len(self.transformations)}")

    def to_markdown(self, which: str = "summary", **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_dataframe`."""
        return df_to_markdown(self.to_dataframe(which), **kwargs)

    def to_latex(self, which: str = "summary", **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_dataframe`."""
        return df_to_latex(self.to_dataframe(which), **kwargs)

    def to_typst(self, which: str = "summary", **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_dataframe`."""
        return df_to_typst(self.to_dataframe(which), **kwargs)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _reader_report(source: SourceRecord | None, *, year: int | None, edition: str, units: str,
                   n: int, s: int, adjustments: Sequence[dict[str, Any]], transformations: Sequence[str],
                   path: str) -> MRIOBuildReport:
    return MRIOBuildReport(
        source=path, sha256=source.sha256 if source else None, md5=source.md5 if source else None,
        year=year, edition=edition, units=units, n_countries=n, n_sectors=s,
        rounding_adjustments=tuple(dict(a) for a in adjustments), transformations=tuple(transformations),
    )


def read_oecd_native(
    path: str | Path,
    year: int,
    *,
    check: bool = True,
    whitelist: Mapping[int, str] | None = None,
    corrupted: Mapping[str, str] | None = None,
    reference_country: str = "ROW",
) -> MRIOTable:
    """Read an OECD ICIO 2023-edition ``{year}_SML.csv`` (or the year member of the ZIP).

    Wraps :func:`puremacro.trade._oecd_icio.read_native` (labelled parse of the
    77 x 45 layout with the TLS/VA/OUT margins) and maps the six final uses to
    C = HFCE + NPISH + GGFC, G = GFCF, X = DPABR and V = INVNT; there is no VAL
    category. ``output`` is the provider OUT column. With ``check=True`` the
    file is hashed through :func:`check_oecd_source`, which refuses the
    known-corrupted digest and warns on an unknown edition. The original six
    components stay in ``metadata["final_use_components"]``. No labour/capital
    split is observed in ICIO, so the factor fields are ``None``.
    """
    from ._oecd_icio import read_native

    path = Path(path)
    if check:
        record = check_oecd_source(path, year=int(year), whitelist=whitelist, corrupted=corrupted)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as bundle:
            names = [n for n in bundle.namelist() if Path(n).name == f"{int(year)}_SML.csv"]
        if len(names) != 1:
            raise MRIOIntegrityError(f"Expected exactly one {int(year)}_SML.csv member in {path}")
        record = verify_zip_member(path, names[0])
    else:
        record = identify_source(path)
    raw = read_native(path, int(year))
    ccodes = tuple(raw.countries)
    ref = reference_country if reference_country in ccodes else ccodes[-1]
    meta = {
        "dataset": "oecd_icio", "year": int(year), "units": "USD million, current prices",
        "edition": "OECD ICIO 2023 edition", "source_url": OECD_SOURCE_URL,
        "sources": (record,), "transformations": (), "whitelist_status": record.status,
        "consumption_convention": "Resident consumption with a separate DPABR basket; consumption includes government",
        "inventory_convention": "INVNT is signed inventory change; no stock is observed",
        "output_convention": "provider OUT column; row totals differ by published rounding",
        "production_taxes_observed": False,
        "factor_share_provenance": {"kind": "unavailable", "description": "ICIO VA has no observed labor/capital split"},
        "provider_accounting": dict(raw.metadata.get("provider_accounting", {})),
    }
    table = MRIOTable.from_raw(raw, reference_country=ref, metadata=meta,
                               merchandise_mask=goods_mask(raw.sectors, "oecd45"))
    report = _reader_report(record, year=int(year), edition=meta["edition"], units=meta["units"],
                            n=table.n_countries, s=table.n_sectors, adjustments=(), transformations=(),
                            path=str(path))
    return table.with_metadata(build_report=report)


def _figaro_rounding_repair(Z: np.ndarray, F: np.ndarray, VA: np.ndarray, TLS: np.ndarray,
                            labels: Sequence[str], countries: Sequence[str]) -> list[dict[str, Any]]:
    """The two bounded repairs of ``calibrate_databases_2019.figaro`` (in place)."""
    adjustments: list[dict[str, Any]] = []
    y = Z.sum(axis=1) + F.sum(axis=(1, 2))
    # Source is rounded to 0.001 million EUR. Only truly empty production nodes
    # with at most one rounding unit of negative output are zeroed.
    for i in np.flatnonzero(y < 0):
        mass = float(np.abs(Z[i]).sum() + np.abs(Z[:, i]).sum() + np.abs(F[i]).sum() + abs(VA[i]) + abs(TLS[i]))
        if y[i] < -0.001000001 or mass > 0.002000001:
            raise MRIOIntegrityError(f"Non-rounding negative output: {labels[i]}, {y[i]}")
        adjustments.append({"cell": labels[i], "reason": "negative empty-cell rounding",
                            "output": float(y[i]), "absolute_input_mass": mass})
        Z[i] = 0.0
        Z[:, i] = 0.0
        F[i] = 0.0
        VA[i] = 0.0
        TLS[i] = 0.0
    # Move negative consumption rounding to inventories, preserving every
    # output row and the country's total expenditure exactly.
    for i, k in np.argwhere(F[:, :, 0] < 0):
        value = float(F[i, k, 0])
        if value < -0.001000001:
            raise MRIOIntegrityError(f"Material negative consumption cannot be regularized: {labels[i]} -> {countries[k]}")
        F[i, k, 3] += value
        F[i, k, 0] = 0.0
        adjustments.append({"cell": labels[i], "destination": countries[k],
                            "reason": "rounding reclassified from C to V", "amount": value})
    return adjustments


def read_figaro_native(
    path: str | Path,
    year: int = 2019,
    *,
    expected_sha256: str | None = None,
    rounding_repair: bool = True,
    sparse_matrix: bool = False,
) -> MRIOTable:
    """Read the official Eurostat FIGARO industry-by-industry CSV (``matrix_eu-ic-io_ind-by-ind_*.csv``).

    Layout (label-driven, dimensions derived from the labels): a ``rowLabels``
    header, ``M = N x S`` cell columns ``{region}_{industry}`` in Cartesian
    order, five final-use columns per region (P3_S13, P3_S14, P3_S15, P51G,
    P5M), and six trailing rows W2_D21X31, W2_OP_RES, W2_OP_NRES, W2_D1,
    W2_D29X39, W2_B2A3G. The table carries ``fd_codes = ("C", "G", "X", "V")``
    with C = P3_S13 + P3_S14 + P3_S15 (government included), G = P51G, X
    identically zero (FIGARO records territorial consumption and has no
    residents-abroad basket; the W2_OP_RES / W2_OP_NRES totals are kept in
    metadata) and V = P5M, which "combines inventory changes and valuables;
    the source does not separate them". ``VA = W2_D1 + W2_D29X39 + W2_B2A3G``
    with the three components kept as labour, production taxes and gross
    operating surplus ("gross operating surplus and mixed income"); ``TLS`` is
    the W2_D21X31 cell row and ``TFD`` its final-use columns condensed the
    same way. Output is the row total ("no separate output column"). Region
    codes are mapped to ISO-3 through :data:`FIGARO_ISO3`.

    ``rounding_repair`` applies the two bounded repairs of the IO reader and
    records them: an empty cell whose output is one rounding unit negative
    (``-0.001 <= y < 0``, absolute input mass ``<= 0.002``) is zeroed, and a
    consumption entry in ``[-0.001, 0)`` is moved to V so that row totals and
    country expenditure are unchanged. Anything larger raises.

    The file is hashed; ``expected_sha256`` is enforced when given, otherwise
    the digest is compared with :data:`FIGARO_2026ED_2019_SHA256` (the
    registered 2026-edition 2019 file) and a ``RuntimeWarning`` flags an
    unauthenticated edition. The registered file read with ``year != 2019``
    raises :class:`MRIOIntegrityError`, as :func:`check_oecd_source` does for
    the OECD tables.
    """
    path = Path(path)
    record = identify_source(path, expected_sha256=expected_sha256)
    if record.sha256 == FIGARO_2026ED_2019_SHA256 and int(year) != 2019:
        raise MRIOIntegrityError(f"{path}: sha256 {record.sha256} is the registered FIGARO 2026-edition 2019 table, "
                                 f"not the requested year {int(year)}")
    if expected_sha256 is None:
        if record.sha256 == FIGARO_2026ED_2019_SHA256:
            record = replace(record, status="verified", note="Eurostat FIGARO 2026 edition, 2019 table")
        else:
            warnings.warn(f"{path}: sha256 {record.sha256} is not the registered FIGARO 2026-edition 2019 file; "
                          "the file is read as an unauthenticated edition", RuntimeWarning, stacklevel=2)
            record = replace(record, status="unknown_edition")
    df = pd.read_csv(path, index_col=0)
    rows = [str(x) for x in df.index]
    cols = [str(x) for x in df.columns]
    trailing = list(FIGARO_TRAILING_ROWS)
    if len(rows) <= len(trailing) or rows[-len(trailing):] != trailing:
        raise MRIOIntegrityError(f"FIGARO source must end with the rows {trailing}")
    labels = rows[:-len(trailing)]
    m = len(labels)
    if cols[:m] != labels:
        raise MRIOIntegrityError("FIGARO columns do not repeat the row labels of the cells")
    countries_src = tuple(dict.fromkeys(x.split("_", 1)[0] for x in labels))
    n = len(countries_src)
    if m % n:
        raise MRIOIntegrityError("FIGARO cells are not a Cartesian region x industry grid")
    s = m // n
    sectors = tuple(x.split("_", 1)[1] for x in labels[:s])
    if labels != [f"{c}_{x}" for c in countries_src for x in sectors]:
        raise MRIOIntegrityError("Source country-sector labels are not in their declared Cartesian order")
    fd_cols = [f"{c}_{f}" for c in countries_src for f in FIGARO_FD_CODES]
    if cols[m:] != fd_cols:
        raise MRIOIntegrityError("FIGARO final-use columns do not follow the P3_S13/P3_S14/P3_S15/P51G/P5M layout")
    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise MRIOIntegrityError("FIGARO source contains NaN or Inf")
    Z = values[:m, :m].copy()
    rawF = values[:m, m:].reshape(m, n, 5)
    W = {name: values[m + i] for i, name in enumerate(trailing)}
    TLS = W["W2_D21X31"][:m].copy()
    rawTFD = W["W2_D21X31"][m:].reshape(n, 5)
    lab, prodtax, cap = W["W2_D1"][:m].copy(), W["W2_D29X39"][:m].copy(), W["W2_B2A3G"][:m].copy()
    F = np.stack([rawF[:, :, :3].sum(axis=2), rawF[:, :, 3], np.zeros((m, n)), rawF[:, :, 4]], axis=2)
    TFD = np.column_stack([rawTFD[:, :3].sum(axis=1), rawTFD[:, 3], np.zeros(n), rawTFD[:, 4]])
    VA = lab + prodtax + cap
    adjustments: list[dict[str, Any]] = []
    transformations: tuple[str, ...] = ()
    if rounding_repair:
        adjustments = _figaro_rounding_repair(Z, F, VA, TLS, labels, countries_src)
        if adjustments:
            transformations = (f"figaro rounding repair: {len(adjustments)} bounded adjustments recorded in "
                               "metadata['rounding_adjustments']",)
    unmapped = [c for c in countries_src if c not in FIGARO_ISO3]
    if unmapped:
        warnings.warn(f"FIGARO region codes without an ISO-3 mapping are kept verbatim: {unmapped}",
                      RuntimeWarning, stacklevel=2)
    ccodes = tuple(FIGARO_ISO3.get(c, c) for c in countries_src)
    meta = {
        "dataset": "figaro", "year": int(year), "units": "EUR million, current basic prices",
        "edition": "Eurostat FIGARO 2026 edition", "source_url": FIGARO_SOURCE_URL,
        "sources": (record,), "transformations": transformations,
        "source_country_codes": countries_src, "unmapped_country_codes": tuple(unmapped),
        "production_taxes_observed": True,
        "factor_share_provenance": {"kind": "observed", "labor_row": "W2_D1", "capital_row": "W2_B2A3G",
                                    "production_tax_row": "W2_D29X39",
                                    "capital_interpretation": "gross operating surplus and mixed income"},
        "consumption_convention": "Published territorial consumption, including government; no invented tourism redistribution",
        "inventory_convention": "P5M combines inventory changes and valuables; the source does not separate them",
        "output_convention": "Row total derived from published transactions; no separate output column",
        "purchases_abroad_residents": float(W["W2_OP_RES"][m:].sum()),
        "purchases_by_nonresidents": float(-W["W2_OP_NRES"][m:].sum()),
        "source_final_categories": FIGARO_FD_CODES,
        "final_use_mapping": {"P3_S13": "C", "P3_S14": "C", "P3_S15": "C", "P51G": "G", "P5M": "V"},
        "final_use_components": {code: _frozen(rawF[:, :, i]) for i, code in enumerate(FIGARO_FD_CODES)},
        "final_tax_components": {code: _frozen(rawTFD[:, i]) for i, code in enumerate(FIGARO_FD_CODES)},
        "rounding_adjustments": tuple(adjustments),
        "residence_welfare_comparable": False,
    }
    mask = goods_mask(sectors, "figaro64") if set(sectors) <= set(FIGARO_NATIVE_64_SECTORS) else None
    table = MRIOTable.from_arrays(
        sparse.csr_matrix(Z) if sparse_matrix else Z, F, VA, TLS, TFD,
        country_codes=ccodes, sector_codes=sectors, fd_codes=("C", "G", "X", "V"),
        production_taxes=prodtax, labor_compensation=lab, operating_surplus=cap,
        merchandise_mask=mask, reference_country="ROW" if "ROW" in ccodes else None, metadata=meta,
    )
    report = _reader_report(record, year=int(year), edition=meta["edition"], units=meta["units"],
                            n=n, s=s, adjustments=adjustments, transformations=transformations, path=str(path))
    return table.with_metadata(build_report=report)


_EXIOBASE_MEMBERS = ("Z.txt", "Y.txt", "x.txt", "industries.txt", "satellite/F.txt", "satellite/F_Y.txt")
_EXIOBASE_OPTIONAL = ("finaldemands.txt", "metadata.json")


def _exiobase_archive_record(archive: Path, expected_sha256: str | None, year: int) -> SourceRecord:
    """Hash an EXIOBASE archive against the caller's or the registered digest."""
    record = identify_source(archive, expected_sha256=expected_sha256)
    if record.sha256 == EXIOBASE_382_2019_IXI_SHA256 and int(year) != 2019:
        raise MRIOIntegrityError(f"{archive}: sha256 {record.sha256} is the registered EXIOBASE 3.8.2 2019 ixi "
                                 f"archive, not the requested year {int(year)}")
    if expected_sha256 is None:
        if record.sha256 == EXIOBASE_382_2019_IXI_SHA256:
            return replace(record, status="verified", note="EXIOBASE 3.8.2 IOT_2019_ixi.zip")
        warnings.warn(f"{archive}: sha256 {record.sha256} is not the registered EXIOBASE 3.8.2 2019 ixi "
                      "archive; the archive is read as an unauthenticated edition", RuntimeWarning, stacklevel=4)
        return replace(record, status="unknown_edition", note="sha256 not registered")
    return record


def _exiobase_prefix(names: set[str], prefix: str, archive: Path) -> str:
    if all(prefix + m in names for m in _EXIOBASE_MEMBERS):
        return prefix
    if all(m in names for m in _EXIOBASE_MEMBERS):
        return ""
    raise MRIOIntegrityError(f"{archive} lacks EXIOBASE members {_EXIOBASE_MEMBERS} under {prefix!r}")


class _ExiobaseSource:
    """Uniform access to an extracted EXIOBASE directory or the ZIP archive.

    Three cases: the archive itself (hashed, members read and CRC-checked from
    it); an extracted directory with its ``archive`` (the archive is hashed and
    every extracted member is checked against the archive entry's size and
    CRC-32, as the IO ``_check_extracted_member`` does); an extracted directory
    alone (members hashed only, a ``RuntimeWarning``, status
    ``"unknown_edition"``).
    """

    def __init__(self, path: Path, prefix: str, expected_sha256: str | None,
                 archive: Path | None = None, year: int = 2019) -> None:
        self.path = path
        self.records: list[SourceRecord] = []
        self.bundle = None
        self.archive: Path | None = None
        self.prefix = ""
        if path.is_dir():
            self.base = path
            if archive is None:
                if expected_sha256 is not None:
                    raise ValueError("expected_sha256 is the digest of the EXIOBASE archive; pass archive=... to "
                                     "authenticate an extracted directory against it")
                warnings.warn(f"{path}: extracted EXIOBASE directory read without its archive; the members are "
                              "hashed but not authenticated (pass archive=... to check them against the archive "
                              "CRCs)", RuntimeWarning, stacklevel=3)
            else:
                archive = Path(archive)
                if not archive.is_file():
                    raise FileNotFoundError(f"EXIOBASE archive not found: {archive}")
                self.records.append(_exiobase_archive_record(archive, expected_sha256, year))
                with zipfile.ZipFile(archive) as bundle:
                    self.prefix = _exiobase_prefix(set(bundle.namelist()), prefix, archive)
                self.archive = archive
        elif path.is_file() and path.suffix.lower() == ".zip":
            if archive is not None:
                raise ValueError("archive= applies to an extracted directory; the path is already the archive")
            self.records.append(_exiobase_archive_record(path, expected_sha256, year))
            self.bundle = zipfile.ZipFile(path)
            try:
                self.prefix = _exiobase_prefix(set(self.bundle.namelist()), prefix, path)
            except MRIOIntegrityError:
                self.bundle.close()
                raise
        else:
            raise FileNotFoundError(f"EXIOBASE source required (extracted directory or archive): {path}")

    def has(self, member: str) -> bool:
        if self.bundle is None:
            return (self.base / member).is_file()
        return (self.prefix + member) in set(self.bundle.namelist())

    def record(self, member: str) -> None:
        if self.bundle is not None:
            self.records.append(verify_zip_member(self.path, self.prefix + member))
        elif self.archive is not None:
            # Size and CRC-32 of the extracted file against the archive entry (raises on any change).
            self.records.append(verify_zip_member(self.archive, self.prefix + member, extracted_path=self.base / member))
        else:
            self.records.append(replace(identify_source(self.base / member), archive_member=member,
                                        status="unknown_edition", note="extracted member; no archive to check against"))

    def open(self, member: str):
        if self.bundle is None:
            return (self.base / member).open("rb")
        return self.bundle.open(self.prefix + member)

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()


def read_exiobase_native(
    path_or_zip: str | Path,
    year: int = 2019,
    *,
    model: str = "ixi",
    archive: str | Path | None = None,
    expected_sha256: str | None = None,
    sparse_matrix: bool = True,
    chunksize: int = 128,
) -> MRIOTable:
    """Read EXIOBASE 3 industry-by-industry tables from ``IOT_{year}_{model}.zip`` or its extracted core.

    Members read (under ``IOT_{year}_{model}/`` inside the archive): ``Z.txt``,
    ``Y.txt``, ``x.txt``, ``industries.txt``, ``satellite/F.txt``,
    ``satellite/F_Y.txt`` (``finaldemands.txt`` and ``metadata.json`` when
    present). ``Z`` is streamed in ``chunksize``-row blocks into CSR (no
    7,987-square dense matrix is kept) unless ``sparse_matrix=False``.

    Authentication. The archive is hashed and compared with
    ``expected_sha256`` (enforced when given) or with
    :data:`EXIOBASE_382_2019_IXI_SHA256` (a ``RuntimeWarning`` flags any other
    digest; the registered archive read with ``year != 2019`` raises). Members
    read from the archive are checked against its CRC-32 entries. An extracted
    directory is authenticated only when its ``archive`` is passed: every
    member is then checked against the archive entry's size and CRC-32 (the IO
    loader's check), and a changed byte raises. A directory read without its
    archive is hashed member by member, recorded with status
    ``"unknown_edition"`` and flagged with a ``RuntimeWarning``;
    ``expected_sha256`` without ``archive`` is refused for a directory because
    the digest belongs to the archive.

    Final uses: C = households + NPISH + government, G = GFCF, X identically
    zero (no OECD DPABR basket exists), V = changes in inventories, VAL =
    changes in valuables, so ``fd_codes = ("C", "G", "X", "V", "VAL")``. The
    ``Exports: Total (fob)`` column must be zero (otherwise an explicit
    destination model would be needed) and the final-demand primary-factor
    rows must be zero. ``TLS`` is the "Taxes less subsidies on products
    purchased" row, ``production_taxes`` "Other net taxes on production",
    labour the three skill rows and ``operating_surplus`` the four surplus rows
    ("gross operating surplus including land/resources; not a measured
    reproducible-capital stock"); ``VA`` is their sum. Output is ``x.txt``.
    Industries are identified by ``CodeNr`` (i01.a, ...); regions are mapped
    through :data:`EXIOBASE_ISO3` and the five rest-of-world regions stay
    distinct. "This 2019 release is a nowcast; no stronger measurement claim
    is implied."
    """
    path = Path(path_or_zip)
    src = _ExiobaseSource(path, f"IOT_{int(year)}_{model}/", expected_sha256,
                          archive=None if archive is None else Path(archive), year=int(year))
    try:
        for member in _EXIOBASE_MEMBERS:
            if not src.has(member):
                raise FileNotFoundError(f"EXIOBASE member {member} is missing under {path}")
            src.record(member)
        for member in _EXIOBASE_OPTIONAL:
            if src.has(member):
                src.record(member)
        with src.open("industries.txt") as fh:
            industry = pd.read_csv(fh, sep="\t")
        for col in ("Name", "CodeNr"):
            if col not in industry.columns:
                raise MRIOIntegrityError("industries.txt must have Name and CodeNr columns")
        names = tuple(str(x) for x in industry["Name"])
        sectors = tuple(str(x) for x in industry["CodeNr"])
        if src.has("finaldemands.txt"):
            with src.open("finaldemands.txt") as fh:
                fdtab = pd.read_csv(fh, sep="\t")
            if tuple(str(x) for x in fdtab["Name"]) != EXIOBASE_FINAL_NAMES:
                raise MRIOIntegrityError("finaldemands.txt categories differ from the EXIOBASE 3 layout")
        with src.open("Y.txt") as fh:
            ydf = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1])
        with src.open("x.txt") as fh:
            xdf = pd.read_csv(fh, sep="\t", index_col=[0, 1])
        with src.open("satellite/F.txt") as fh:
            factors = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=0, nrows=9)
        with src.open("satellite/F_Y.txt") as fh:
            final_factors = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=0, nrows=9)
        regions = tuple(str(x) for x in dict.fromkeys(ydf.index.get_level_values(0)))
        n, s = len(regions), len(sectors)
        m = n * s
        cells = [(r, nm) for r in regions for nm in names]
        fd_cols = [(r, cat) for r in regions for cat in EXIOBASE_FINAL_NAMES]
        if ydf.index.tolist() != cells or xdf.index.tolist() != cells:
            raise MRIOIntegrityError("EXIOBASE output/final-use labels do not match the region x industry ordering")
        if ydf.columns.tolist() != fd_cols or final_factors.columns.tolist() != fd_cols:
            raise MRIOIntegrityError("EXIOBASE final-use column labels do not match the native categories")
        if factors.columns.tolist() != cells:
            raise MRIOIntegrityError("EXIOBASE factor columns do not match the region x industry ordering")
        if tuple(factors.index) != EXIOBASE_FACTOR_NAMES or tuple(final_factors.index) != EXIOBASE_FACTOR_NAMES:
            raise MRIOIntegrityError("EXIOBASE monetary factor labels have changed")
        blocks: list[Any] = []
        offset = 0
        with src.open("Z.txt") as fh:
            for block in pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1], chunksize=int(chunksize)):
                if block.columns.tolist() != cells or block.index.tolist() != cells[offset:offset + len(block)]:
                    raise MRIOIntegrityError("EXIOBASE intermediate labels are inconsistent with final uses")
                values = block.to_numpy(dtype=float)
                blocks.append(sparse.csr_matrix(values) if sparse_matrix else values)
                offset += len(block)
        if offset != m:
            raise MRIOIntegrityError("EXIOBASE intermediate table has an unexpected row count")
        Z = sparse.vstack(blocks, format="csr") if sparse_matrix else np.vstack(blocks)
        del blocks
        rawF = ydf.to_numpy(dtype=float).reshape(m, n, 7)
        ftaxes = final_factors.iloc[0].to_numpy(dtype=float).reshape(n, 7)
        if np.any(rawF[:, :, 6] != 0) or np.any(ftaxes[:, 6] != 0):
            raise MRIOIntegrityError("EXIOBASE unallocated exports require an explicit destination model")
        if np.any(final_factors.iloc[1:].to_numpy(dtype=float) != 0):
            raise MRIOIntegrityError("Direct final-demand primary-factor payments require an explicit extension")
        fac = factors.to_numpy(dtype=float)
        lab, cap, prodtax, tls = fac[2:5].sum(axis=0), fac[5:9].sum(axis=0), fac[1], fac[0]
        F = np.stack([rawF[:, :, :3].sum(axis=2), rawF[:, :, 3], np.zeros((m, n)), rawF[:, :, 4], rawF[:, :, 5]], axis=2)
        TFD = np.column_stack([ftaxes[:, :3].sum(axis=1), ftaxes[:, 3], np.zeros(n), ftaxes[:, 4], ftaxes[:, 5]])
        output = xdf.iloc[:, 0].to_numpy(dtype=float)
        archive_meta = None
        if src.has("metadata.json"):
            with src.open("metadata.json") as fh:
                archive_meta = json.loads(fh.read().decode("utf-8"))
    finally:
        src.close()
    unmapped = [c for c in regions if c not in EXIOBASE_ISO3]
    if unmapped:
        warnings.warn(f"EXIOBASE region codes without an ISO-3 mapping are kept verbatim: {unmapped}",
                      RuntimeWarning, stacklevel=2)
    ccodes = tuple(EXIOBASE_ISO3.get(c, c) for c in regions)
    meta = {
        "dataset": "exiobase", "year": int(year), "units": "EUR million, current basic prices",
        "edition": f"EXIOBASE 3.8.2, industry by industry ({model})", "source_url": EXIOBASE_SOURCE_URL,
        "sources": tuple(src.records), "transformations": (),
        "source_authenticated": all(r.status == "verified" for r in src.records),
        "source_country_codes": regions, "unmapped_country_codes": tuple(unmapped), "sector_names": names,
        "nowcast": True, "archive_metadata": archive_meta, "production_taxes_observed": True,
        "factor_share_provenance": {"kind": "observed", "labor_rows": list(EXIOBASE_FACTOR_NAMES[2:5]),
                                    "capital_rows": list(EXIOBASE_FACTOR_NAMES[5:]),
                                    "production_tax_row": EXIOBASE_FACTOR_NAMES[1],
                                    "capital_interpretation": "gross operating surplus including land/resources; not a measured reproducible-capital stock"},
        "consumption_convention": "Published EXIOBASE consumption, including government; no separate OECD DPABR basket",
        "inventory_convention": "Inventory changes and valuables are separate flows; no stock is observed",
        "output_convention": "x.txt industry output column",
        "source_final_categories": EXIOBASE_FINAL_NAMES,
        "final_use_mapping": {"y01": "C", "y02.a": "C", "y02.b": "C", "y04": "G", "y05.a": "V", "y05.b": "VAL",
                              "y06": "must be zero"},
        "final_use_components": {name: _frozen(rawF[:, :, i]) for i, name in enumerate(EXIOBASE_FINAL_NAMES)},
        "final_tax_components": {name: _frozen(ftaxes[:, i]) for i, name in enumerate(EXIOBASE_FINAL_NAMES)},
        "factor_components": {name: _frozen(fac[i]) for i, name in enumerate(EXIOBASE_FACTOR_NAMES)},
        "residence_welfare_comparable": False,
    }
    mask = goods_mask(sectors, "exiobase163") if all(_EXIOBASE_CODE.match(c) for c in sectors) else None
    ref = "ROW_ASIA_PACIFIC" if "ROW_ASIA_PACIFIC" in ccodes else None
    table = MRIOTable.from_arrays(
        Z, F, lab + cap + prodtax, tls, TFD, country_codes=ccodes, sector_codes=sectors,
        fd_codes=("C", "G", "X", "V", "VAL"), output=output, production_taxes=prodtax,
        labor_compensation=lab, operating_surplus=cap, merchandise_mask=mask, reference_country=ref, metadata=meta,
    )
    report = _reader_report(src.records[0], year=int(year), edition=meta["edition"], units=meta["units"],
                            n=n, s=s, adjustments=(), transformations=(), path=str(path))
    return table.with_metadata(build_report=report)


# ---------------------------------------------------------------------------
# Regularization (the IO table_from_arrays contract, steps 2-8)
# ---------------------------------------------------------------------------


def regularize_table(
    table: MRIOTable,
    *,
    inactive: str = "phantom",
    phantom: float = 1e-6,
    va_floor_rel: float = 1e-3,
    row_identity_tol: float = 1e-5,
    dtls_world_tol: float = 1e-4,
    dtls_cell_tol: float = 1e-2,
    xn_sum_tol: float = 1e-9,
    strict: bool = True,
    spectral: bool = False,
) -> tuple[MRIOTable, MRIOBuildReport]:
    """Balance a raw table with the recorded contract of the IO ``table_from_arrays``.

    Steps and gates (``out`` is the table's output column, ``row`` the row
    totals, ``active = out > 0``):

    1. Identity guard: ``median_{active} |row_i - out_i| / out_i <= row_identity_tol``
       and zero-output cells are completely empty (no entry in their Z row or
       column, F row, VA or TLS); negative output is refused.
    2. Phantoms (``inactive="phantom"``): every zero-output cell gets
       ``F[i, own country, C] = VA_i = phantom`` so its row and column agree
       exactly; ``inactive="mask"`` keeps them empty.
    3. Value-added floor: ``VA_i = va_floor_rel * out_i`` where active and
       ``VA_i <= 0`` (only there; positive VA is never changed).
    4. Net product taxes are recomputed as the residual that closes every
       column, ``tls_j = row_j - sum_i Z_ij - VA_j``. The change at cells that
       were not floored must satisfy ``sum |dtls| <= dtls_world_tol * sum |TLS|``
       and ``max |dtls_j| / max(out_j, 1) <= dtls_cell_tol`` (with the IO's
       ``1e-9`` relative slack on the second gate).
    5. Trade balances are measured fob over all uses; their world sum must be
       ``<= xn_sum_tol * |world GDP|`` with ``world GDP = sum VA + sum tls + sum TFD``.

    The returned table has ``output`` equal to the row totals after phantoms,
    ``TLS`` replaced by the residual and the transformation appended to
    ``metadata["transformations"]``. With ``strict=False`` no gate raises: the
    report carries every gate value with its pass flag (and the offending
    cells where a gate is about cells), ``passed`` is False when any failed,
    and the returned table then records ``metadata["regularized"] = False``,
    ``metadata["regularization_passed"] = False`` and
    ``metadata["failed_gates"]``, and its transformation string names the
    failed gates; :func:`aggregate_mrio` and :func:`to_calibration_matrix`
    warn when they receive such a table. Under ``strict=False`` a phantom is
    given only to inactive cells whose output is exactly zero and that are
    completely empty; cells with negative output or stray entries keep their
    data and are listed in ``report.skipped_phantoms``.

    Units: ``phantom`` and the ``max(out, 1)`` scale of the per-cell gate are
    absolute amounts in the table's units. They follow the IO convention for
    tables in millions of currency units (USD or EUR million); for a table in
    other units rescale ``phantom`` (and read the per-cell gate accordingly),
    since regularization is not scale-equivariant. ``spectral=True`` additionally reports the
    Collatz-Wielandt bounds of the input-coefficient matrix ``Z / output``
    through :func:`puremacro.trade.regularize.compute_spectral_radius`
    (dense; skipped for sparse tables). This differs from
    :func:`puremacro.trade.regularize.regularize_mrio_table`, which floors
    every active cell's value added at ``max(1e-3 Y, min(1, 0.5 Y))``
    (4.3.0: ``max(1e-3 Y, 1)``) and re-derives output from row sales before
    the residual unless ``Y`` is supplied.
    """
    if not isinstance(table, MRIOTable):
        raise TypeError("table must be an MRIOTable")
    if inactive not in ("phantom", "mask"):
        raise ValueError("inactive must be 'phantom' or 'mask'")
    for name, val in (("phantom", phantom), ("va_floor_rel", va_floor_rel), ("row_identity_tol", row_identity_tol),
                      ("dtls_world_tol", dtls_world_tol), ("dtls_cell_tol", dtls_cell_tol), ("xn_sum_tol", xn_sum_tol)):
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    n, s, m = table.n_countries, table.n_sectors, table.n_cells
    Z = table.Z
    F = np.array(table.F, dtype=float, copy=True)
    VA = np.array(table.VA, dtype=float, copy=True)
    TLS_raw = np.asarray(table.TLS, dtype=float)
    TFD = np.asarray(table.TFD, dtype=float)
    out = np.asarray(table.output, dtype=float)
    labels = table.cell_labels
    gates: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def named(idx: np.ndarray, cap: int = 5) -> str:
        shown = [labels[i] for i in idx[:cap]]
        return ", ".join(shown) + (f" and {idx.size - cap} more" if idx.size > cap else "")

    def gate(name: str, value: float, tol: float, ok: bool, message: str, cells: np.ndarray | None = None) -> None:
        gates[name] = {"value": float(value), "tolerance": float(tol), "passed": bool(ok)}
        if cells is not None:
            gates[name]["cells"] = tuple(labels[i] for i in cells[:20])
            gates[name]["n_cells"] = int(cells.size)
        if not ok:
            failures.append(message)

    col_z = _colsum(Z)
    row_raw = _rowsum(Z) + F.sum(axis=(1, 2))
    negative = np.flatnonzero(out < 0)
    gate("nonnegative_output", float(out.min(initial=0.0)), 0.0, negative.size == 0,
         f"negative gross output at {named(negative)}", negative)
    active = out > 0
    rel = np.abs(row_raw - out)[active] / out[active]
    med = float(np.median(rel)) if rel.size else 0.0
    gate("row_identity_median_rel", med, row_identity_tol, med <= row_identity_tol,
         f"median relative row identity error {med:.2e} exceeds {row_identity_tol:.0e}: not a balanced source table")
    zero = ~active
    if sparse.issparse(Z):
        Zc = Z.tocsr()  # |Z| on the shared index arrays: EXIOBASE has 38 million stored entries
        absZ = sparse.csr_matrix((np.abs(Zc.data), Zc.indices, Zc.indptr), shape=Zc.shape)
    else:
        absZ = np.abs(np.asarray(Z, dtype=float))
    mass = (_rowsum(absZ) + _colsum(absZ) + np.abs(F).sum(axis=(1, 2)) + np.abs(VA) + np.abs(TLS_raw))
    junk = float(mass[zero].sum())
    dirty = np.flatnonzero(zero & (mass > 0))
    gate("zero_output_cells_empty", junk, 0.0, junk == 0.0,
         f"a zero-output cell has non-zero entries: {named(dirty)}", dirty)
    if strict and failures:
        raise MRIOIntegrityError("; ".join(failures))

    country = table.country_of_cell
    eligible = zero & (out == 0) & (mass == 0)
    zero_idx = np.flatnonzero(eligible)
    skipped = np.flatnonzero(zero & ~eligible)
    if inactive == "phantom" and zero_idx.size:
        if "C" not in table.fd_codes:
            raise ValueError("phantom demand needs a C category in the table")
        ic = table.category_index("C")
        F[zero_idx, country[zero_idx], ic] = phantom
        VA[zero_idx] = phantom
    hit = active & (VA <= 0)
    floored = tuple((labels[i], float(VA[i]), float(va_floor_rel * out[i])) for i in np.flatnonzero(hit))
    VA[hit] = va_floor_rel * out[hit]
    row = _rowsum(Z) + F.sum(axis=(1, 2))
    tls = row - col_z - VA
    dtls = tls - TLS_raw
    acc = np.where(hit, 0.0, dtls)
    world_abs_tls = float(np.abs(TLS_raw).sum())
    cell_scale = np.maximum(out, 1.0)
    shares = np.zeros(m)
    shares[active] = np.abs(acc[active]) / cell_scale[active]
    acc_share = float(shares.max(initial=0.0)) if active.any() else 0.0
    worst = np.array([int(np.argmax(shares))]) if active.any() else np.array([], dtype=int)
    over = np.flatnonzero(shares > dtls_cell_tol * (1.0 + 1e-9))
    acc_abs = float(np.abs(acc).sum())
    gate("dtls_world_share", acc_abs / world_abs_tls if world_abs_tls > 0 else 0.0, dtls_world_tol,
         world_abs_tls == 0.0 or acc_abs <= dtls_world_tol * world_abs_tls,
         f"residual TLS changes the accounts by {acc_abs:.3e}, more than {dtls_world_tol:.0e} of world |TLS| ({world_abs_tls:.3e})")
    gate("dtls_cell_share_of_output", acc_share, dtls_cell_tol, acc_share <= dtls_cell_tol * (1.0 + 1e-9),
         f"residual TLS moves cell {named(worst)} by {acc_share:.2%} of max(output, 1) "
         f"({over.size} cells above {dtls_cell_tol:g})", over)
    gates["dtls_cell_share_of_output"]["worst_cell"] = labels[int(worst[0])] if worst.size else None
    xn0, raw_sum = _net_exports_raw(Z, F, n, s)
    world_gdp = float(VA.sum() + tls.sum() + TFD.sum())
    gate("trade_balance_world_sum", abs(raw_sum) / abs(world_gdp) if world_gdp else abs(raw_sum), xn_sum_tol,
         abs(raw_sum) <= xn_sum_tol * abs(world_gdp),
         f"world sum of trade balances {raw_sum:.3e} exceeds {xn_sum_tol:.0e} of world GDP")
    if strict and failures:
        raise MRIOIntegrityError("; ".join(failures))
    rho = None
    if spectral and not sparse.issparse(Z):
        from .regularize import compute_spectral_radius

        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.divide(np.asarray(Z, dtype=float), row[None, :], out=np.zeros((m, m)), where=row[None, :] > 0)
        rho = tuple(float(v) for v in compute_spectral_radius(A))
    passed = not failures
    failed = tuple(k for k, v in gates.items() if not v["passed"])
    transformation = (f"regularize_table: inactive={inactive}, {len(zero_idx)} phantoms of {phantom:g} own-country C "
                      f"and value added, {len(floored)} value-added floors at {va_floor_rel:g} of output, "
                      "net product taxes recomputed as the column residual")
    if skipped.size:
        transformation += f"; {skipped.size} inactive cells with negative output or stray entries left unchanged"
    if not passed:
        transformation += f"; FAILED gates: {', '.join(failed)} (strict=False)"
    transformations = table.transformations + (transformation,)
    prior = table.build_report
    src = table.sources[0] if table.sources else None
    report = MRIOBuildReport(
        source=prior.source if prior else (src.path if src else table.dataset),
        sha256=src.sha256 if src else None, md5=src.md5 if src else None, year=table.year,
        edition=table.edition, units=table.units, n_countries=n, n_sectors=s,
        row_identity_median_rel=med, n_zero_output=int(zero.sum()),
        phantoms=tuple(labels[i] for i in zero_idx), floored=floored,
        dtls_accounting_abs=acc_abs, dtls_floor_abs=float(np.abs(np.where(hit, dtls, 0.0)).sum()),
        dtls_accounting_max_share_of_output=acc_share, world_abs_tls=world_abs_tls,
        xn0_raw_sum=raw_sum, world_gdp=world_gdp, inactive=inactive,
        negative_fd_cells={c: int((F[:, :, k] < 0).sum()) for k, c in enumerate(table.fd_codes)},
        gates=gates, passed=passed, spectral_radius=rho,
        rounding_adjustments=tuple(table.metadata.get("rounding_adjustments", ())),
        transformations=transformations, skipped_phantoms=tuple(labels[i] for i in skipped),
    )
    new = replace(table, F=_frozen(F), VA=_frozen(VA), TLS=_frozen(tls), output=_frozen(row),
                  metadata={**table.metadata, "transformations": transformations, "build_report": report,
                            "regularized": bool(passed), "regularization_passed": bool(passed),
                            "failed_gates": failed, "inactive": inactive, "net_exports_raw_sum": raw_sum})
    return new, report


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_mrio(
    table: MRIOTable,
    concordance: Any,
    *,
    coarse_codes: Sequence[str] | None = None,
    region_concordance: Any | None = None,
    coarse_country_codes: Sequence[str] | None = None,
    tol: float = 1e-8,
) -> MRIOTable:
    """Exact aggregation ``Z_G = P Z P^T`` of sectors (and optionally regions).

    ``concordance`` is a :class:`Concordance` on the table's sector codes, a
    registered name (``"agregar"``, ``"isic_section"`` for the 45 OECD
    industries), a mapping ``{fine: coarse}`` or a sequence of 1-based group
    ids (with ``coarse_codes``). With ``P = I_N kron C^T``: ``Z_G = P Z P^T``,
    ``F_G[:, n, k] = P F[:, n, k]``, ``VA_G = P VA``, ``TLS_G = P TLS``,
    ``output_G = P output`` and ``TFD`` is unchanged; the optional factor
    detail is summed the same way. A ``region_concordance`` (on the country
    codes) additionally merges origin and destination countries and sums
    ``TFD`` over them. Both identities are linear, so aggregation commutes with
    the residual-tax closure of :func:`regularize_table` whenever no
    value-added floor applies (floors act cell by cell), and every total is
    conserved exactly (up to floating-point summation).

    Gate: when the fine table is balanced (``max_j |row_j - col_j| / max(|col_j|, 1)
    <= tol``) the coarse table must be too; the residual is recorded in
    ``metadata["aggregation"]`` in every case. The merchandise mask of a
    coarse group is True only when every member is merchandise; mixed groups
    are listed in ``metadata["aggregation"]["mixed_goods_groups"]``.

    A table whose :func:`regularize_table` gates failed (``strict=False``) is
    aggregated, but a ``RuntimeWarning`` names the failed gates and the
    coarse table keeps ``build_report.passed = False``,
    ``metadata["regularized"] = False`` and ``metadata["failed_gates"]``.
    """
    if not isinstance(table, MRIOTable):
        raise TypeError("table must be an MRIOTable")
    prior_report = table.build_report
    if prior_report is not None and not prior_report.passed:
        warnings.warn(f"aggregating a table whose regularize_table gates failed ({', '.join(prior_report.failed_gates)}); "
                      "the coarse table carries the failure in build_report and metadata['failed_gates']",
                      RuntimeWarning, stacklevel=2)
    conc = _resolve_concordance(concordance, fine_codes=table.sector_codes, coarse_codes=coarse_codes)
    n, s, k = table.n_countries, table.n_sectors, table.n_categories
    C = conc.matrix
    g = conc.n_coarse
    if region_concordance is None:
        rconc = Concordance.identity(table.country_codes, "identity")
    else:
        rconc = _resolve_concordance(region_concordance, fine_codes=table.country_codes,
                                     coarse_codes=coarse_country_codes, what="country")
    Pc = rconc.matrix.T  # (N', N)
    n2 = rconc.n_coarse
    R = sparse.kron(sparse.csr_matrix(Pc), sparse.csr_matrix(C.T), format="csr")  # (N' G, N S)
    if sparse.issparse(table.Z):
        Z = (R @ table.Z @ R.T).tocsr()
    else:
        Z = np.asarray(R @ np.asarray(table.Z) @ R.T)
    F = np.asarray(R @ table.F.reshape(n * s, n * k)).reshape(n2 * g, n, k)
    F = np.einsum("mnk,pn->mpk", F, Pc)
    VA = np.asarray(R @ table.VA)
    TLS = np.asarray(R @ table.TLS)
    output = np.asarray(R @ table.output)
    TFD = Pc @ table.TFD
    detail = {name: (None if arr is None else np.asarray(R @ arr))
              for name, arr in (("production_taxes", table.production_taxes),
                                ("labor_compensation", table.labor_compensation),
                                ("operating_surplus", table.operating_surplus))}
    fine_row, fine_col = table.row_sales(), table.column_outlays()
    fine_rel = float(np.max(np.abs(fine_row - fine_col) / np.maximum(np.abs(fine_col), 1.0), initial=0.0))
    rows = _rowsum(Z) + F.sum(axis=(1, 2))
    cols = _colsum(Z) + TLS + VA
    rel = float(np.max(np.abs(rows - cols) / np.maximum(np.abs(cols), 1.0), initial=0.0))
    balanced = fine_rel <= tol
    if balanced and rel > tol:
        raise MRIOIntegrityError(f"aggregated table breaks row = column by {rel:.2e} (fine table {fine_rel:.2e})")
    mixed: list[str] = []
    mask = None
    if table.merchandise_mask is not None:
        mask = conc.coarse_mask(table.merchandise_mask, "all")
        any_mask = conc.coarse_mask(table.merchandise_mask, "any")
        mixed = [conc.coarse_codes[i] for i in np.flatnonzero(any_mask & ~mask)]
    ref = table.reference_country or table.country_codes[-1]
    ref_coarse = rconc.group_of(ref)
    info = {
        "sector_concordance": conc.name, "coarse_sector_codes": conc.coarse_codes,
        "region_concordance": rconc.name if region_concordance is not None else None,
        "fine_identity_rel": fine_rel, "aggregated_identity_rel": rel, "fine_balanced": bool(balanced),
        "tolerance": float(tol), "mixed_goods_groups": tuple(mixed),
        "conservation": "exact P M P^T summation; totals conserved up to floating-point rounding",
        "input_gates_passed": None if prior_report is None else bool(prior_report.passed),
    }
    transformations = table.transformations + (
        f"aggregate_mrio: sectors {s}->{g} with concordance {conc.name!r}"
        + (f", regions {n}->{n2} with concordance {rconc.name!r}" if region_concordance is not None else ""),
    )
    meta = {**table.metadata, "transformations": transformations, "aggregation": info,
            "final_use_components": None, "final_tax_components": None, "factor_components": None}
    prior = table.build_report
    if prior is not None:
        meta["build_report"] = replace(prior, n_countries=n2, n_sectors=g, concordance=conc.name,
                                       aggregated_identity_rel=rel, transformations=transformations,
                                       negative_fd_cells={c: int((F[:, :, i] < 0).sum())
                                                          for i, c in enumerate(table.fd_codes)})
    return MRIOTable.from_arrays(
        Z, F, VA, TLS, TFD, country_codes=rconc.coarse_codes, sector_codes=conc.coarse_codes,
        fd_codes=table.fd_codes, output=output, production_taxes=detail["production_taxes"],
        labor_compensation=detail["labor_compensation"], operating_surplus=detail["operating_surplus"],
        merchandise_mask=mask, reference_country=ref_coarse, metadata=meta, strict=False,
    )


# ---------------------------------------------------------------------------
# Coarse tariff rules (IO coarse_11o / coarse_11b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CoarseTariffResult:
    """Coarse ad-valorem rates and duty-equivalent wedges from :func:`coarse_tariff_rates`.

    ``rates`` ``(N, G)`` are the import-weighted averages (rule ``"output"``),
    ``weights`` ``(N, S)`` the importer's basic-price purchases used as
    weights. Under rule ``"bilateral"`` ``tau`` ``(N G, N G)`` and ``tau_fd``
    ``(N G, N, K)`` hold buyer-specific multipliers ``1 + r`` (ones for
    domestic flows, for non-importer columns and for categories not tariffed;
    X is never tariffed). ``benchmark_revenue_fine`` is the duty revenue at
    benchmark flows under the fine rates; ``benchmark_revenue_coarse`` is the
    same revenue under the coarse rates or wedges. The output rule preserves
    the total whenever each origin-group weight is positive (its weights are
    the tariffed purchases). The bilateral rule preserves it block by block
    where benchmark purchases are positive: under ``negative_weights="keep"``
    a block with nonpositive purchases keeps a unit multiplier, so the two
    totals differ by that block's fine duty; under ``"clip"`` they agree to
    rounding.
    """

    rule: str
    importer: str | None
    rates: np.ndarray
    weights: np.ndarray
    country_codes: tuple[str, ...]
    coarse_codes: tuple[str, ...]
    fd_codes: tuple[str, ...]
    fd_tariffed: tuple[str, ...]
    concordance: str
    tau: np.ndarray | None = None
    tau_fd: np.ndarray | None = None
    benchmark_revenue_fine: float | None = None
    benchmark_revenue_coarse: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Coarse rates, countries by coarse sectors."""
        return pd.DataFrame(self.rates, index=pd.Index(self.country_codes, name="origin"), columns=self.coarse_codes)

    def summary(self) -> str:
        rev = ("" if self.benchmark_revenue_fine is None else
               f"; benchmark revenue fine={self.benchmark_revenue_fine:.6g}, coarse={self.benchmark_revenue_coarse:.6g}")
        return (f"coarse tariff rates ({self.rule}, importer={self.importer}, concordance={self.concordance}): "
                f"max rate {float(np.max(self.rates, initial=0.0)):.4f}{rev}")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)


def coarse_tariff_rates(
    fine_rates: Any,
    weights: Any,
    concordance: Any,
    rule: str = "output",
    *,
    importer: str | None = None,
    fd_tariffed: Sequence[str] | None = None,
    coarse_codes: Sequence[str] | None = None,
    negative_weights: str = "keep",
) -> CoarseTariffResult:
    """Coarsen origin-by-sector ad-valorem rates with the IO rules ``11o`` and ``11b``.

    ``fine_rates`` is ``(N, S)``: the additional duty ``r_{os} >= 0`` that the
    importer levies on sector ``s`` from origin ``o`` (multiplier
    ``tau = 1 + r``; the importer's own row is ignored and set to zero).
    Negative rates (import subsidies) raise ``ValueError``, following the
    shared convention ``r >= 0`` and the IO scenario contract.
    ``fd_tariffed`` defaults to ``("C", "G", "V", "VAL")``, of which the
    categories the table carries are used: valuables are tariffed with
    inventories, as in the IO EXIOBASE build (which merged them into V) and in
    FIGARO, whose P5M already contains them.
    ``weights`` is the fine :class:`MRIOTable` at benchmark basic prices, from
    which the importer's purchases are formed, or an explicit ``(N, S)`` array
    of import weights (rule ``"output"`` only).

    Rule ``"output"`` (IO ``coarse_11o``): with
    ``m_{os} = sum_{j in importer} Z_{(o,s),j} + sum_{f in fd_tariffed} F_{(o,s),importer,f}``,
    ``r_{oG} = sum_{s in G} m_{os} r_{os} / sum_{s in G} m_{os}`` and zero when
    the weight sum is not positive. Rates lie between the minimum and maximum
    fine rate of the group when the weights are nonnegative.

    Rule ``"bilateral"`` (IO ``coarse_11b``): buyer-specific duty equivalents
    ``tau_{(o,G),(imp,J)} = 1 + sum_{s in G, j in J} Z_{(o,s),j} r_{os} / sum_{s in G, j in J} Z_{(o,s),j}``
    and, per tariffed final-use category f,
    ``tau^f_{(o,G),imp} = 1 + sum_{s in G} F^f_{(o,s),imp} r_{os} / sum_{s in G} F^f_{(o,s),imp}``,
    ones where the denominator is not positive. These preserve benchmark duty
    revenue block by block wherever the block's benchmark purchases are
    positive; a block with nonpositive purchases (an inventory drawdown under
    ``negative_weights="keep"``) keeps a unit multiplier and its fine-schedule
    duty is dropped, exactly as the IO rule's ``tot > 0`` guard does.
    ``benchmark_revenue_fine`` and ``benchmark_revenue_coarse`` report both
    sides, so the dropped amount is visible; with ``negative_weights="clip"``
    they agree to rounding. Because the output rule's weights are exactly the
    tariffed purchases, it preserves the revenue of each origin-group total as
    well; what it loses is the allocation across buyer groups and categories.

    Sign conventions: signed inventories can make a weight negative; the IO
    rules use the raw sums (``negative_weights="keep"``) and only guard the
    denominators. ``negative_weights="clip"`` clips each final-use
    contribution at zero before weighting (a drawdown is not an import).
    Category X is never tariffed and is rejected in ``fd_tariffed``.
    """
    if rule not in ("output", "bilateral"):
        raise ValueError("rule must be 'output' or 'bilateral'")
    if negative_weights not in ("keep", "clip"):
        raise ValueError("negative_weights must be 'keep' or 'clip'")
    tariffed = ("C", "G", "V", "VAL") if fd_tariffed is None else tuple(str(c) for c in fd_tariffed)
    if "X" in tariffed:
        raise ValueError("residents' purchases abroad (X) are never tariffed")
    if any(c not in FINAL_USE_CATEGORIES for c in tariffed):
        raise ValueError(f"fd_tariffed must be drawn from {FINAL_USE_CATEGORIES}")
    rates = np.asarray(fine_rates, dtype=float)
    if rates.ndim != 2 or not np.isfinite(rates).all():
        raise ValueError("fine_rates must be a finite (N, S) array")
    if np.any(rates < 0.0):
        raise ValueError("fine rates must be nonnegative ad-valorem duties (r >= 0; tau = 1 + r)")
    table: MRIOTable | None = weights if isinstance(weights, MRIOTable) else None
    if table is not None:
        n, s = table.n_countries, table.n_sectors
        ccodes, scodes, fcodes = table.country_codes, table.sector_codes, table.fd_codes
    else:
        if rule != "output":
            raise ValueError("rule 'bilateral' needs the fine MRIOTable as weights")
        W = np.asarray(weights, dtype=float)
        if W.shape != rates.shape or not np.isfinite(W).all():
            raise ValueError("explicit weights must be a finite array with the shape of fine_rates")
        n, s = W.shape
        ccodes = tuple(f"C{i:02d}" for i in range(n))
        if isinstance(concordance, Concordance):
            scodes = tuple(concordance.fine_codes)
        elif isinstance(concordance, str) and concordance in CONCORDANCES:
            scodes = tuple(CONCORDANCES[concordance].fine_codes)
        else:
            scodes = tuple(f"S{i:02d}" for i in range(s))
        if len(scodes) != s:
            raise ValueError(f"concordance is defined on {len(scodes)} sectors, weights have {s} columns")
        fcodes = ()
    if rates.shape != (n, s):
        raise ValueError(f"fine_rates shape {rates.shape} must be ({n}, {s})")
    conc = _resolve_concordance(concordance, fine_codes=scodes, coarse_codes=coarse_codes)
    C = conc.matrix
    g = conc.n_coarse
    if importer is None:
        if table is None:
            imp = None
        elif "USA" in ccodes:
            imp = ccodes.index("USA")
        else:
            raise ValueError("importer is required when the table has no USA")
    else:
        if table is None:
            raise ValueError("importer needs the fine MRIOTable as weights")
        imp = table.country_index(str(importer))
    if table is not None:
        cols = np.arange(imp * s, (imp + 1) * s)
        Zi = table.Z[:, cols]
        Zi = Zi.toarray() if sparse.issparse(Zi) else np.asarray(Zi, dtype=float)
        present = [c for c in tariffed if c in fcodes]
        fparts = {c: np.array(table.F[:, imp, fcodes.index(c)], copy=True) for c in present}
        if negative_weights == "clip":
            fparts = {c: np.maximum(v, 0.0) for c, v in fparts.items()}
        f_imp = sum(fparts.values(), np.zeros(n * s))
        W = (Zi.sum(axis=1) + f_imp).reshape(n, s)
    W = np.array(W, dtype=float, copy=True)
    r = np.array(rates, dtype=float, copy=True)
    if imp is not None:
        W[imp, :] = 0.0
        r[imp, :] = 0.0
    num, den = (W * r) @ C, W @ C
    coarse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    if imp is not None:
        coarse[imp, :] = 0.0
    tau = tau_fd = None
    rev_fine = rev_coarse = None
    if table is not None:
        r_flat = r.reshape(-1)
        R = conc.projection(n, sparse_matrix=True)  # (N G, N S)
        ZG = np.asarray(R @ Zi @ C)  # (N G, G)
        ZG_r = np.asarray(R @ (Zi * r_flat[:, None]) @ C)
        FG = {c: np.asarray(R @ v) for c, v in fparts.items()}
        FG_r = {c: np.asarray(R @ (v * r_flat)) for c, v in fparts.items()}
        own = slice(imp * g, (imp + 1) * g)
        rev_fine = float(ZG_r.sum() + sum(v.sum() for v in FG_r.values()))
        if rule == "bilateral":
            # 1 + weighted rate where the block has positive purchases, 1 elsewhere
            block = 1.0 + np.divide(ZG_r, ZG, out=np.zeros_like(ZG), where=ZG > 0)
            block[own, :] = 1.0
            tau = np.ones((n * g, n * g))
            tau[:, own] = block
            tau_fd = np.ones((n * g, n, len(fcodes)))
            for c in fparts:
                col = 1.0 + np.divide(FG_r[c], FG[c], out=np.zeros(n * g), where=FG[c] > 0)
                col[own] = 1.0
                tau_fd[:, imp, fcodes.index(c)] = col
            rev_coarse = float((ZG * (block - 1.0)).sum()
                               + sum((FG[c] * (tau_fd[:, imp, fcodes.index(c)] - 1.0)).sum() for c in fparts))
        else:
            rate_rows = coarse.reshape(-1)  # (N G,), origin-major like the rows of R
            rev_coarse = float((ZG.sum(axis=1) * rate_rows).sum()
                               + sum((FG[c] * rate_rows).sum() for c in fparts))
    return CoarseTariffResult(
        rule=rule, importer=(ccodes[imp] if imp is not None else None), rates=_frozen(coarse), weights=_frozen(W),
        country_codes=tuple(ccodes), coarse_codes=conc.coarse_codes, fd_codes=tuple(fcodes),
        fd_tariffed=tariffed, concordance=conc.name, tau=None if tau is None else _frozen(tau),
        tau_fd=None if tau_fd is None else _frozen(tau_fd),
        benchmark_revenue_fine=rev_fine, benchmark_revenue_coarse=rev_coarse,
        metadata={"negative_weights": negative_weights,
                  "weights": "importer's basic-price purchases: Z columns plus tariffed final uses",
                  "fd_tariffed_present": tuple(c for c in tariffed if c in fcodes),
                  "domestic_multiplier": 1.0, "X_tariffed": False},
    )


# ---------------------------------------------------------------------------
# Bridge to the legacy calibration matrix
# ---------------------------------------------------------------------------


def to_calibration_matrix(
    table: MRIOTable,
    *,
    production_taxes: str = "to_tls",
    labor_share: float = 2.0 / 3.0,
    check_balance: bool = True,
    return_details: bool = False,
) -> Any:
    """The ``(M + 3, M + 3 N)`` transaction matrix for :func:`calibrate_trade_model`.

    Rows ``0..M-1`` are ``[Z | F3]`` with three final-use columns per
    destination country in the legacy order C, I, Cx where I = G + V + VAL
    (signed inventories and valuables are merged into investment, as the
    bundled layout does) and Cx = X (zeros when the table has no residents'
    purchases abroad, as for FIGARO, WIOD, Eora and EXIOBASE tables; solve
    such a calibration with ``accounting="consistent"``, because the legacy
    demand ``theta * Y / ppfd`` is 0/0 for an empty category). Row ``M`` is
    TLS with the final-use taxes condensed the same way; rows ``M + 1`` and
    ``M + 2`` are labour and capital.

    Factor rows. ``calibrate_trade_model`` reconstructs every column from a
    Cobb-Douglas split with capital share ``alpha = K / (K + L)`` and accepts
    a column only when ``0 < alpha < 1`` (both factors strictly positive) or
    when both factors are zero. The rows are therefore built as follows.

    * Factor value added ``fva_j``: with ``production_taxes="to_tls"``
      (default) observed other net production taxes move to the TLS row,
      which the legacy calibration models as an output tax, and
      ``fva_j = VA_j - prod_j``; at a cell where that would leave
      ``fva_j <= 0`` (production taxes at least as large as value added:
      measured on the regularized 2019 tables, 3 FIGARO cells and 30
      EXIOBASE cells, 9 of which the plain subtraction would have left with
      negative factor income) the cell's production taxes stay in value
      added instead, and the cell is listed in a ``RuntimeWarning``. ``"to_factors"`` keeps every
      production tax inside value added.
    * Split: the observed proportions ``L / (L + K)`` where labour
      compensation and operating surplus are both strictly positive, and
      ``labor_share`` (default 2/3, the puremacro assumption) everywhere else,
      including cells with a zero, negative or unobserved factor (the
      regularized FIGARO 2019 table has 30 active cells with zero surplus and
      7 with zero labour).
    * A cell with negative factor value added cannot be calibrated and
      raises :class:`MRIOIntegrityError` naming it; :func:`regularize_table`
      floors value added where it is not positive.

    Column outlays equal ``sum_i Z_ij + TLS_j + VA_j`` exactly. When
    ``check_balance`` is set and they differ from the row totals (a raw
    table), a ``RuntimeWarning`` recommends :func:`regularize_table` first;
    a table whose regularization gates failed also warns. With
    ``return_details=True`` the result is ``(matrix, details)`` where
    ``details`` lists the cells on the fallback split and the cells whose
    production taxes stayed in value added.
    """
    if not isinstance(table, MRIOTable):
        raise TypeError("table must be an MRIOTable")
    if production_taxes not in ("to_tls", "to_factors"):
        raise ValueError("production_taxes must be 'to_tls' or 'to_factors'")
    if not 0.0 < float(labor_share) < 1.0:
        raise ValueError("labor_share must lie strictly between 0 and 1")
    report = table.build_report
    if report is not None and not report.passed:
        warnings.warn(f"the table failed regularize_table gates ({', '.join(report.failed_gates)}); the calibration "
                      "matrix inherits that failure", RuntimeWarning, stacklevel=2)
    m, n = table.n_cells, table.n_countries
    labels = table.cell_labels
    raw = table.to_raw("cix")
    Z = np.asarray(raw.intermediate_matrix, dtype=float)
    F3 = np.asarray(raw.final_demand_matrix, dtype=float)
    tfd = np.asarray(raw.taxes_less_subsidies_fd, dtype=float)
    TLS = np.array(table.TLS, dtype=float, copy=True)
    VA = np.array(table.VA, dtype=float, copy=True)
    prod = table.production_taxes
    factor = VA.copy()
    retained = np.zeros(m, dtype=bool)
    if production_taxes == "to_tls" and prod is not None:
        net = VA - prod
        move = net > 0
        retained = ~move & (prod != 0)
        TLS[move] += prod[move]
        factor[move] = net[move]
    negative = np.flatnonzero(factor < 0)
    if negative.size:
        shown = ", ".join(labels[i] for i in negative[:5])
        raise MRIOIntegrityError(
            f"{negative.size} cells have negative factor value added ({shown}); calibrate_trade_model cannot split "
            "them into labour and capital. Run regularize_table first (it floors value added where VA <= 0)."
        )
    share = np.full(m, float(labor_share))
    observed = np.zeros(m, dtype=bool)
    if table.labor_compensation is not None:
        L, K = table.labor_compensation, table.operating_surplus
        observed = (L > 0) & (K > 0)
        share[observed] = L[observed] / (L[observed] + K[observed])
    fallback = (factor > 0) & ~observed
    if retained.any():
        idx = np.flatnonzero(retained)
        shown = ", ".join(labels[i] for i in idx[:5]) + (f" and {idx.size - 5} more" if idx.size > 5 else "")
        warnings.warn(f"production taxes of {idx.size} cell{'' if idx.size == 1 else 's'} are at least as large as "
                      f"value added ({shown}); "
                      "they stay in value added instead of moving to the TLS row", RuntimeWarning, stacklevel=2)
    labor, capital = share * factor, (1.0 - share) * factor
    zeros = np.zeros(3 * n)
    matrix = np.vstack([
        np.hstack([Z, F3]),
        np.hstack([TLS, tfd]),
        np.hstack([labor, zeros]),
        np.hstack([capital, zeros]),
    ])
    gap = 0.0
    if check_balance:
        rows = matrix[:m].sum(axis=1)
        cols = matrix[:, :m].sum(axis=0)
        gap = float(np.max(np.abs(rows - cols), initial=0.0))
        scale = float(np.max(np.abs(cols), initial=1.0))
        if gap > 1e-9 * max(scale, 1.0):
            warnings.warn(f"table rows and columns differ by up to {gap:.3e}; run regularize_table first for a "
                          "balanced calibration matrix", RuntimeWarning, stacklevel=2)
    if not return_details:
        return matrix
    details = {
        "production_taxes": production_taxes,
        "production_taxes_retained_cells": tuple(labels[i] for i in np.flatnonzero(retained)),
        "share_fallback_cells": tuple(labels[i] for i in np.flatnonzero(fallback)),
        "observed_share_cells": int(observed.sum()),
        "labor_share_fallback": float(labor_share),
        "zero_factor_cells": tuple(labels[i] for i in np.flatnonzero(factor == 0)),
        "row_column_max_absolute_gap": gap if check_balance else None,
    }
    return matrix, details
