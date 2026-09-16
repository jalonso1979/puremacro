"""OECD ICIO Data Ingestion and Regional/Sectoral Indexing.

This module provides data loading, canonical identifier registries, and slice
abstractions for the 77-country 11-sector empirical Input-Output dataset
originating from the OECD Inter-Country Input-Output (ICIO) tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Canonical Registries
# ---------------------------------------------------------------------------

CANONICAL_COUNTRY_CODES: tuple[str, ...] = (
    "ARG", "AUS", "AUT", "BEL", "BGD", "BGR", "BLR", "BRA", "BRN", "CAN",
    "CHE", "CHL", "CHN", "CIV", "CMR", "COL", "CRI", "CYP", "CZE", "DEU",
    "DNK", "EGY", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HKG", "HRV",
    "HUN", "IDN", "IND", "IRL", "ISL", "ISR", "ITA", "JOR", "JPN", "KAZ",
    "KHM", "KOR", "LAO", "LTU", "LUX", "LVA", "MAR", "MEX", "MLT", "MMR",
    "MYS", "NGA", "NLD", "NOR", "NZL", "PAK", "PER", "PHL", "POL", "PRT",
    "ROU", "RUS", "SAU", "SEN", "SGP", "SVK", "SVN", "SWE", "THA", "TUN",
    "TUR", "TWN", "UKR", "USA", "VNM", "ZAF", "ROW",
)

EU_COUNTRY_CODES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
)

CANONICAL_SECTOR_CODES: tuple[str, ...] = (
    "AGRI", "MINQ", "MANU", "ENEG", "CONS", "TRAD", "TRAN", "INFO", "FIN", "GOV", "OTHS",
)

CANONICAL_SECTOR_NAMES: dict[str, str] = {
    "AGRI": "Agriculture, forestry and fishing",
    "MINQ": "Mining and quarrying",
    "MANU": "Manufacturing",
    "ENEG": "Electricity, gas, steam, water, waste management",
    "CONS": "Construction",
    "TRAD": "Wholesale and retail trade",
    "TRAN": "Transportation and storage",
    "INFO": "Information and communication",
    "FIN": "Financial and insurance, real estate",
    "GOV": "Public administration, defense, education, health",
    "OTHS": "Other services (accommodation, arts, household services)",
}

RAW_45_SECTOR_CODES: tuple[str, ...] = (
    "A01_02", "A03", "B05_06", "B07_08", "B09",
    "C10T12", "C13T15", "C16", "C17_18", "C19",
    "C20", "C21", "C22", "C23", "C24",
    "C25", "C26", "C27", "C28", "C29",
    "C30", "C31T33", "D", "E", "F",
    "G", "H49", "H50", "H51", "H52",
    "H53", "I", "J58T60", "J61", "J62_63",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
)

RAW_45_SECTOR_NAMES: dict[str, str] = {
    "A01_02": "Agriculture, hunting, forestry",
    "A03": "Fishing and aquaculture",
    "B05_06": "Mining and quarrying, energy producing products",
    "B07_08": "Mining and quarrying, non-energy producing products",
    "B09": "Mining support service activities",
    "C10T12": "Food products, beverages and tobacco",
    "C13T15": "Textiles, textile products, leather and footwear",
    "C16": "Wood and products of wood and cork",
    "C17_18": "Paper products and printing",
    "C19": "Coke and refined petroleum products",
    "C20": "Chemical and chemical products",
    "C21": "Pharmaceuticals, medicinal chemical and botanical products",
    "C22": "Rubber and plastics products",
    "C23": "Other non-metallic mineral products",
    "C24": "Basic metals",
    "C25": "Fabricated metal products",
    "C26": "Computer, electronic and optical equipment",
    "C27": "Electrical equipment",
    "C28": "Machinery and equipment, nec",
    "C29": "Motor vehicles, trailers and semi-trailers",
    "C30": "Other transport equipment",
    "C31T33": "Manufacturing nec; repair and installation of machinery and equipment",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles",
    "H49": "Land transport and transport via pipelines",
    "H50": "Water transport",
    "H51": "Air transport",
    "H52": "Warehousing and support activities for transportation",
    "H53": "Postal and courier activities",
    "I": "Accommodation and food service activities",
    "J58T60": "Publishing, audiovisual and broadcasting activities",
    "J61": "Telecommunications",
    "J62_63": "IT and other information services",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support services",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use",
}

CANONICAL_FINAL_DEMAND_CODES: tuple[str, ...] = (
    "C", "I", "Cx",
)

RAW_6_FINAL_DEMAND_CODES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR",
)


def get_country_codes() -> list[str]:
    """Return the canonical list of 77 ISO-3 country codes."""
    return list(CANONICAL_COUNTRY_CODES)


def get_eu_country_codes() -> list[str]:
    """Return the list of 27 European Union (EU-27) member state ISO-3 country codes."""
    return list(EU_COUNTRY_CODES)


def get_sector_codes(sectors: int = 11) -> list[str]:
    """Return the canonical list of sector codes (11 composite or 45 raw)."""
    if sectors == 45:
        return list(RAW_45_SECTOR_CODES)
    return list(CANONICAL_SECTOR_CODES)


def get_sector_names(sectors: int = 11) -> dict[str, str]:
    """Return descriptive mapping from sector codes to full names (11 composite or 45 raw)."""
    if sectors == 45:
        return dict(RAW_45_SECTOR_NAMES)
    return dict(CANONICAL_SECTOR_NAMES)


def get_raw_45_sector_codes() -> list[str]:
    """Return the canonical list of 45 unaggregated OECD ICIO sector codes."""
    return list(RAW_45_SECTOR_CODES)


def get_raw_45_sector_names() -> dict[str, str]:
    """Return descriptive mapping from 45 unaggregated sector codes to full names."""
    return dict(RAW_45_SECTOR_NAMES)


def get_final_demand_codes() -> list[str]:
    """Return the canonical list of 3 final demand category codes (['C', 'I', 'Cx'])."""
    return list(CANONICAL_FINAL_DEMAND_CODES)


def get_raw_final_demand_codes() -> list[str]:
    """Return the canonical list of 6 raw OECD ICIO final demand category codes."""
    return list(RAW_6_FINAL_DEMAND_CODES)


# ---------------------------------------------------------------------------
# Structured ICIO Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICIOData:
    """Structured container for 77-country 11-sector ICIO transaction table.

    Attributes
    ----------
    matrix : np.ndarray, shape (850, 1078) or (3468, 3696)
        Full float64 transaction table.
    country_codes : tuple[str, ...]
        Canonical 77 ISO-3 country codes.
    sector_codes : tuple[str, ...]
        Canonical 11 ISIC Rev.4 or 45 unaggregated sector codes.
    fd_codes : tuple[str, ...]
        Canonical 3 final demand codes ('C', 'I', 'Cx').
    """

    matrix: np.ndarray
    country_codes: tuple[str, ...] = CANONICAL_COUNTRY_CODES
    sector_codes: tuple[str, ...] = CANONICAL_SECTOR_CODES
    fd_codes: tuple[str, ...] = CANONICAL_FINAL_DEMAND_CODES

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            raise TypeError(f"matrix must be np.ndarray, got {type(self.matrix).__name__}")
        if self.matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-dimensional, got shape {self.matrix.shape}")
        if not isinstance(self.country_codes, tuple):
            object.__setattr__(self, "country_codes", tuple(self.country_codes))
        if not isinstance(self.sector_codes, tuple):
            object.__setattr__(self, "sector_codes", tuple(self.sector_codes))
        if not isinstance(self.fd_codes, tuple):
            object.__setattr__(self, "fd_codes", tuple(self.fd_codes))

        # Auto-detect 45 sectors if default 11 sectors was passed but matrix has 3468 rows
        nc = len(self.country_codes)
        if len(self.sector_codes) == 11 and self.matrix.shape[0] == 45 * nc + 3:
            object.__setattr__(self, "sector_codes", RAW_45_SECTOR_CODES)

    @property
    def intermediate_matrix(self) -> np.ndarray:
        """(847, 847) intermediate transactions Z(i, j)."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[:n_ind, :n_ind]

    @property
    def final_demand_matrix(self) -> np.ndarray:
        """(847, 231) final demand deliveries FD(i, j)."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_fd = len(self.country_codes) * len(self.fd_codes)
        return self.matrix[:n_ind, n_ind:n_ind + n_fd]

    @property
    def net_taxes_intermediate(self) -> np.ndarray:
        """(847,) net production taxes across all 847 country-sector pairs."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind, :n_ind]

    @property
    def net_taxes_final_demand(self) -> np.ndarray:
        """(231,) net taxes on final demand across 77 countries and 3 categories."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_fd = len(self.country_codes) * len(self.fd_codes)
        return self.matrix[n_ind, n_ind:n_ind + n_fd]

    @property
    def labor_va(self) -> np.ndarray:
        """(847,) labor compensation across 847 country-sector pairs."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind + 1, :n_ind]

    @property
    def capital_va(self) -> np.ndarray:
        """(847,) gross capital return across 847 country-sector pairs."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind + 2, :n_ind]

    @property
    def gross_output(self) -> np.ndarray:
        """(847,) gross production output y computed as column outlay sums."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_rows = n_ind + 3
        return np.sum(self.matrix[:n_rows, :n_ind], axis=0)


# ---------------------------------------------------------------------------
# Data Loading Entrypoints
# ---------------------------------------------------------------------------

def _load_raw_table(file_path: Path) -> np.ndarray:
    """Load raw matrix from TSV or CSV with optional header detection."""
    with open(file_path, "r", encoding="utf-8") as f:
        sample = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                sample = stripped
                break
    delim = "\t" if "\t" in sample else ","
    if "V1" in sample or sample.startswith(","):
        import pandas as pd
        df = pd.read_csv(file_path, index_col=0)
        return df.to_numpy(dtype=np.float64)
    return np.loadtxt(file_path, delimiter=delim, dtype=np.float64)


def bundled_icio_path() -> Path:
    """Absolute path to the ICIO matrix bundled inside the installed package.

    puremacro ships the 77-country, 11-sector OECD ICIO transaction matrix as a
    compressed ``.npz`` so the trade model reproduces without MATLAB and without
    any file outside the distribution. Before 4.0.0 this matrix was read from
    ``data_77c_11s.mat`` somewhere above the checkout, so every trade parity test
    silently skipped for anyone but the author.
    """
    return Path(__file__).resolve().parent / "_datafiles" / "icio_77c_11s.npz"


def bundled_workbook_path() -> Path:
    """Absolute path to the bundled MATLAB ``results.xls`` sheet values."""
    return Path(__file__).resolve().parent / "_datafiles" / "trade_results_workbook.npz"


def load_reference_workbook_sheet(sheet: str) -> np.ndarray:
    """Return one sheet of the MATLAB ``results.xls`` reference workbook.

    Verbatim cell values, stored as a string grid so the published headers
    survive, which keeps the Geary-Khamis comparison an *external* check.

    Parameters
    ----------
    sheet : str
        ``'growth GDP%'``, ``'inflaction%'`` or ``'xn_over_gd%'`` (the spelling
        is the workbook's own).
    """
    path = bundled_workbook_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled reference workbook missing at {path}; reinstall puremacro."
        )
    key = sheet.replace(" ", "_").replace("%", "pct") + "__values"
    with np.load(path, allow_pickle=False) as bundle:
        if key not in bundle.files:
            available = sorted(n.removesuffix("__values") for n in bundle.files)
            raise KeyError(f"No bundled sheet {sheet!r}; available: {available}")
        return np.asarray(bundle[key])


def bundled_reference_path() -> Path:
    """Absolute path to the bundled MATLAB-derived reference solutions."""
    return Path(__file__).resolve().parent / "_datafiles" / "trade_reference_solutions.npz"


def available_reference_scenarios() -> list[str]:
    """Tariff scenarios whose reference solution ships with puremacro."""
    path = bundled_reference_path()
    if not path.is_file():
        return []
    with np.load(path) as bundle:
        return sorted({name.split("__", 1)[0] for name in bundle.files})


def load_reference_solution(scenario: str = "base") -> dict[str, np.ndarray]:
    """Return the reference equilibrium arrays for one tariff scenario.

    These are verbatim copies of the MATLAB ``results_77c_11s_*.mat`` outputs of
    the sectoral-misallocation computation, so comparing puremacro against them
    remains an *external* check rather than a self-referential golden. They ship
    with the package, so the parity suites no longer need MATLAB or any file
    outside the distribution.

    Parameters
    ----------
    scenario : str, default 'base'
        One of :func:`available_reference_scenarios`, e.g. ``'base'`` or ``'t10'``.

    Raises
    ------
    KeyError
        If the scenario has no bundled reference. ``t10_54`` is the known gap:
        its source file is a dataless placeholder in the author's storage.
    """
    path = bundled_reference_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled trade reference solutions missing at {path}; reinstall puremacro."
        )
    with np.load(path) as bundle:
        prefix = f"{scenario}__"
        arrays = {
            name[len(prefix):]: np.asarray(bundle[name])
            for name in bundle.files
            if name.startswith(prefix)
        }
    if not arrays:
        raise KeyError(
            f"No bundled reference for scenario {scenario!r}; "
            f"available: {available_reference_scenarios()}"
        )
    return arrays


def _load_icio_array(custom_path: str | Path | None = None) -> np.ndarray:
    """Return the (850, 1078) ICIO matrix as float64.

    With no argument the bundled dataset is used. ``custom_path`` accepts a
    ``.npz`` written by numpy (key ``data``, or the single stored array) or a
    delimited text table; MATLAB ``.mat`` input was removed in 4.0.0.
    """
    if custom_path is None:
        target = bundled_icio_path()
        if not target.is_file():
            raise FileNotFoundError(
                f"Bundled ICIO dataset missing at {target}. A source checkout or "
                "wheel should always carry it; reinstall puremacro."
            )
    else:
        target = Path(custom_path)
        if target.is_dir():
            for name in ("icio_77c_11s.npz", "data_77c_11s.npz"):
                if (target / name).is_file():
                    target = target / name
                    break
        # Check the suffix before existence: someone migrating from 3.x passes a
        # .mat path, and "file not found" would send them hunting for the file
        # rather than telling them the format is gone.
        if target.suffix == ".mat":
            raise ValueError(
                "MATLAB .mat input is no longer supported (removed in 4.0.0). "
                "Convert it once with "
                "`numpy.savez_compressed(out, data=scipy.io.loadmat(src)['data'])`, "
                "or omit `path` to use the dataset bundled with puremacro."
            )
        if not target.is_file():
            raise FileNotFoundError(f"Specified ICIO data file not found: {target}")

    if target.suffix == ".npz":
        with np.load(target) as bundle:
            key = "data" if "data" in bundle.files else bundle.files[0]
            return np.asarray(bundle[key], dtype=np.float64)
    return _load_raw_table(target)


def _resolve_raw_45_path(custom_path: str | Path | None = None) -> Path:
    """Resolve absolute path to unaggregated OECD ICIO dataset (data_2020_SML.csv)."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_dir():
            for name in ("data_2020_SML.csv", "2020_SML.csv", "2020.SML.csv"):
                candidate = p / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No OECD ICIO CSV found in directory: {p}")
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified ICIO raw data file not found: {p}")

    # Check environment variables
    env_vars = ["IO_RAW45_PATH", "IO_DATA_PATH", "IO_COMPUTATION_DIR"]
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.is_dir():
                for name in ("data_2020_SML.csv", "2020_SML.csv", "2020.SML.csv"):
                    candidate = p / name
                    if candidate.exists():
                        return candidate
            elif p.exists():
                return p

    # Canonical search candidates
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "IO" / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv",
        here.parents[2] / "IO" / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv",
        here.parents[4] / "IO" / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv",
        Path.cwd() / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv",
        Path.cwd() / "IO" / "computation" / "7_TIO_77c_vf" / "data_2020_SML.csv",
        here.parents[3] / "IO" / "ICIOextended" / "2020_SML.csv",
        here.parents[2] / "IO" / "ICIOextended" / "2020_SML.csv",
        here.parents[4] / "IO" / "ICIOextended" / "2020_SML.csv",
        Path.cwd() / "ICIOextended" / "2020_SML.csv",
        Path.cwd() / "IO" / "ICIOextended" / "2020_SML.csv",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not automatically locate 'data_2020_SML.csv'. Please specify the file "
        "path via `load_raw_45sector_icio(path=...)` or set the IO_RAW45_PATH environment variable."
    )


def load_raw_45sector_icio(
    path: str | Path | None = None,
    *,
    regularize: bool = True,
    return_structured: bool = False,
    nc: int = 77,
    ns: int = 45,
    nfd0: int = 6,
    nfd: int = 3,
) -> np.ndarray | ICIOData:
    """Load and process the unaggregated 45-sector OECD ICIO dataset.

    Loads the raw 2020 OECD ICIO table (3,468 x 3,928) from ``data_2020_SML.csv``,
    condenses the 6 final demand categories to 3 (C, I, Cx), applies economic
    regularization (flooring inactive sector output y >= 1e-6 and negative VA
    VA >= max(1e-4*y, 1.0) with residual TLS absorption), splits value added into
    labor (2/3) and capital (1/3), and returns the calibrated (3,468, 3,696) matrix.

    Parameters
    ----------
    path : str or Path, optional
        Path to ``data_2020_SML.csv`` or ``2020_SML.csv``. If omitted, searches default locations.
    regularize : bool, default True
        If True, applies economic regularizations ensuring non-negative output and VA,
        with exact Walrasian accounting (Sales == Outlays via residual net production tax TLS).
        If False, returns unregularized factor split without floors.
    return_structured : bool, default False
        If True, returns an :class:`ICIOData` container exposing slicing properties.
        If False, returns the raw float64 array of shape (3468, 3696).
    nc : int, default 77
        Number of countries.
    ns : int, default 45
        Number of raw industrial sectors.
    nfd0 : int, default 6
        Number of raw final demand categories.
    nfd : int, default 3
        Number of condensed final demand categories.

    Returns
    -------
    np.ndarray or ICIOData
        Processed ICIO transaction matrix of shape (3468, 3696).
    """
    csv_file = _resolve_raw_45_path(path)
    raw = _load_raw_table(csv_file)
    n_ind = nc * ns

    if raw.shape[0] < n_ind + 2:
        raise ValueError(
            f"Raw matrix has {raw.shape[0]} rows; expected at least {n_ind + 2} for nc={nc}, ns={ns}."
        )

    # Slice intermediate and final demand blocks
    # Rows 0..n_ind-1: intermediate flows
    # Row n_ind: TLS (net taxes less subsidies)
    # Row n_ind+1: VA (value added at basic prices)
    data = raw[:n_ind + 2, :n_ind + nfd0 * nc].copy()

    # 1. Condense 6 final demand categories to 3:
    # C = HFCE + NPISH + GGFC (cols 0, 1, 2)
    # I = GFCF + INVNT (cols 3, 4)
    # Cx = DPABR (col 5)
    fd = np.zeros((n_ind + 2, nfd * nc), dtype=np.float64)
    for k1 in range(nc):
        fd[:, 0 + nfd * k1] = np.sum(data[:, n_ind + nfd0 * k1 : n_ind + nfd0 * k1 + 3], axis=1)
        fd[:, 1 + nfd * k1] = np.sum(data[:, n_ind + nfd0 * k1 + 3 : n_ind + nfd0 * k1 + 5], axis=1)
        fd[:, 2 + nfd * k1] = data[:, n_ind + nfd0 * k1 + 5]

    data_condensed = np.hstack([data[:, :n_ind], fd])

    if regularize:
        # 2. Regularization of zero sales and negative VA
        sales = np.sum(data_condensed[:n_ind, :], axis=1)
        for ik in range(nc):
            for isec in range(ns):
                idx = isec + ns * ik
                s_val = sales[idx]
                va_val = data_condensed[n_ind + 1, idx]
                if s_val == 0.0:
                    s_val = 1e-6
                    va_val = 1e-6
                    data_condensed[n_ind + 1, idx] = va_val
                else:
                    min_va = max(0.05 * s_val, 1.0)
                    if va_val < min_va:
                        va_val = min_va
                        data_condensed[n_ind + 1, idx] = va_val
                purch = np.sum(data_condensed[:n_ind, idx])
                data_condensed[n_ind, idx] = s_val - purch - va_val

    # 3. Factor split: Labor (2/3 VA) and Capital (1/3 VA)
    va_row = data_condensed[n_ind + 1, :]
    labor = (2.0 / 3.0) * va_row
    capital = (1.0 / 3.0) * va_row
    clean_matrix = np.vstack([data_condensed[:n_ind + 1, :], labor, capital])

    if return_structured:
        return ICIOData(
            matrix=clean_matrix,
            country_codes=CANONICAL_COUNTRY_CODES if nc == 77 else tuple(f"C{i:02d}" for i in range(nc)),
            sector_codes=RAW_45_SECTOR_CODES if ns == 45 else tuple(f"S{i:02d}" for i in range(ns)),
            fd_codes=CANONICAL_FINAL_DEMAND_CODES if nfd == 3 else tuple(f"FD{i:02d}" for i in range(nfd)),
        )

    return clean_matrix


def load_icio_data(
    path: str | Path | None = None,
    *,
    return_structured: bool = False,
    sectors: int = 11,
    aggregate_sectors: bool = True,
    regularize: bool = True,
) -> np.ndarray | ICIOData:
    """Load the canonical 77-country ICIO dataset (11-sector or raw 45-sector).

    Parameters
    ----------
    path : str or Path, optional
        Path to a ``.npz`` or delimited text table. If omitted, the dataset
        bundled with puremacro is used, so no file outside the installation is
        needed. MATLAB ``.mat`` input was removed in 4.0.0.
    return_structured : bool, default False
        If True, returns an :class:`ICIOData` container exposing slicing properties.
        If False, returns the raw float64 array of shape (850, 1078) or (3468, 3696).
    sectors : int, default 11
        Number of industrial sectors (11 for composite baseline, 45 for unaggregated OECD ICIO).
    aggregate_sectors : bool, default True
        If False, loads the unaggregated 45-sector OECD ICIO dataset.
    regularize : bool, default True
        If True, applies economic regularization to 45-sector data (non-negativity and residual TLS).

    Returns
    -------
    np.ndarray or ICIOData
        The loaded ICIO transaction matrix.
    """
    is_45 = (sectors == 45) or (not aggregate_sectors)
    if path is not None:
        p_str = str(path)
        if p_str.endswith(".csv") or "2020" in p_str or "45s" in p_str:
            is_45 = True

    if is_45:
        return load_raw_45sector_icio(
            path=path,
            regularize=regularize,
            return_structured=return_structured,
        )

    raw_data = _load_icio_array(path)

    if return_structured:
        return ICIOData(
            matrix=raw_data,
            country_codes=CANONICAL_COUNTRY_CODES,
            sector_codes=CANONICAL_SECTOR_CODES,
            fd_codes=CANONICAL_FINAL_DEMAND_CODES,
        )

    return raw_data


__all__ = [
    "bundled_icio_path",
    "bundled_reference_path",
    "bundled_workbook_path",
    "load_reference_workbook_sheet",
    "available_reference_scenarios",
    "load_reference_solution",
    "CANONICAL_COUNTRY_CODES",
    "EU_COUNTRY_CODES",
    "CANONICAL_SECTOR_CODES",
    "CANONICAL_SECTOR_NAMES",
    "RAW_45_SECTOR_CODES",
    "RAW_45_SECTOR_NAMES",
    "CANONICAL_FINAL_DEMAND_CODES",
    "RAW_6_FINAL_DEMAND_CODES",
    "get_country_codes",
    "get_eu_country_codes",
    "get_sector_codes",
    "get_sector_names",
    "get_raw_45_sector_codes",
    "get_raw_45_sector_names",
    "get_final_demand_codes",
    "get_raw_final_demand_codes",
    "ICIOData",
    "load_icio_data",
    "load_raw_45sector_icio",
]
