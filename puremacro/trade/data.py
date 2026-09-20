"""OECD ICIO Data Ingestion and Regional/Sectoral Indexing.

This module provides data loading, canonical identifier registries, and slice
abstractions for the 77-country 11-sector empirical Input-Output dataset
originating from the OECD Inter-Country Input-Output (ICIO) tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.regularize import regularize_mrio_table

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

# ---------------------------------------------------------------------------
# Authentic Database Rosters (FIGARO, EXIOBASE, WIOD, Eora, OECD)
# ---------------------------------------------------------------------------

FIGARO_COUNTRIES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
    "ARG", "AUS", "BRA", "CAN", "CHE", "CHN", "GBR", "IDN", "IND", "JPN",
    "KOR", "MEX", "NOR", "RUS", "SAU", "TUR", "USA", "ZAF", "W2",
)

FIGARO_64_SECTORS: tuple[str, ...] = (
    "A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18",
    "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28",
    "C29", "C30", "C31_C32", "C33", "D35", "E36", "E37-E39", "F", "G45",
    "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60",
    "J61", "J62_J63", "K64", "K65", "K66", "L68", "M69_M70", "M71",
    "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82", "O84", "P85",
    "Q86", "Q87_Q88", "R90-R92", "R93", "S94", "S95", "S96", "T", "U",
)

FIGARO_FD_CATEGORIES: tuple[str, ...] = (
    "P3_S14", "P3_S15", "P3_S13", "P51G", "P5M",
)

EXIOBASE_COUNTRIES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
    "AUS", "BRA", "CAN", "CHE", "CHN", "IDN", "IND", "JPN", "KOR", "MEX",
    "NOR", "RUS", "TUR", "TWN", "USA", "ZAF",
    "WA", "WE", "WF", "WL", "WM",
)

EXIOBASE_163_SECTORS: tuple[str, ...] = tuple(f"i{i:03d}" for i in range(1, 164))
EXIOBASE_200_SECTORS: tuple[str, ...] = tuple(f"p{i:03d}" for i in range(1, 201))
EXIOBASE_FD_CATEGORIES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "VALUABLES", "EXPORT",
)

WIOD_44_COUNTRIES: tuple[str, ...] = (
    "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHN", "CYP", "CZE",
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN",
    "IDN", "IND", "IRL", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX",
    "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE",
    "TUR", "TWN", "USA", "ROW",
)

WIOD_56_SECTORS: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(1, 57))
WIOD_FD_CATEGORIES: tuple[str, ...] = (
    "CONS_h", "CONS_np", "CONS_g", "GFCF", "INVT",
)

EORA_189_COUNTRIES: tuple[str, ...] = (
    "AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "ABW", "AUS",
    "AUT", "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN",
    "BMU", "BTN", "BOL", "BIH", "BWA", "BRA", "VGB", "BRN", "BGR", "BFA",
    "BDI", "KHM", "CMR", "CAN", "CPV", "CYM", "CAF", "TCD", "CHL", "CHN",
    "COL", "COG", "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI",
    "DOM", "ECU", "EGY", "SLV", "ERI", "EST", "ETH", "FJI", "FIN", "FRA",
    "PYF", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRL", "GTM", "GIN",
    "GUY", "HTI", "HND", "HKG", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ",
    "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KWT", "KGZ",
    "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", "MAC",
    "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MRT", "MUS", "MEX", "MDA",
    "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NPL", "NLD", "ANT",
    "NCL", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN", "PAK",
    "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "PSE", "QAT", "KOR",
    "ROU", "RUS", "RWA", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC",
    "SLE", "SGP", "SVK", "SVN", "SOM", "ZAF", "SSD", "ESP", "LKA", "SDN",
    "SUD", "SUR", "SWZ", "SWE", "CHE", "SYR", "TWN", "TJK", "TZA", "THA",
    "TGO", "TTO", "TUN", "TUR", "TKM", "UGA", "UKR", "ARE", "GBR", "USA",
    "URY", "UZB", "VUT", "VEN", "VNM", "YEM", "ZMB", "ZWE", "ROW",
)

EORA_26_SECTORS: tuple[str, ...] = tuple(f"SEC{i:02d}" for i in range(1, 27))
EORA_FD_CATEGORIES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "ACQ_VAL",
)

OECD_77_COUNTRIES: tuple[str, ...] = CANONICAL_COUNTRY_CODES
OECD_45_SECTORS: tuple[str, ...] = RAW_45_SECTOR_CODES
OECD_FD_CATEGORIES: tuple[str, ...] = RAW_6_FINAL_DEMAND_CODES


@dataclass
class RawIOData:
    """Container for unaggregated/raw multi-country input-output transaction matrices."""

    countries: Sequence[str]
    sectors: Sequence[str]
    intermediate_matrix: np.ndarray  # (M, M)
    final_demand_matrix: np.ndarray  # (M, C * K_F) or (M, C, K_F)
    value_added: np.ndarray  # (M,) or (K_VA, M)
    taxes_less_subsidies: np.ndarray  # (M,)
    gross_output: np.ndarray  # (M,)
    fd_categories: Sequence[str]
    year: int = 0
    source: str = ""
    unit: str = "M_USD"
    taxes_less_subsidies_fd: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def C(self) -> int:
        return len(self.countries)

    @property
    def S(self) -> int:
        return len(self.sectors)

    @property
    def M(self) -> int:
        return self.C * self.S

    @property
    def K_F(self) -> int:
        return len(self.fd_categories)

    @property
    def Z(self) -> np.ndarray:
        return self.intermediate_matrix

    @property
    def F(self) -> np.ndarray:
        return self.final_demand_matrix

    @property
    def VA(self) -> np.ndarray:
        return self.value_added

    @property
    def TLS(self) -> np.ndarray:
        return self.taxes_less_subsidies

    @property
    def Y(self) -> np.ndarray:
        return self.gross_output



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


# ---------------------------------------------------------------------------
# High-Fidelity Deterministic Synthetic MRIO Generator
# ---------------------------------------------------------------------------


def generate_synthetic_mrio(
    dataset: str,
    year: int = 2019,
    seed: int = 42,
    n_inactive: int = 0,
    n_neg_va: int = 0,
    custom_c: int | None = None,
    custom_s: int | None = None,
    model: str = "ixi",
    unit: str = "M_USD",
) -> RawIOData:
    """Generate a high-fidelity deterministic synthetic Multi-Regional Input-Output table.

    Parameters
    ----------
    dataset : str
        Target database: 'figaro', 'exiobase', 'wiod', 'eora', or 'oecd' / 'oecd_icio'.
    year : int, default=2019
        Reference year.
    seed : int, default=42
        Deterministic random seed.
    n_inactive : int, default=0
        Number of inactive sector nodes (zero sales and outlays) to inject for testing.
    n_neg_va : int, default=0
        Number of negative value-added sector nodes to inject for testing.
    custom_c : int, optional
        Override number of economies for fast testing.
    custom_s : int, optional
        Override number of sectors for fast testing.
    model : str, default='ixi'
        EXIOBASE model formulation ('ixi' or 'pxp').
    unit : str, default='M_USD'
        Unit of accounts ('M_USD', 'M_EUR', or '000_USD').

    Returns
    -------
    RawIOData
        Container encapsulating synthetic MRIO transaction matrices.
    """
    ds = dataset.lower().strip()
    if ds in ("figaro", "eurostat_figaro"):
        full_countries = list(FIGARO_COUNTRIES)
        full_sectors = list(FIGARO_64_SECTORS)
        fd_categories = list(FIGARO_FD_CATEGORIES)
    elif ds in ("exiobase", "exiobase3"):
        full_countries = list(EXIOBASE_COUNTRIES)
        full_sectors = list(EXIOBASE_200_SECTORS) if model == "pxp" else list(EXIOBASE_163_SECTORS)
        fd_categories = list(EXIOBASE_FD_CATEGORIES)
    elif ds in ("wiod", "wiod2016"):
        full_countries = list(WIOD_44_COUNTRIES)
        full_sectors = list(WIOD_56_SECTORS)
        fd_categories = list(WIOD_FD_CATEGORIES)
    elif ds in ("eora", "eora26"):
        full_countries = list(EORA_189_COUNTRIES)
        full_sectors = list(EORA_26_SECTORS)
        fd_categories = list(EORA_FD_CATEGORIES)
    elif ds in ("oecd", "oecd_icio", "icio"):
        full_countries = list(OECD_77_COUNTRIES)
        full_sectors = list(OECD_45_SECTORS)
        fd_categories = list(OECD_FD_CATEGORIES)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: 'figaro', 'exiobase', 'wiod', 'eora', 'oecd'."
        )

    countries = full_countries[:custom_c] if (custom_c is not None and custom_c > 0) else full_countries
    sectors = full_sectors[:custom_s] if (custom_s is not None and custom_s > 0) else full_sectors

    C = len(countries)
    S = len(sectors)
    M = C * S
    K_F = len(fd_categories)

    rng = np.random.default_rng(seed)

    # 1. Base economic size distribution (lognormal GDP)
    base_gdp = rng.lognormal(mean=8.0, sigma=1.0, size=C)
    if unit == "000_USD":
        base_gdp *= 1000.0
    for c_idx, c_code in enumerate(countries):
        if c_code in ("USA", "CHN"):
            base_gdp[c_idx] *= 8.0
        elif c_code in ("DEU", "JPN", "GBR", "FRA"):
            base_gdp[c_idx] *= 4.0
        elif c_code in ("ITA", "CAN", "KOR", "IND", "BRA", "RUS"):
            base_gdp[c_idx] *= 2.5

    # 2. Sectoral weight distribution
    sector_weights = rng.uniform(0.5, 2.0, size=S)
    sector_weights /= np.sum(sector_weights)

    Y_target_2d = np.outer(base_gdp, sector_weights)  # (C, S)
    Y_target = Y_target_2d.reshape(M)

    # 3. Bilateral trade gravity matrix with domestic preference
    dist = rng.uniform(1.0, 3.0, size=(C, C))
    np.fill_diagonal(dist, 0.2)
    friction = np.exp(-0.8 * dist)
    gravity = base_gdp[:, None] * friction

    if C > 1:
        log_gdp = np.log(base_gdp)
        log_range = float(np.max(log_gdp) - np.min(log_gdp))
        gdp_norm = (log_gdp - np.min(log_gdp)) / log_range if log_range > 1e-12 else np.zeros(C)
        omega_dom = 0.70 + 0.18 * gdp_norm

        for d in range(C):
            off_sum = float(np.sum(gravity[:, d]) - gravity[d, d])
            if off_sum > 0:
                gravity[d, d] = off_sum * (omega_dom[d] / (1.0 - omega_dom[d]))

    trade_share = gravity / np.sum(gravity, axis=0, keepdims=True)  # (C_orig, C_dest)

    # 4. Intermediate Technical Coefficients & Flows
    inter_cost_share = rng.uniform(0.42, 0.52, size=M)
    va_cost_share = rng.uniform(0.42, 0.50, size=M)

    io_tech = rng.uniform(0.05, 0.35, size=(S, S))
    np.fill_diagonal(io_tech, np.diag(io_tech) + 0.4)
    io_tech /= np.sum(io_tech, axis=0, keepdims=True)

    total_purchases = (inter_cost_share * Y_target).reshape(C, S)
    input_flows = io_tech[None, :, :] * total_purchases[:, None, :]
    Z_4d = trade_share[:, None, :, None] * np.transpose(input_flows, (1, 0, 2))[None, :, :, :]
    Z = Z_4d.reshape(M, M)

    # 5. Final Demand Flows
    fd_cat_weights = np.array([0.58, 0.04, 0.18, 0.15, 0.03, 0.01, 0.01][:K_F], dtype=np.float64)
    fd_cat_weights /= np.sum(fd_cat_weights)

    good_fd = (base_gdp * 0.75)[:, None] * sector_weights[None, :]
    cat_flow = good_fd[:, :, None] * fd_cat_weights[None, None, :]
    F_4d = trade_share[:, None, :, None] * np.transpose(cat_flow, (1, 0, 2))[None, :, :, :]
    F_2d = F_4d.reshape(M, C * K_F)

    # 6. Gross Output from Row Sales
    Y = np.sum(Z, axis=1) + np.sum(F_2d, axis=1)

    # 7. Value Added & Residual TLS
    VA = va_cost_share * Y
    TLS = Y - np.sum(Z, axis=0) - VA

    # 8. Test Edge Cases Injection
    if n_inactive > 0:
        inactive_indices = [(step * S) % M for step in range(n_inactive)]
        for idx in inactive_indices:
            Z[idx, :] = 0.0
            Z[:, idx] = 0.0
            F_2d[idx, :] = 0.0
            VA[idx] = 0.0
            TLS[idx] = 0.0
            Y[idx] = 0.0
        # Reconcile Y and TLS for remaining sectors so row sales and column outlays match
        Y = np.sum(Z, axis=1) + np.sum(F_2d, axis=1)
        for idx in inactive_indices:
            Y[idx] = 0.0
        TLS = Y - np.sum(Z, axis=0) - VA
        for idx in inactive_indices:
            TLS[idx] = 0.0

    if n_neg_va > 0:
        start_idx = (n_inactive * S + 1) % M
        neg_indices = [(start_idx + step * 2) % M for step in range(n_neg_va)]
        for idx in neg_indices:
            VA[idx] = -50.0

    return RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F_2d,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=fd_categories,
        year=year,
        source=f"Synthetic_{dataset.upper()}",
        unit=unit,
        metadata={"is_synthetic": True, "seed": seed, "generator": "generate_synthetic_mrio", "model": model},
    )


# ---------------------------------------------------------------------------
# Calibration Pipeline Packaging
# ---------------------------------------------------------------------------


def package_mrio_to_calibration_result(
    raw: RawIOData,
    regularize: bool = True,
    validate: bool = True,
) -> TradeCalibrationResult:
    """Package RawIOData flows into a canonical TradeCalibrationResult.

    Optionally regularizes flows via ``regularize_mrio_table``, decomposes value added
    into labor (2/3) and capital (1/3), constructs the calibrated transaction table,
    and runs ``calibrate_trade_model`` with invariant validation.

    Parameters
    ----------
    raw : RawIOData
        Container with unaggregated transaction matrices.
    regularize : bool, default=True
        Whether to apply phantom output injection, VA flooring with dual TLS debit,
        and residual TLS reconciliation.
    validate : bool, default=True
        Whether to enforce budget balance and table reconstruction checks.

    Returns
    -------
    TradeCalibrationResult
        Calibrated trade model result container.
    """
    from puremacro.trade.calibration import calibrate_trade_model

    C, S, K_F = raw.C, raw.S, raw.K_F
    M = C * S

    Z = np.array(raw.intermediate_matrix, dtype=np.float64, copy=True)
    F = np.array(raw.final_demand_matrix, dtype=np.float64, copy=True)
    VA = np.array(raw.value_added, dtype=np.float64, copy=True)
    TLS = np.array(raw.taxes_less_subsidies, dtype=np.float64, copy=True)
    Y = np.array(raw.gross_output, dtype=np.float64, copy=True)

    expected_shapes = {"Z": (M, M), "F": (M, C * K_F), "TLS": (M,), "Y": (M,)}
    F = F.reshape(M, C * K_F) if F.ndim == 3 else F
    for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("Y", Y)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite data")
        if name in expected_shapes and arr.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape {arr.shape} != {expected_shapes[name]}")
    if VA.shape not in ((M,),) and not (VA.ndim == 2 and VA.shape[1] == M):
        raise ValueError("Value added must have one column per country-sector node")
    if len(set(raw.countries)) != C or len(set(raw.sectors)) != S:
        raise ValueError("Country and sector labels must be unique")
    before = {name: arr.copy() for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("Y", Y))}
    if regularize:
        # Provider OUT margins can differ from the released transactions.
        # Calibrate to actual sales and record the change from reported Y;
        # otherwise TLS repair leaves consumer budgets unbalanced.
        Y = Z.sum(axis=1) + F.sum(axis=1)
        Z, F, VA, TLS, Y = regularize_mrio_table(
            Z, F, VA, TLS, Y, n_countries=C, n_sectors=S
        )

    va_row = np.sum(VA, axis=0) if VA.ndim == 2 else VA
    labor = (2.0 / 3.0) * va_row
    capital = (1.0 / 3.0) * va_row

    if getattr(raw, "taxes_less_subsidies_fd", None) is not None:
        tfd = np.array(raw.taxes_less_subsidies_fd, dtype=np.float64, copy=True).reshape(C * K_F)
        TLS_full = np.hstack([TLS, tfd])
    else:
        TLS_full = np.hstack([TLS, np.zeros(C * K_F)])
    labor_full = np.hstack([labor, np.zeros(C * K_F)])
    capital_full = np.hstack([capital, np.zeros(C * K_F)])

    F_2d = F.reshape(M, C * K_F) if F.ndim == 3 else F
    matrix = np.vstack([
        np.hstack([Z, F_2d]),
        TLS_full,
        labor_full,
        capital_full,
    ])

    result = calibrate_trade_model(
        matrix,
        ns=S,
        nc=C,
        nfd=K_F,
        country_codes=raw.countries,
        sector_codes=raw.sectors,
        validate=validate,
    )


    adjustments = {
        name: {"changed_entries": int(np.count_nonzero(arr != before[name])),
               "max_abs_change": float(np.max(np.abs(arr - before[name]), initial=0.0)),
               "net_change": float(np.sum(arr - before[name]))}
        for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("Y", Y))
    }
    return replace(result, metadata={
        **result.metadata, **raw.metadata,
        "source": raw.source, "year": raw.year, "unit": raw.unit,
        "schema": raw.metadata.get("schema", "RawIOData country-major sector ordering"),
        "is_synthetic": bool(raw.metadata.get("is_synthetic", raw.source.startswith("Synthetic_"))),
        "regularized": regularize, "adjustments": adjustments,
        "output_reconciliation": "transaction row sales" if regularize else "provider output",
        "labor_share_assumption": 2.0 / 3.0, "capital_share_assumption": 1.0 / 3.0,
        "fd_categories": tuple(raw.fd_categories),
    })


# ---------------------------------------------------------------------------
# Harmonized Ingestion Adapters for 5 Major Databases
# ---------------------------------------------------------------------------


def load_figaro(
    year: int = 2019,
    eur_to_usd: float = 1.1195,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load Eurostat FIGARO table with automatic EUR -> USD currency harmonization.

    Schema: 46 countries x 64 NACE Rev. 2 sectors = 2,944 nodes, 5 final demand categories.

    Parameters
    ----------
    year : int, default=2019
        Reference year.
    eur_to_usd : float, default=1.1195
        Assumed USD per EUR conversion factor. The default is a 2019 convention;
        supply the appropriate factor explicitly for other years.
    fallback_to_synthetic : bool, default=False
        Explicitly opt into generated teaching data if the archive is missing.
    data_dir : str or Path, optional
        Directory containing FIGARO CSV/TSV files.
    file_path : str or Path, optional
        Exact path to FIGARO table file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_eur = generate_synthetic_mrio("figaro", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_EUR")
        raw_usd = RawIOData(
            countries=raw_eur.countries,
            sectors=raw_eur.sectors,
            intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
            final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
            value_added=raw_eur.value_added * eur_to_usd,
            taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
            gross_output=raw_eur.gross_output * eur_to_usd,
            fd_categories=raw_eur.fd_categories,
            year=year,
            source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(raw_usd, regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_FIGARO_DIR"]) ] if os.environ.get("IO_FIGARO_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        for d in candidates:
            if d.is_dir():
                matches = list(d.glob(f"*{year}*.csv")) + list(d.glob(f"*{year}*.tsv"))
                if matches:
                    target = matches[0]
                    break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_eur = generate_synthetic_mrio("figaro", year=year, seed=seed, unit="M_EUR")
            raw_usd = RawIOData(
                countries=raw_eur.countries,
                sectors=raw_eur.sectors,
                intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
                final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
                value_added=raw_eur.value_added * eur_to_usd,
                taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
                gross_output=raw_eur.gross_output * eur_to_usd,
                fd_categories=raw_eur.fd_categories,
                year=year,
                source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(raw_usd, regularize=regularize)
        raise FileNotFoundError(f"Eurostat FIGARO table for year {year} not found.")

    sep = "\t" if str(target).endswith(".tsv") else ","
    df = pd.read_csv(target, sep=sep, index_col=0, low_memory=False)

    # Supported interchange schema: labeled country-major intermediate rows
    # and columns, five final-demand categories per country, then TLS and VA.
    # Native Eurostat archives require conversion to this explicit layout.
    if len(df) < 3 or [str(x).upper() for x in df.index[-2:]] != ["TLS", "VA"]:
        raise ValueError("FIGARO harmonized table requires final TLS and VA rows")
    M = len(df) - 2
    int_cols = list(df.columns[:M])
    if any("_" not in label for label in int_cols):
        raise ValueError("Intermediate columns must be labeled COUNTRY_SECTOR")
    countries = list(dict.fromkeys(label.split("_", 1)[0] for label in int_cols))
    sectors = list(dict.fromkeys(label.split("_", 1)[1] for label in int_cols))
    N, S = len(countries), len(sectors)
    expected = [f"{c}_{sector}" for c in countries for sector in sectors]
    if int_cols != expected or list(df.index[:M]) != expected:
        raise ValueError("Intermediate rows and columns must share complete country-major labels")
    n_fd = N * len(FIGARO_FD_CATEGORIES)
    if df.shape[1] != M + n_fd:
        raise ValueError("Final-demand columns must contain five categories per country")
    fd_cols = df.columns[M:]
    if [str(c).split("_", 1)[0] for c in fd_cols] != [c for c in countries for _ in FIGARO_FD_CATEGORIES]:
        raise ValueError("Final-demand columns must have matching country-major labels")
    Z = df.iloc[:M, :M].to_numpy(dtype=float) * eur_to_usd
    F = df.iloc[:M, M:].to_numpy(dtype=float) * eur_to_usd
    TLS = df.iloc[M, :M].to_numpy(dtype=float) * eur_to_usd
    TLS_fd = df.iloc[M, M:].to_numpy(dtype=float) * eur_to_usd
    VA = df.iloc[M + 1, :M].to_numpy(dtype=float) * eur_to_usd
    OUT = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=OUT,
        fd_categories=list(FIGARO_FD_CATEGORIES),
        year=year,
        source="Eurostat_FIGARO_Harmonized",
        unit="M_USD",
        taxes_less_subsidies_fd=TLS_fd,
    )
    raw.metadata.update({
        "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR",
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(raw, regularize=regularize)


def load_exiobase(
    year: int = 2019,
    model: str = "ixi",
    eur_to_usd: float = 1.1195,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load EXIOBASE 3 table with automatic EUR -> USD currency harmonization.

    Schema: 49 regions x 163 industries (ixi) or 200 products (pxp), 7 final demand categories.

    Parameters
    ----------
    year : int, default=2019
        Reference year.
    model : str, default='ixi'
        Model formulation: 'ixi' (industry-by-industry) or 'pxp' (product-by-product).
    eur_to_usd : float, default=1.1195
        Assumed USD per EUR conversion factor. The default is a 2019 convention;
        supply the appropriate factor explicitly for other years.
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing EXIOBASE files.
    file_path : str or Path, optional
        Exact path to EXIOBASE file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_eur = generate_synthetic_mrio("exiobase", year=year, model=model, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_EUR")
        raw_usd = RawIOData(
            countries=raw_eur.countries,
            sectors=raw_eur.sectors,
            intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
            final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
            value_added=raw_eur.value_added * eur_to_usd,
            taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
            gross_output=raw_eur.gross_output * eur_to_usd,
            fd_categories=raw_eur.fd_categories,
            year=year,
            source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(raw_usd, regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_EXIOBASE_DIR"]) ] if os.environ.get("IO_EXIOBASE_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        for d in candidates:
            if d.is_dir():
                matches = list(d.glob(f"*{model}*{year}*")) + list(d.glob(f"*{year}*{model}*"))
                if matches:
                    target = matches[0]
                    break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_eur = generate_synthetic_mrio("exiobase", year=year, model=model, seed=seed, unit="M_EUR")
            raw_usd = RawIOData(
                countries=raw_eur.countries,
                sectors=raw_eur.sectors,
                intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
                final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
                value_added=raw_eur.value_added * eur_to_usd,
                taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
                gross_output=raw_eur.gross_output * eur_to_usd,
                fd_categories=raw_eur.fd_categories,
                year=year,
                source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(raw_usd, regularize=regularize)
        raise FileNotFoundError(f"EXIOBASE 3 ({model}) table for year {year} not found.")

    df = pd.read_csv(target, sep="\t", header=None, low_memory=False)
    N = 49
    S = 200 if model == "pxp" else 163
    M = N * S
    n_fd_cols = N * len(EXIOBASE_FD_CATEGORIES)

    Z = df.iloc[:M, :M].values.astype(np.float64) * eur_to_usd
    F = df.iloc[:M, M : M + n_fd_cols].values.astype(np.float64) * eur_to_usd

    va_raw = df.iloc[M : M + 7, :M].values.astype(np.float64) * eur_to_usd
    VA = np.sum(va_raw, axis=0)
    TLS = np.zeros(M, dtype=np.float64)
    Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=list(EXIOBASE_COUNTRIES),
        sectors=list(EXIOBASE_200_SECTORS if model == "pxp" else EXIOBASE_163_SECTORS),
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(EXIOBASE_FD_CATEGORIES),
        year=year,
        source=f"EXIOBASE3_{model.upper()}_Harmonized",
        unit="M_USD",
    )
    raw.metadata.update({
        "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR",
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(raw, regularize=regularize)


def load_wiod(
    year: int = 2014,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load World Input-Output Database (WIOD) 2016 release table.

    Schema: 44 countries x 56 sectors = 2,464 nodes, 5 final demand categories.

    Parameters
    ----------
    year : int, default=2014
        Reference year (2000 to 2014).
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing WIOD Stata/Excel files.
    file_path : str or Path, optional
        Exact path to WIOD table file (.dta or .csv).
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw = generate_synthetic_mrio("wiod", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_USD")
        return package_mrio_to_calibration_result(raw, regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_WIOD_DIR"]) ] if os.environ.get("IO_WIOD_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        for d in candidates:
            if d.is_dir():
                matches = list(d.glob(f"*{year}*.dta")) + list(d.glob(f"*{year}*.csv"))
                if matches:
                    target = matches[0]
                    break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw = generate_synthetic_mrio("wiod", year=year, seed=seed, unit="M_USD")
            return package_mrio_to_calibration_result(raw, regularize=regularize)
        raise FileNotFoundError(f"WIOD table for year {year} not found.")

    if str(target).endswith(".dta"):
        df = pd.read_stata(target)
    else:
        df = pd.read_csv(target, low_memory=False)

    all_countries = df["Country"].dropna().unique()
    countries = [str(c) for c in all_countries if str(c) not in ("TOT", "")]
    N = len(countries)
    S = 56
    M = N * S

    int_cols: list[str] = []
    for c in countries:
        for r in range(1, S + 1):
            int_cols.append(f"v{c}{r}")

    fd_cols: list[str] = []
    for c in countries:
        for r in range(57, 62):
            fd_cols.append(f"v{c}{r}")

    sectors = [f"S{r:02d}" for r in range(1, S + 1)]

    Z = df.iloc[:M][int_cols].values.astype(np.float64)
    F = df.iloc[:M][fd_cols].values.astype(np.float64)

    va_mask = (df["IndustryCode"] == "VA")
    txsp_mask = (df["IndustryCode"] == "TXSP")
    go_mask = (df["IndustryCode"] == "GO")

    if not va_mask.any() or not txsp_mask.any() or not go_mask.any():
        raise ValueError("Accounting rows (VA, TXSP, GO) missing in WIOD table.")

    va_idx = df[va_mask].index[0]
    txsp_idx = df[txsp_mask].index[0]
    go_idx = df[go_mask].index[0]

    VA = df.iloc[va_idx][int_cols].values.astype(np.float64)
    TLS = df.iloc[txsp_idx][int_cols].values.astype(np.float64)
    Y = df.iloc[go_idx][int_cols].values.astype(np.float64)

    raw = RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(WIOD_FD_CATEGORIES),
        year=year,
        source="WIOD_2016",
        unit="M_USD",
    )
    raw.metadata.update({
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(raw, regularize=regularize)


def load_eora(
    year: int = 2015,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load Eora26 MRIO table with unit harmonization from Thousand USD to Million USD.

    Schema: 189 countries x 26 sectors = 4,914 nodes, 6 final demand categories.

    Parameters
    ----------
    year : int, default=2015
        Reference year.
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing Eora files.
    file_path : str or Path, optional
        Exact path to Eora table file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    scale = 1.0 / 1000.0
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_th = generate_synthetic_mrio("eora", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="000_USD")
        raw = RawIOData(
            countries=raw_th.countries,
            sectors=raw_th.sectors,
            intermediate_matrix=raw_th.intermediate_matrix * scale,
            final_demand_matrix=raw_th.final_demand_matrix * scale,
            value_added=raw_th.value_added * scale,
            taxes_less_subsidies=raw_th.taxes_less_subsidies * scale,
            gross_output=raw_th.gross_output * scale,
            fd_categories=raw_th.fd_categories,
            year=year,
            source=raw_th.source,
            metadata={**raw_th.metadata, "unit_scale": scale, "original_unit": "000_USD"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(raw, regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_EORA_DIR"]) ] if os.environ.get("IO_EORA_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        for d in candidates:
            if d.is_dir():
                matches = list(d.glob(f"*Eora26*{year}*.txt")) + list(d.glob(f"*Eora26*{year}*.csv"))
                if matches:
                    target = matches[0]
                    break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_th = generate_synthetic_mrio("eora", year=year, seed=seed, unit="000_USD")
            raw = RawIOData(
                countries=raw_th.countries,
                sectors=raw_th.sectors,
                intermediate_matrix=raw_th.intermediate_matrix * scale,
                final_demand_matrix=raw_th.final_demand_matrix * scale,
                value_added=raw_th.value_added * scale,
                taxes_less_subsidies=raw_th.taxes_less_subsidies * scale,
                gross_output=raw_th.gross_output * scale,
                fd_categories=raw_th.fd_categories,
                year=year,
                source=raw_th.source,
            metadata={**raw_th.metadata, "unit_scale": scale, "original_unit": "000_USD"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(raw, regularize=regularize)
        raise FileNotFoundError(f"Eora26 table for year {year} not found.")

    sep = "\t" if str(target).endswith(".txt") else ","
    df = pd.read_csv(target, sep=sep, header=None, low_memory=False)

    N = 189
    S = 26
    M = N * S
    n_fd_cols = N * len(EORA_FD_CATEGORIES)

    Z = df.iloc[:M, :M].values.astype(np.float64) * scale
    F = df.iloc[:M, M : M + n_fd_cols].values.astype(np.float64) * scale

    va_raw = df.iloc[M : M + 6, :M].values.astype(np.float64) * scale
    VA = np.sum(va_raw, axis=0)
    TLS = np.zeros(M, dtype=np.float64)
    Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=list(EORA_189_COUNTRIES),
        sectors=list(EORA_26_SECTORS),
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(EORA_FD_CATEGORIES),
        year=year,
        source="Eora26_Harmonized",
        unit="M_USD",
    )
    raw.metadata.update({
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(raw, regularize=regularize)


def load_oecd_icio_granular(
    year: int = 2019,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load granular OECD ICIO 45-sector transaction dataset.

    Reads the OECD 2023 regular native labeled CSV or ZIP: 77 economies,
    45 activities, TLS/VA/OUT margins and six final uses. Final demand is
    mapped to C=(HFCE, NPISH, GGFC), I=(GFCF, INVNT), Cx=DPABR before
    calibration, so the foreign-balance closure applies to investment.
    Provider output margins and pre-repair accounting discrepancies are
    retained in metadata. Native 2025 and extended archives are unsupported.

    Parameters
    ----------
    year : int, default=2019
        Reference year (1995 to 2020).
    fallback_to_synthetic : bool, default=False
        Explicitly opt into generated teaching data if the archive is missing.
    data_dir : str or Path, optional
        Directory containing OECD ICIO CSV files.
    file_path : str or Path, optional
        Exact path to an OECD 2023 regular CSV table or ZIP archive.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw = generate_synthetic_mrio("oecd", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_USD")
        return package_mrio_to_calibration_result(raw, regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates: list[Path] = []
        if data_dir:
            candidates.append(Path(data_dir))
        env_dir = os.environ.get("IO_OECD_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        for d in candidates:
            if d.is_dir():
                for name in (f"{year}_SML.csv", f"data_{year}_SML.csv", f"{year}.SML.csv"):
                    candidate_file = d / name
                    if candidate_file.is_file():
                        target = candidate_file
                        break
            if target is not None:
                break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw = generate_synthetic_mrio("oecd", year=year, seed=seed, unit="M_USD")
            return package_mrio_to_calibration_result(raw, regularize=regularize)
        raise FileNotFoundError(f"OECD ICIO granular CSV for year {year} not found.")

    from ._oecd_icio import read_native, condense_final_demand
    raw = condense_final_demand(read_native(target, year))
    return package_mrio_to_calibration_result(raw, regularize=regularize)


# ---------------------------------------------------------------------------
# Accounting & Balance Invariant Helpers
# ---------------------------------------------------------------------------


def compute_trade_balances(
    Z: np.ndarray,
    F: np.ndarray,
    nc: int,
    ns: int,
    nfd: int,
) -> np.ndarray:
    """Compute multilateral net foreign surplus vector XN across economies."""
    M = nc * ns
    F_2d = F.reshape(M, nc * nfd) if F.ndim == 3 else F
    T_inter = np.sum(Z.reshape(nc, ns, nc, ns), axis=(1, 3))
    T_fd = np.sum(F_2d.reshape(nc, ns, nc, nfd), axis=(1, 3))

    np.fill_diagonal(T_inter, 0.0)
    np.fill_diagonal(T_fd, 0.0)

    X0 = np.sum(T_inter, axis=1)
    M0 = np.sum(T_inter, axis=0)
    XFD = np.sum(T_fd, axis=1)
    MFD = np.sum(T_fd, axis=0)

    return (X0 + XFD - M0 - MFD)


def verify_zero_leakage(XN: np.ndarray, tol: float = 1e-10) -> bool:
    """Verify that multilateral trade balances sum to identically zero."""
    leakage = float(np.abs(np.sum(XN)))
    scale = float(np.sum(np.abs(XN)))
    if scale > 0:
        return (leakage / scale) <= tol or leakage <= 1e-8
    return leakage <= tol


def verify_accounting_invariants(calib: TradeCalibrationResult) -> dict[str, Any]:
    """Verify core accounting and balance invariants on a calibrated trade model.

    Checks:
    1. Gross output sales match column outlays: sum_i Z_ij + VA_j + TLS_j == Y_j
    2. Consumer budget balance: sum_f theta_c^f == 1.0 for each country c
    3. Multilateral zero-leakage trade balance: sum_c XN_c == 0.0
    4. Non-negative factor endowments: KT > 0 and LT > 0
    """
    data = calib.data_calibra
    nc = calib.nc
    ns = calib.ns
    n_ind = nc * ns

    # Outlays vs Sales
    col_outlays = np.sum(data[:, :n_ind], axis=0)
    sales = np.sum(data[:n_ind, :], axis=1)
    outlays_sales_err = float(np.max(np.abs(col_outlays - sales)))

    # Zero leakage: sum_c invforT_c == 0
    xn_sum = float(np.sum(calib.invforT))
    total_xn = float(np.sum(np.abs(calib.invforT)))
    zero_leakage = bool((abs(xn_sum) <= 1e-8) or (total_xn > 0 and (abs(xn_sum) / total_xn) <= 1e-10))

    # Budget balance: theta sums to 1.0 for each country
    theta_country_sums = np.sum(calib.theta, axis=1).flatten()
    budget_err = float(np.max(np.abs(theta_country_sums - 1.0)))

    # Factor non-negativity
    factors_positive = bool(np.all(calib.KT > 0) and np.all(calib.LT > 0))

    valid = bool(
        outlays_sales_err <= 1e-9 * float(np.max(sales))
        and zero_leakage
        and budget_err <= 1e-9
        and factors_positive
    )

    return {
        "valid": valid,
        "max_outlays_sales_error": outlays_sales_err,
        "trade_leakage_sum": xn_sum,
        "zero_leakage_certified": zero_leakage,
        "max_budget_error": budget_err,
        "factors_positive": factors_positive,
    }


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
    "FIGARO_COUNTRIES",
    "FIGARO_64_SECTORS",
    "FIGARO_FD_CATEGORIES",
    "EXIOBASE_COUNTRIES",
    "EXIOBASE_163_SECTORS",
    "EXIOBASE_200_SECTORS",
    "EXIOBASE_FD_CATEGORIES",
    "WIOD_44_COUNTRIES",
    "WIOD_56_SECTORS",
    "WIOD_FD_CATEGORIES",
    "EORA_189_COUNTRIES",
    "EORA_26_SECTORS",
    "EORA_FD_CATEGORIES",
    "OECD_77_COUNTRIES",
    "OECD_45_SECTORS",
    "OECD_FD_CATEGORIES",
    "get_country_codes",
    "get_eu_country_codes",
    "get_sector_codes",
    "get_sector_names",
    "get_raw_45_sector_codes",
    "get_raw_45_sector_names",
    "get_final_demand_codes",
    "get_raw_final_demand_codes",
    "ICIOData",
    "RawIOData",
    "load_icio_data",
    "load_raw_45sector_icio",
    "generate_synthetic_mrio",
    "package_mrio_to_calibration_result",
    "load_figaro",
    "load_exiobase",
    "load_wiod",
    "load_eora",
    "load_oecd_icio_granular",
    "compute_trade_balances",
    "verify_zero_leakage",
    "verify_accounting_invariants",
]
