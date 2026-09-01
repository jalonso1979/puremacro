"""4-state F/I/U/N monthly transition matrices from ENOE microdata (Mexico).

Reads the .dta ENOE quarter files distributed by INEGI (2005q2-present),
links persons across consecutive interviews via the 5-quarter rotating
panel, computes observed 3-month transitions per calendar reference month,
and recovers 1-month transitions via the principal logarithm of the
3-month matrix (Israel-Rosenthal-Wei regularization where needed).

The four labor-force states are:
    F = formally employed (employee with health-system access via job)
    I = informally employed (no employer-provided health-system access)
    U = unemployed (desocupado, actively searching)
    N = not in labor force (inactivo)

**Side-channel note** (same convention as ``narrative.sources`` documenting
its network connectors): the *loading* half of this module
(``load_enoe_quarter`` / ``transitions_from_enoe``) reads raw INEGI ``.dta``
quarter files from a local mirror via ``pandas.read_stata``. That is an
inherently out-of-browser workflow — the microdata are hundreds of MB per
quarter and are not redistributable inside a Pyodide bundle. It adds **no**
dependencies beyond numpy/pandas/scipy, but it expects a local filesystem
with the INEGI quarter directories. The *analytical* half
(``make_person_id``, ``assign_labor_status``, ``link_consecutive_quarters``,
``quarterly_transitions_from_pairs``, ``quarterly_to_monthly_matrix``,
``monthly_stocks_from_quarter``, ``validate_stocks_vs_published``) is pure
DataFrame/ndarray -> DataFrame/ndarray and runs anywhere, browser included.

Public API
----------
STATES                              : ("F", "I", "U", "N")
TransitionPanelENOE                 : dataclass (monthly, quarterly, stocks, quarterly_observed)
load_enoe_quarter(root, y, q)       : load+merge HOG+SDEM+COE1 for one quarter
make_person_id(df)                  : 13-char ENOE linking key (ENT+CON+V_SEL+N_HOG+H_MUD+N_REN)
assign_labor_status(df)             : map CLASE2 + formality to F/I/U/N
link_consecutive_quarters(q1, q2)   : inner-join two quarters on person_id
quarterly_transitions_from_pairs    : weighted 3-month transitions per ref month
quarterly_to_monthly_matrix(P_Q)    : 1-month via logm / 3 + expm + IRW reg
monthly_stocks_from_quarter(df)     : F/I/U/N stocks by interview month
transitions_from_enoe(root, qs)     : end-to-end orchestrator
validate_stocks_vs_published(...)   : per-state max-pct-diff tolerance vs INEGI tabulation

Ported from the shocks-and-transitions project module
``shocks_paper/labor_flows_enoe.py``. The source module's loaders for the
pre-ENOE surveys (ENE 2000-2004, ENEU 1987-2004) and the ETOE 2020
telephonic survey are NOT ported: they read dBase ``.dbf`` files via the
``dbfread`` package, which is outside this package's runtime dependency
set (numpy/pandas/scipy only).

References
----------
Elsby, M., Hobijn, B., & Sahin, A. (2015). On the importance of the
participation margin for labor market fluctuations. JME 72.
Israel, R. B., Rosenthal, J. S., & Wei, J. Z. (2001). Finding generators
for Markov chains via empirical transition matrices. Math. Finance 11(2).
Shimer, R. (2012). Reassessing the ins and outs of unemployment. RED 15.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg as _spla


STATES: tuple[str, str, str, str] = ("F", "I", "U", "N")


@dataclass
class TransitionPanelENOE:
    """Monthly 4-by-4 F/I/U/N transition probabilities and stocks for Mexico."""
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    stocks: pd.DataFrame
    quarterly_observed: pd.DataFrame

    @property
    def states(self) -> tuple[str, str, str, str]:
        return STATES

    def matrix_at(self, date) -> np.ndarray:
        row = self.monthly.loc[date]
        return np.array([[row[f"p_{a}{b}"] for b in STATES] for a in STATES])


# ENOE labor-status mapping
# -------------------------
# CLASE2 codes (ENOE technical note):
#   0 = ineligible (children under 12)
#   1 = ocupado (employed)
#   2 = desocupado (unemployed, actively searching)
#   3 = disponible (not searching but available for work; PEI subset)
#   4 = no disponible (not in labor force, not available; PEI subset)
#
# PEI (poblacion economicamente inactiva) = CLASE2 in {3, 4}; both map to N.
# PEA (poblacion economicamente activa) = CLASE2 in {1, 2}.
#
# Formality follows the ILO/INEGI definition used by ENOE in its formal
# vs informal series (TIA-1): an employee is formal iff she has access
# to a social-security health institution through her employer. The
# binary access flag differs across survey formats:
#   - ENOE Format A     (most Q1 + 2005 + a few): P3J (1=yes, 2=no)
#   - ENOE Format B     (most Q2/Q3/Q4): P3K1 (1=yes, 2=no)
#   - ENE-era files:    A_SEG_SOC (Acceso a Seguridad Social, 1=yes, 0=no)
# In Format A, P3K1 is an INSTITUTION code (not a binary access flag),
# but its ==1 subset is contained in P3J==1 so the OR still gives the
# right answer. P3J1 in Format B is a NARROWER follow-up question and
# is NOT a substitute.
# Formal iff (any of A_SEG_SOC, P3J, P3K1) == 1.

FORMALITY_COLS = ("A_SEG_SOC", "P3J", "P3K1")


def _eq_one(s: pd.Series) -> pd.Series:
    """Robust equality-to-1 that works on either numeric or string columns.

    ENOE .dta stores codes as floats; other INEGI distributions store
    codes as strings.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return s.astype(str).str.strip() == "1"
    return s == 1


def _isin_codes(s: pd.Series, codes: list) -> pd.Series:
    """Robust isin that works on either numeric or string columns."""
    if not pd.api.types.is_numeric_dtype(s):
        target = {str(c) for c in codes}
        return s.astype(str).str.strip().isin(target)
    return s.isin(codes)


def assign_labor_status(df: pd.DataFrame) -> pd.Series:
    """Map ENOE rows to one of F, I, U, N."""
    if "CLASE2" not in df.columns:
        raise ValueError("ENOE row data must contain CLASE2 (labor-force status).")

    clase2 = df["CLASE2"]
    status = pd.Series(index=df.index, dtype=object)
    employed_mask = _eq_one(clase2)
    status.loc[_isin_codes(clase2, [2])] = "U"
    status.loc[_isin_codes(clase2, [3, 4])] = "N"

    if not employed_mask.any():
        return status

    formality_present = [c for c in FORMALITY_COLS if c in df.columns]
    if not formality_present:
        status.loc[employed_mask] = "I"
        return status

    formal_mask = pd.Series(False, index=df.index)
    for col in formality_present:
        formal_mask = formal_mask | _eq_one(df[col])
    status.loc[employed_mask & formal_mask] = "F"
    status.loc[employed_mask & ~formal_mask] = "I"
    return status


# Person-ID linking columns (ENOE 2005-present, .dta lowercase upcased).
# ENT (state, 2d) + CON (control number within state, 5d) + V_SEL (sample
# selection, 2d) + N_HOG (household within dwelling, 1d) + H_MUD (housing-
# move flag, 1d) + N_REN (roster line, 2d) = 13-char key unique within a
# quarter and stable across the 5-quarter panel rotation.
#
# Key history of failures with simpler keys:
#   - (CD_A, V_SEL, N_HOG, H_MUD, N_REN) — original CSV-era key.
#     390,712 rows but only 6,907 distinct tuples in 2018q1 (~57:1).
#   - (UPM, V_SEL, N_HOG, H_MUD, N_REN) — works for 2018+, but in 2005q2
#     UPM/V_SEL overlap across panels (121,501 HOG rows, only 34,575
#     unique on (UPM, V_SEL, N_HOG, H_MUD)).
# The (ENT, CON, V_SEL, N_HOG, H_MUD) household key is verified 1:1
# unique in BOTH 2005q2 (121,501) and 2018q1 (130k-ish).
LINKING_COLS: tuple[tuple[str, int], ...] = (
    ("ENT", 2),
    ("CON", 5),
    ("V_SEL", 2),
    ("N_HOG", 1),
    ("H_MUD", 1),
    ("N_REN", 2),
)


def _id_part(s: pd.Series, width: int) -> pd.Series:
    """One zero-padded person-ID component.

    Fully-numeric columns (ENOE float codes) take the integer path --
    byte-identical to the historical key. Columns that carry genuine
    alphanumeric codes (letter-prefixed dwelling selections in some
    non-ENOE distributions) fall back to the stripped string so the code
    is preserved instead of raising ``ValueError: invalid literal for int()``.
    """
    num = pd.to_numeric(s, errors="coerce")
    has_alpha = bool((num.isna() & s.notna()).any())
    if not has_alpha:                       # ENOE -- unchanged
        return s.astype("Int64").astype(str).str.zfill(width)
    as_int = num.astype("Int64").astype(str)
    return as_int.where(num.notna(), s.astype(str).str.strip()).str.zfill(width)


def make_person_id(df: pd.DataFrame) -> pd.Series:
    """Construct ENOE 13-character person identifier."""
    missing = [c for c, _ in LINKING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ENOE row data missing linking column(s): {missing}")
    pieces = [_id_part(df[c], width) for c, width in LINKING_COLS]
    out = pieces[0]
    for p in pieces[1:]:
        out = out + p
    return out


# Household-level join cols (HOG -> SDEM/COE1). N_REN excluded (person-level).
HOG_JOIN_COLS = ["ENT", "CON", "V_SEL", "N_HOG", "H_MUD"]


def _find_dta(directory: Path, table: str) -> Path | None:
    """Find the .dta file for a given table, handling prefix and case variants.

    ENOE distributes files with three prefix variants:
      - bare (2005-2019): SDEMT115.dta  (case can be lower in some years)
      - ENOE_  (2020q1, 2024q1+): ENOE_SDEMT120.dta
      - ENOEN_ (2020q3-2023):     ENOEN_SDEMT320.dta
    Some quarters (2012, 2013, 2019) use lowercase filenames.
    """
    upper = sorted(directory.glob(f"*{table}T*.dta"))
    lower = sorted(directory.glob(f"*{table.lower()}t*.dta"))
    matches = upper + [p for p in lower if p not in upper]
    return matches[0] if matches else None


# Default search roots for the INEGI quarter directories. Empty by design:
# a distributed package must not hardcode a personal mirror. Either pass
# ``root=`` explicitly to every loader call, or set this once per session:
#     import puremacro.labor_flows_enoe as lfe
#     lfe.DEFAULT_ROOTS = (Path("/path/to/my/ENOE"),)
DEFAULT_ROOTS: tuple[Path, ...] = ()


# ENEU 1987-coverage 16-city set. These are the metropolitan areas surveyed
# by ENEU from its inception in 1987 and consistently covered by all
# subsequent surveys (incl. ENOE 2005-present). Codes match both
# ENEU/ENE/ETOE A_MET and ENOE CD_A conventions (INEGI's standard
# city/metro ordering). Used by the optional ``urban_filter`` argument to
# restrict ENOE to a historically-consistent urban universe.
ENEU_1987_CITIES: dict[int, str] = {
    1:  "Mexico City (ZMVM)",
    2:  "Guadalajara",
    3:  "Monterrey",
    4:  "Puebla-Tlaxcala",
    5:  "Leon",
    6:  "Torreon-Lerdo-Gomez Palacio",
    7:  "San Luis Potosi-Soledad",
    8:  "Merida",
    9:  "Chihuahua",
    10: "Tampico-Madero-Altamira",
    11: "Veracruz-Boca del Rio",
    12: "Orizaba",
    13: "Ciudad Juarez",
    14: "Aguascalientes",
    15: "Tijuana",
    16: "Saltillo",
}

# ENEU 1992-coverage 34-city set (1987 + 18 additions). Use if you want
# a slightly later but broader urban universe.
ENEU_1992_ADDITIONS: dict[int, str] = {
    17: "Morelia", 18: "Toluca", 19: "Hermosillo", 20: "Tepic",
    21: "Acapulco", 22: "Aguascalientes-est. ext", 23: "Cuernavaca",
    24: "Oaxaca", 25: "Zacatecas", 26: "Colima", 27: "Villahermosa",
    28: "Tuxtla Gutierrez", 29: "Tlaxcala", 30: "La Paz",
    31: "Cancun", 32: "Pachuca", 33: "Tepic-2nd", 34: "Campeche",
}


def _coerce_urban_filter(urban_filter) -> set | None:
    """Resolve an urban_filter argument to a set of integer city codes.

    Accepts:
      None             -> no filter (national sample)
      "eneu_1987"      -> 16 cities
      "eneu_1992"      -> 34 cities (1987 + 1992 additions)
      iterable of int  -> use as-is
    """
    if urban_filter is None:
        return None
    if isinstance(urban_filter, str):
        if urban_filter == "eneu_1987":
            return set(ENEU_1987_CITIES.keys())
        if urban_filter == "eneu_1992":
            return set(ENEU_1987_CITIES.keys()) | set(ENEU_1992_ADDITIONS.keys())
        raise ValueError(f"unknown urban_filter '{urban_filter}'.")
    return set(int(c) for c in urban_filter)


def _apply_urban_filter(df: pd.DataFrame, cities: set | None) -> pd.DataFrame:
    """Filter df to rows in the given city set.

    Uses CD_A if present (ENOE), else A_MET (ENEU/ENE). Both encode the
    same city numbering convention.
    """
    if cities is None:
        return df
    city_col = "CD_A" if "CD_A" in df.columns else (
        "A_MET" if "A_MET" in df.columns else None)
    if city_col is None:
        return df  # No city column; can't filter — return unfiltered
    codes = pd.to_numeric(df[city_col], errors="coerce")
    return df.loc[codes.isin(cities)].copy()


def _coerce_roots(roots: Path | str | tuple | list | None) -> tuple[Path, ...]:
    """Allow either a single root or a sequence; return tuple of Paths."""
    if roots is None:
        if not DEFAULT_ROOTS:
            raise ValueError(
                "No ENOE root directory given. Pass root= (the directory that "
                "contains the INEGI quarter folders, e.g. 2018trim1_dta/), or "
                "set puremacro.labor_flows_enoe.DEFAULT_ROOTS to your local "
                "mirror(s) once per session."
            )
        return DEFAULT_ROOTS
    if isinstance(roots, (str, Path)):
        return (Path(roots),)
    return tuple(Path(r) for r in roots)


def _quarter_dir(roots: tuple[Path, ...], year: int, quarter: int) -> Path:
    """Locate the quarter directory under any of the candidate roots.

    Accepts both directory naming conventions:
        <root>/<YYYY>trim<Q>_dta/         (most quarters, 2005-2025q2)
        <root>/enoe_<YYYY>_trim<Q>_dta/   (2025q3, 2025q4 onward)
    Returns the first existing directory; raises FileNotFoundError if none.
    """
    patterns = [f"{year}trim{quarter}_dta", f"enoe_{year}_trim{quarter}_dta"]
    for r in roots:
        for p in patterns:
            d = r / p
            if d.exists():
                return d
    raise FileNotFoundError(
        f"ENOE {year}q{quarter} dir missing under any root.\n"
        f"  roots tried: {[str(r) for r in roots]}\n"
        f"  patterns:    {patterns}"
    )


# Columns we actually need from each table. Loading all 270+ columns wastes
# memory and slows merges by orders of magnitude. Names are uppercased
# AFTER load (since .dta files use lowercase).

_HOG_COLS_WANTED = {
    # join keys (incl. ENE renames hog/per)
    "upm", "v_sel", "n_hog", "h_mud", "n_ent", "ent", "con", "tipo",
    "cve_ent",  # 2025q3+ alias for ent
    "hog",      # ENE/ENEU alias for n_hog
    "cd_a", "a_met",  # city / metro area code (for urban_filter)
    # interview date
    "d_anio", "d_mes", "d_dia", "p_anio", "p_mes", "mes_cal",
    # weight (used as fallback; fac_np = ETOE non-panel weight)
    "fac", "fac_tri", "fac_men", "fac_np",
}

_SDEM_COLS_WANTED = {
    "upm", "v_sel", "n_hog", "h_mud", "n_ren", "n_ent", "ent", "con", "tipo",
    # 2025q3+ INEGI renamed some keys: cve_ent -> ENT (handled in loader).
    "cve_ent",
    # ENE/ENEU 2000-2004 renames: HOG -> N_HOG, PER -> N_REN
    "hog", "per",
    "cd_a", "a_met",  # city / metro area code (for urban_filter)
    "eda", "sex", "clase1", "clase2", "n_ent",
    "fac", "fac_tri", "fac_men", "fac_np",
    # Education
    "cs_p13_1", "cs_p13_2", "cs_p17", "niv_ins", "anios_esc",
    # Wages & hours
    "ing7c", "ingocup", "ing_x_hrs", "salario", "hrsocup", "remune2c",
    # Sector / occupation
    "scian", "rama", "rama_est1", "rama_est2", "c_ocu11c", "emp_ppal",
    "mes_cal",
}

_COE1_COLS_WANTED = {
    "upm", "v_sel", "n_hog", "h_mud", "n_ren", "n_ent", "ent", "con", "tipo",
    "cve_ent",
    "p3", "p3a", "p3b",
    # Formality flags across survey formats (ENE/ENOE-A/ENOE-B/ETOE).
    "p3j", "p3j1", "p3j2", "p3k1", "p3k2", "p3l1", "p3l2",
    "a_seg_soc", "o_seg_soc", "imssissste",  # ENE-era formality
}

# COE2 (search module) — opt-in via load_enoe_quarter(with_coe2=True). Carries
# the retrospective separation-reason battery (supply-vs-demand mechanism
# work): p9c job-loss, p9d separation, p9e business-closure, p8b OTJ search
# motive. Not loaded by default (existing callers unaffected/unslowed).
_COE2_COLS_WANTED = {
    "upm", "v_sel", "n_hog", "h_mud", "n_ren", "n_ent", "ent", "con", "tipo",
    "cve_ent",
    "p8b",                              # on-the-job search motive (precautionary)
    "p9", "p9c", "p9d", "p9e",          # job-loss / separation / business-closure reason
    "p9f", "p9f_anio", "p9f_mes",       # timing of the separation
}


def _read_stata_subset(path: Path, wanted: set[str]) -> pd.DataFrame:
    """Read a .dta file and keep only the wanted columns (case-insensitive).

    Single-pass: we read the full file then drop unwanted columns. Reading
    .dta is I/O-bound and pandas does not support skipping unread columns
    at the binary level, so double-passing with iterator=True just doubles
    runtime. The in-memory cost of holding all columns briefly is fine.
    """
    df = pd.read_stata(path, convert_categoricals=False)
    keep = [c for c in df.columns if c.lower() in wanted]
    df = df.loc[:, keep]
    df.columns = df.columns.str.upper()
    return df


def _normalize_anio(v) -> int | float:
    """Normalize ANIO field to 4-digit year. Accepts int, float, or str.

    ENOE pre-2025 stores D_ANIO as 2-digit (18.0 = 2018). Some
    distributions store it as a 4-digit string ('2020').
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return v
    if n < 50:
        return 2000 + n
    if n < 100:
        return 1900 + n
    return n


def load_enoe_quarter(
    root: Path | str | tuple | list | None,
    year: int,
    quarter: int,
    urban_filter=None,
    *,
    with_coe2: bool = False,
) -> pd.DataFrame:
    """Load and merge HOG + SDEM + COE1 tables for one ENOE quarter.

    root: either a single Path/str, a list of candidate roots, or None to
    use the module-level DEFAULT_ROOTS (which must then be non-empty).

    HOG carries the interview date (d_anio/d_mes/d_dia) at the household
    level; we propagate it to persons via the household-level join. SDEM
    carries demographics and CLASE2; COE1 carries formality (P3J, P3K1).

    Returns a DataFrame with one row per person-interview. Only a curated
    subset of columns is kept (see module constants _HOG_COLS_WANTED etc.).
    Derived columns added:
        person_id     : 13-char ENOE linking key
        labor_status  : "F"/"I"/"U"/"N"
        D_ANIO/D_MES/D_DIA : interview date (from HOG)
        FAC           : sampling weight (uses FAC_TRI when split, else FAC)

    Raises FileNotFoundError if the quarter directory or required tables
    are missing (e.g., 2005q1 has no SDEM; 2020q2 dir does not exist).

    Out-of-browser side-channel: reads local .dta microdata mirrors — see
    the module docstring.
    """
    roots = _coerce_roots(root)
    qdir = _quarter_dir(roots, year, quarter)

    hog_path = _find_dta(qdir, "HOG")
    sdem_path = _find_dta(qdir, "SDEM")
    coe1_path = _find_dta(qdir, "COE1")
    missing = [
        name for name, p in (("HOG", hog_path), ("SDEM", sdem_path), ("COE1", coe1_path))
        if p is None
    ]
    if missing:
        raise FileNotFoundError(
            f"ENOE {year}q{quarter} missing table(s): {missing} in {qdir}"
        )

    hog = _read_stata_subset(hog_path, _HOG_COLS_WANTED)
    sdem = _read_stata_subset(sdem_path, _SDEM_COLS_WANTED)
    coe1 = _read_stata_subset(coe1_path, _COE1_COLS_WANTED)

    # Column-name harmonization across ENOE eras:
    #   2025q3+ ENOE:        CVE_ENT -> ENT  (new INEGI 'clave' convention)
    #   ENE-era files:       HOG     -> N_HOG;  PER -> N_REN
    # Apply uniformly so downstream code doesn't have to special-case.
    _RENAMES = {"CVE_ENT": "ENT", "HOG": "N_HOG", "PER": "N_REN"}
    for df in (hog, sdem, coe1):
        cols_now = set(df.columns)
        rename_map = {old: new for old, new in _RENAMES.items()
                      if old in cols_now and new not in cols_now}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    # FAC reconciliation. ENOE pre-2020q3 has a single FAC column that
    # behaves as a per-row weight summing to ~quarterly population, with
    # each row attributable to one D_MES via HOG. Sum-by-D_MES then gives
    # the monthly cohort. Post-2020q3 ENOE-N stores FAC_MEN (per-row =
    # FAC_TRI * 3) and FAC_TRI separately. ETOE (2020 Apr/May/Jun) uses
    # FAC_NP (non-panel). Priority order: FAC > FAC_TRI > FAC_NP.
    for df in (hog, sdem, coe1):
        if "FAC" in df.columns:
            continue
        for alt in ("FAC_TRI", "FAC_NP", "FAC_MEN"):
            if alt in df.columns:
                df["FAC"] = df[alt]
                break

    # Dedup within-quarter. Post-2020q3 ENOE-N allows the same person to
    # appear in multiple N_ENT (interview rounds) within one quarter
    # (e.g., 2020q3 SDEM has 35k duplicate person keys due to COVID-era
    # interview rescheduling). The SDEM-COE1 merge would cartesian-product
    # these. Keep one row per (person, N_ENT), and for stocks/transitions
    # we further collapse to one row per person below.
    person_full_key = [
        c for c in ("ENT", "CON", "V_SEL", "N_HOG", "H_MUD", "N_REN", "N_ENT")
        if c in sdem.columns
    ]
    person_full_key_coe1 = [
        c for c in person_full_key if c in coe1.columns
    ]
    sdem = sdem.drop_duplicates(person_full_key, keep="first")
    coe1 = coe1.drop_duplicates(person_full_key_coe1, keep="first")

    # HOG -> SDEM (household-level): bring interview date columns onto each person.
    hog_join = [c for c in HOG_JOIN_COLS if c in hog.columns and c in sdem.columns]
    hog_carry = [c for c in ("D_ANIO", "D_MES", "D_DIA", "MES_CAL") if c in hog.columns]
    sdem = sdem.merge(
        hog[hog_join + hog_carry].drop_duplicates(hog_join),
        on=hog_join, how="left",
    )

    # D_ANIO is stored as 2-digit year (e.g., 18 for 2018). Expand to 4-digit.
    if "D_ANIO" in sdem.columns:
        sdem["D_ANIO"] = sdem["D_ANIO"].apply(_normalize_anio)

    # 2020q3 COVID-period interview reshuffling left D_MES with spurious
    # earlier months. Prefer MES_CAL when available (2020q3+) for the
    # reference month; fall back to D_MES otherwise.
    if "MES_CAL" in sdem.columns:
        sdem["D_MES"] = sdem["MES_CAL"].where(
            sdem["MES_CAL"].notna(), sdem.get("D_MES")
        )
        # D_ANIO is the quarter's nominal year; that still matches MES_CAL.

    # SDEM -> COE1 (person-level): include N_REN in the join key.
    person_join = hog_join + ["N_REN"]
    person_join = [c for c in person_join if c in sdem.columns and c in coe1.columns]
    # Avoid duplicating columns already on SDEM.
    coe1_extra = [c for c in coe1.columns if c not in sdem.columns and c not in person_join]
    merged = sdem.merge(
        coe1[person_join + coe1_extra], on=person_join, how="left",
    )

    merged["person_id"] = make_person_id(merged)
    merged["labor_status"] = assign_labor_status(merged)

    # Final dedup to 1 row per person_id (collapsing across N_ENT rounds
    # within the quarter). Keep the row with the largest FAC to preserve
    # the most-weighted interview when multiple exist.
    if merged["person_id"].duplicated().any():
        merged = (
            merged.sort_values("FAC", ascending=False, na_position="last")
                  .drop_duplicates("person_id", keep="first")
                  .reset_index(drop=True)
        )

    # Opt-in COE2 (search module) merge — the retrospective separation-reason
    # battery. Mirrors the HOG/SDEM/COE1 path: read subset, harmonize era
    # column names, build the person key, attach onto the deduped panel.
    # Reason cols are returned LOWER-cased (downstream crosswalks key on p9c/d/e).
    if with_coe2:
        coe2_path = _find_dta(qdir, "COE2")
        if coe2_path is not None:
            coe2 = _read_stata_subset(coe2_path, _COE2_COLS_WANTED)
            rn = {old: new for old, new in _RENAMES.items()
                  if old in coe2.columns and new not in coe2.columns}
            if rn:
                coe2.rename(columns=rn, inplace=True)
            coe2["person_id"] = make_person_id(coe2)
            reason_upper = ["P8B", "P9", "P9C", "P9D", "P9E",
                            "P9F", "P9F_ANIO", "P9F_MES"]
            present = [c for c in reason_upper if c in coe2.columns]
            sub = (coe2[["person_id"] + present]
                   .drop_duplicates("person_id")
                   .rename(columns={c: c.lower() for c in present}))
            merged = merged.merge(sub, on="person_id", how="left")

    return _apply_urban_filter(merged, _coerce_urban_filter(urban_filter))


def link_consecutive_quarters(
    q_origin: pd.DataFrame,
    q_dest: pd.DataFrame,
    verify_identity: bool = True,
    age_increment: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Inner-join two consecutive ENOE quarters on person_id, with optional
    identity verification.

    person_id = ENT+CON+V_SEL+N_HOG+H_MUD+N_REN includes the roster line
    N_REN, which households reuse when a member leaves. With
    ``verify_identity=True`` (default) a linked pair is kept only if SEX
    agrees across waves and the age change EDA_d - EDA_o is in
    ``age_increment`` (default {0,1}: age rises by 0 or 1 over a 3-month
    gap). ENOE top-code/missing ages (97/98/99/NaN) and missing SEX are
    treated as unverifiable and dropped. Diagnostic counts are attached to
    ``.attrs`` (link_n_before, link_n_after).

    Returns one row per verified person observed in BOTH quarters with
    columns person_id, origin, dest, ref_year, ref_month, weight, EDA, SEX
    (EDA/SEX from the origin wave).
    """
    cols = ["person_id", "labor_status", "D_ANIO", "D_MES", "FAC", "EDA", "SEX"]
    o = q_origin[cols].rename(columns={
        "labor_status": "origin", "FAC": "FAC_o",
        "D_ANIO": "ref_year", "D_MES": "ref_month",
    })
    d = q_dest[["person_id", "labor_status", "FAC", "EDA", "SEX"]].rename(columns={
        "labor_status": "dest", "FAC": "FAC_d",
        "EDA": "EDA_d", "SEX": "SEX_d",
    })
    pairs = o.merge(d, on="person_id", how="inner")
    n_before = len(pairs)

    if verify_identity and n_before:
        eda_o = pd.to_numeric(pairs["EDA"], errors="coerce")
        eda_d = pd.to_numeric(pairs["EDA_d"], errors="coerce")
        sex_o = pd.to_numeric(pairs["SEX"], errors="coerce")
        sex_d = pd.to_numeric(pairs["SEX_d"], errors="coerce")
        unverifiable = (
            eda_o.isin([97, 98, 99]) | eda_d.isin([97, 98, 99])
            | eda_o.isna() | eda_d.isna()
            | sex_o.isna() | sex_d.isna()
        )
        sex_ok = sex_o == sex_d
        age_ok = (eda_d - eda_o).isin(age_increment)
        pairs = pairs[sex_ok & age_ok & ~unverifiable].copy()

    n_after = len(pairs)
    pairs["weight"] = (pairs["FAC_o"] + pairs["FAC_d"]) / 2.0
    pairs = pairs.drop(columns=["FAC_o", "FAC_d", "EDA_d", "SEX_d"]).reset_index(drop=True)
    pairs.attrs["link_n_before"] = int(n_before)
    pairs.attrs["link_n_after"] = int(n_after)
    return pairs


# COVID-2020 guards: minimum respondents per interview month and minimum
# matched pairs per link month before a cell is treated as data.
MIN_MONTH_PERSONS = 1000
MIN_LINK_PAIRS = 2000


def quarterly_transitions_from_pairs(pairs: pd.DataFrame, min_pairs: int = 0) -> pd.DataFrame:
    """Weighted 3-month transition matrix per calendar reference month.

    An origin state with no matched pairs in a month is missing data and yields
    NaN, never 0.0 (writing zeros fabricated the all-zero 2020-05 row during the
    COVID suspension). Months with fewer than `min_pairs` matched pairs in total
    are masked to NaN entirely: the 2020 suspension/restart window leaves sliver
    links whose rates are dominated by re-contact selection.
    """
    pairs = pairs.dropna(subset=["origin", "dest", "ref_year", "ref_month"]).copy()
    pairs["ref_date"] = pd.to_datetime(
        pairs["ref_year"].astype(int).astype(str) + "-"
        + pairs["ref_month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    n_pairs = pairs.groupby("ref_date").size()
    g = pairs.groupby(["ref_date", "origin", "dest"])["weight"].sum().reset_index()

    rows = []
    for ref, sub in g.groupby("ref_date"):
        row = {"ref_date": ref}
        thin = int(n_pairs.get(ref, 0)) < int(min_pairs)
        for a in STATES:
            denom = sub.loc[sub["origin"] == a, "weight"].sum()
            for b in STATES:
                num = sub.loc[(sub["origin"] == a) & (sub["dest"] == b), "weight"].sum()
                row[f"p_{a}{b}"] = (num / denom) if (denom > 0 and not thin) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("ref_date").sort_index()


def quarterly_to_monthly_matrix(P_Q: np.ndarray) -> np.ndarray:
    """Recover the 1-month transition matrix from the 3-month matrix.

    Embedding: Q = logm(P_Q) / 3, then P_M = expm(Q). Apply IRW
    regularization (clip negative off-diagonals to zero, re-set diagonals
    so generator row sums to zero) when needed (Israel-Rosenthal-Wei 2001).
    """
    n = P_Q.shape[0]
    if P_Q.shape != (n, n):
        raise ValueError(f"P_Q must be square; got shape {P_Q.shape}.")

    P_Q_work = P_Q.copy()
    zero_row_mask = np.isclose(P_Q_work.sum(axis=1), 0.0)
    for i in np.where(zero_row_mask)[0]:
        P_Q_work[i, i] = 1.0

    log_P = _spla.logm(P_Q_work)
    if np.iscomplexobj(log_P):
        log_P = log_P.real
    Q = log_P / 3.0

    Q_reg = Q.copy()
    off = ~np.eye(n, dtype=bool)
    neg_off = (Q_reg < 0) & off
    if neg_off.any():
        Q_reg[neg_off] = 0.0
        for i in range(n):
            row_sum_excluding_diag = Q_reg[i, :].sum() - Q_reg[i, i]
            Q_reg[i, i] = -row_sum_excluding_diag

    P_M = _spla.expm(Q_reg)
    P_M = np.clip(P_M, 0.0, None)
    row_sums = P_M.sum(axis=1, keepdims=True)
    P_M = P_M / row_sums

    if zero_row_mask.any():
        P_M[zero_row_mask, :] = 0.0

    return P_M


def monthly_stocks_from_quarter(df: pd.DataFrame, min_persons: int = 0) -> pd.DataFrame:
    """Sum FAC by (interview month, labor state) using D_ANIO/D_MES from HOG.

    Months with fewer than `min_persons` respondents are masked to NaN: the
    COVID-2020 fieldwork suspension leaves slivers (2020-03 carries 381
    respondents; the phased 2020-07 restart ~2% of the sample) that otherwise
    enter the monthly series as catastrophic outliers.
    """
    df = df.dropna(subset=["labor_status", "D_ANIO", "D_MES"]).copy()
    df["ref_date"] = pd.to_datetime(
        df["D_ANIO"].astype(int).astype(str) + "-"
        + df["D_MES"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    g = df.groupby(["ref_date", "labor_status"])["FAC"].sum().unstack(fill_value=0.0)
    for s in STATES:
        if s not in g.columns:
            g[s] = 0.0
    g = g[list(STATES)].sort_index()
    if min_persons:
        n = df.groupby("ref_date").size().reindex(g.index).fillna(0)
        g.loc[n < int(min_persons), :] = float("nan")
    return g


def transitions_from_enoe(
    root: Path | str | tuple | list | None,
    quarters: Sequence[tuple[int, int]],
    skip_missing: bool = True,
    verbose: bool = True,
) -> TransitionPanelENOE:
    """Build a TransitionPanelENOE from a sequence of ENOE quarters.

    If skip_missing=True (default), quarters whose directory is absent
    (e.g., 2020q2 ETOE-only, or 2005q1 lacking SDEM) are logged and
    skipped instead of raising.

    Out-of-browser side-channel: reads local .dta microdata mirrors — see
    the module docstring.
    """
    import time
    quarter_dfs: dict[tuple[int, int], pd.DataFrame] = {}
    t_start = time.time()
    for i, (y, q) in enumerate(quarters):
        t0 = time.time()
        try:
            quarter_dfs[(y, q)] = load_enoe_quarter(root, y, q)
        except FileNotFoundError as e:
            if skip_missing:
                if verbose:
                    print(f"[skip] {y}q{q}: {e}", flush=True)
                continue
            raise
        if verbose:
            df = quarter_dfs[(y, q)]
            print(f"  [{i+1:>2d}/{len(quarters)}] {y}q{q}: "
                  f"{len(df):>7,} rows, {df.shape[1]} cols, "
                  f"loaded in {time.time()-t0:.1f}s "
                  f"(total {time.time()-t_start:.1f}s)", flush=True)

    stock_frames = [monthly_stocks_from_quarter(df, min_persons=MIN_MONTH_PERSONS)
                    for df in quarter_dfs.values()]
    if stock_frames:
        stocks = pd.concat(stock_frames).groupby(level=0).sum().sort_index()
    else:
        stocks = pd.DataFrame(columns=list(STATES))

    pair_frames = []
    sorted_q = sorted(quarter_dfs.keys())
    for (y1, q1), (y2, q2) in zip(sorted_q, sorted_q[1:]):
        months_apart = (y2 - y1) * 12 + 3 * (q2 - q1)
        if months_apart != 3:
            continue
        pair_frames.append(
            link_consecutive_quarters(quarter_dfs[(y1, q1)], quarter_dfs[(y2, q2)])
        )
    if pair_frames:
        all_pairs = pd.concat(pair_frames, ignore_index=True)
        quarterly_observed = quarterly_transitions_from_pairs(all_pairs, min_pairs=MIN_LINK_PAIRS)
    else:
        quarterly_observed = pd.DataFrame(
            columns=[f"p_{a}{b}" for a in STATES for b in STATES]
        )

    monthly_rows = []
    for row in quarterly_observed.itertuples():
        ref_date = row.Index
        P_Q = np.array([[getattr(row, f"p_{a}{b}") for b in STATES] for a in STATES])
        try:
            P_M = quarterly_to_monthly_matrix(P_Q)
        except Exception as e:
            print(f"[warn] logm failed for {ref_date}: {e}")
            continue
        monthly_rows.append({
            "date": ref_date,
            **{f"p_{a}{b}": P_M[i, j]
               for i, a in enumerate(STATES)
               for j, b in enumerate(STATES)}
        })
    if monthly_rows:
        monthly = pd.DataFrame(monthly_rows).set_index("date").sort_index()
    else:
        monthly = pd.DataFrame(
            columns=[f"p_{a}{b}" for a in STATES for b in STATES]
        )

    quarterly_chain_rows = []
    for row in monthly.itertuples():
        d = row.Index
        M = np.array([[getattr(row, f"p_{a}{b}") for b in STATES] for a in STATES])
        P3 = M @ M @ M
        qd = d + pd.offsets.QuarterEnd(0)
        quarterly_chain_rows.append({
            "qdate": qd,
            **{f"p_{a}{b}": P3[i, j]
               for i, a in enumerate(STATES)
               for j, b in enumerate(STATES)}
        })
    quarterly = (
        pd.DataFrame(quarterly_chain_rows).set_index("qdate")
        if quarterly_chain_rows else pd.DataFrame()
    )

    return TransitionPanelENOE(
        monthly=monthly,
        quarterly=quarterly,
        stocks=stocks,
        quarterly_observed=quarterly_observed,
    )


def validate_stocks_vs_published(
    ours: pd.DataFrame,
    published: pd.DataFrame,
    tol_pct: float = 1.0,
) -> dict:
    """Compare quarterly F/I/U/N stocks vs INEGI's published tabulation."""
    aligned = ours.reindex(published.index)
    diff_pct = ((aligned - published).abs() / published.abs()) * 100
    per_state = diff_pct.max(axis=0)
    max_diff = float(per_state.max())
    return {
        "max_abs_pct_diff": max_diff,
        "per_state": per_state,
        "passed": bool(max_diff < tol_pct),
    }
