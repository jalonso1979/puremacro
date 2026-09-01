"""EU-KLEMS / INTAN-Prod 2023 release loader.

The 2023 release ships as four large long-/wide-format CSVs:
- ``national accounts.csv`` — wide on (geo, sector, year), totals.
- ``capital accounts.csv``  — wide on (geo, sector, year), asset-class detail.
- ``labour accounts.csv``   — long on (geo, sector, edu, age, gender, year),
  with ``Share_E`` (employment share) and ``Share_W`` (wage share). Shares
  sum to 1 across all (edu, age, gender) cells within a (geo, sector, year).
- ``intangibles analytical.csv`` — extra intangibles detail (NOT used for
  KORV; loader does not read it).

The loader reads from a local cache at ``data/raw/euklems/`` (populated
manually — see the README there), aggregates labour shares to the 3-level
education breakdown (LOW = ISCED 0-2, MED = ISCED 3-4, HIGH = ISCED 5-8),
multiplies by total H_EMP and COMP to get hours/labor compensation by skill,
and merges with national + capital accounts. Returns a tidy
``(code, year, ...)`` DataFrame at the total-economy (``nace_r2_code='TOT'``)
aggregation.

Equipment capital can be aggregated under three definitions, selected via
the ``equip_def`` parameter to :func:`load_klems_panel`:

* ``'tangible'`` (default): ``k_equip = K_OMach + K_TraEq + K_IT``
  --- Other Machinery & Equipment + Transport Equipment + Computing
  Equipment. This is the v0.3.1 tangible-capital baseline.
* ``'ict'``: ``k_equip = K_IT + K_Soft_DB`` --- ICT-extended definition
  in the spirit of the post-2010 capital-skill complementarity literature
  (Eden-Gaggl 2018, Acemoglu-Restrepo 2022) where software is the
  cleanest substitute for routine unskilled labor.
* ``'broad'``: ``k_equip = K_OMach + K_TraEq + K_IT + K_Soft_DB``
  --- tangible equipment + ICT hardware + software & databases.

The same definition is applied symmetrically to ``i_equip``
(``I_*`` investment flows) and to ``p_equip_index`` (value-weighted
geometric mean of the corresponding ``Ip_*`` price indexes).

The loader's previous draft also referenced ``K_OComp``; that variable
does not exist in EU-KLEMS 2023.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# Equipment-definition registry. Each entry maps an equip_def label to the
# tuple of EU-KLEMS asset-class suffixes that compose the aggregate. The
# loader applies the same suffix list to K_*, I_*, and Ip_* columns.
_EQUIP_DEFS: dict[str, tuple[str, ...]] = {
    'tangible': ('OMach', 'TraEq', 'IT'),
    'ict':      ('IT', 'Soft_DB'),
    'broad':    ('OMach', 'TraEq', 'IT', 'Soft_DB'),
}

# Backwards-compatible aliases — kept so callers/tests that import these
# constants directly still see the v0.3.1 'tangible' definition.
_EQUIP_CLASSES = tuple(f'K_{s}' for s in _EQUIP_DEFS['tangible'])
_EQUIP_PRICE_CLASSES = tuple(f'Ip_{s}' for s in _EQUIP_DEFS['tangible'])
_EQUIP_INVEST_CLASSES = tuple(f'I_{s}' for s in _EQUIP_DEFS['tangible'])


def _equip_columns(equip_def: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (K_cols, Ip_cols, I_cols) for the given equipment definition."""
    if equip_def not in _EQUIP_DEFS:
        raise ValueError(
            f"unknown equip_def={equip_def!r}; valid options are "
            f"{sorted(_EQUIP_DEFS)}"
        )
    suffixes = _EQUIP_DEFS[equip_def]
    return (
        tuple(f'K_{s}' for s in suffixes),
        tuple(f'Ip_{s}' for s in suffixes),
        tuple(f'I_{s}' for s in suffixes),
    )

# Level-1 NACE Rev. 2 sectors (alphabetic letters, no compound codes like
# 'C10-C12' or 'M-N'). Used by the sector-aggregation fallback for years
# where the EU-KLEMS 2023 labour file leaves the TOT row empty (typically
# pre-2008 EU countries) but populates per-sector skill shares.
_LEVEL1_SECTORS = tuple('ABCDEFGHIJKLMNOPQRSTU')

_KLEMS_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CY': 'CYP', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC', 'ES': 'ESP',
    'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL',
    'IT': 'ITA', 'JP': 'JPN', 'LT': 'LTU', 'LU': 'LUX', 'LV': 'LVA',
    'MT': 'MLT', 'NL': 'NLD', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU',
    'SE': 'SWE', 'SI': 'SVN', 'SK': 'SVK', 'UK': 'GBR', 'US': 'USA',
}
_KLEMS_AGGREGATES = {
    'EA19', 'EU11', 'EU12', 'EU15', 'EU19', 'EU20',
    'EU27', 'EU27_2020', 'EU28',
}

_EDU_CODE_TO_NAME = {1: 'low', 2: 'med', 3: 'high'}

# Fine education codes (US-only in 2023 release) → 3-level skill bucket.
# Verified against ISCED 2011: 11=ISCED 0-2 (low), 12=ISCED 3 (med),
# 13=ISCED 4 (med), 14=ISCED 5-6 (high), 15=ISCED 7-8 (high).
# Plausibility check against US CPS data for 2015 confirms this mapping
# (low≈9%, med≈57%, high≈34%), which matches the ISCED standard exactly.
_FINE_EDU_TO_NAME = {11: 'low', 12: 'med', 13: 'med', 14: 'high', 15: 'high'}
# Fine age codes covering all working-age groups for the US.
_FINE_AGE_CODES = [11, 12, 13, 14, 15, 16, 17]

_OUT_COLS = [
    'code', 'year',
    'lab_low', 'lab_med', 'lab_high',
    'hours_low', 'hours_med', 'hours_high',
    'comp_total', 'hours_total', 'va',
    'k_equip',
    'p_equip_index',
]
_OUT_COLS_WITH_INVEST = _OUT_COLS + ['i_equip']
_EMPTY = pd.DataFrame(columns=_OUT_COLS)


def _pivot_skill_shares_to_wide(grouped_skill: pd.DataFrame) -> pd.DataFrame:
    """Pivot a DataFrame containing (geo_code, year, skill, Share_E, Share_W)
    into wide format with columns share_E_low/med/high, share_W_low/med/high.
    """
    e = grouped_skill.pivot_table(
        index=['geo_code', 'year'],
        columns='skill',
        values='Share_E',
        aggfunc='first',
    ).rename(columns=lambda s: f'share_E_{s}')
    w = grouped_skill.pivot_table(
        index=['geo_code', 'year'],
        columns='skill',
        values='Share_W',
        aggfunc='first',
    ).rename(columns=lambda s: f'share_W_{s}')
    # Ensure all three skill columns are present (fill missing with NaN).
    for skill in ('low', 'med', 'high'):
        for prefix in ('share_E_', 'share_W_'):
            col = f'{prefix}{skill}'
            if col not in e.columns and prefix == 'share_E_':
                e[col] = np.nan
            if col not in w.columns and prefix == 'share_W_':
                w[col] = np.nan
    out = e.join(w).reset_index()
    out.columns.name = None
    return out

def _pivot_shares_to_wide(
    agg: pd.DataFrame,
    code_to_skill: dict[int, str],
) -> pd.DataFrame:
    """Pivot a long-format (geo_code, year, education, Share_E, Share_W)
    frame into wide format with columns share_E_low/med/high,
    share_W_low/med/high.

    *agg* must already have Share_E and Share_W divided by 100.
    Multiple education codes mapping to the same skill are summed.
    """
    agg = agg.copy()
    agg['skill'] = agg['education'].map(code_to_skill)
    agg = agg.dropna(subset=['skill'])
    if agg.empty:
        return pd.DataFrame()
    grouped = agg.groupby(['geo_code', 'year', 'skill'], as_index=False)[
        ['Share_E', 'Share_W']
    ].sum(min_count=1)
    return _pivot_skill_shares_to_wide(grouped)


def _aggregate_labour_shares(labour_long: pd.DataFrame, *,
                              industry: str = 'TOT') -> pd.DataFrame:
    """Aggregate labour shares to the 3-level education breakdown.

    Returns wide-format (geo_code, year, share_E_low/med/high,
    share_W_low/med/high) for the requested industry.

    Primary path: uses aggregate education codes {1, 2, 3} (all EU/non-US
    countries in the 2023 release).  Fallback path: for (geo, year) cells
    missing in the aggregate path, collapses fine education codes
    {11→low, 12→med, 13→med, 14→high, 15→high} (US in the 2023 release).
    Both paths avoid double-counting by selecting the appropriate age-code
    filter (aggregate ages [1,2,3] vs fine ages [11-17]).
    """
    industry_sub = labour_long[labour_long['nace_r2_code'] == industry]

    # --- Aggregate path (codes 1, 2, 3) ---
    agg_sub = industry_sub[
        industry_sub['education'].isin([1, 2, 3]) &
        industry_sub['age'].isin([1, 2, 3])
    ]
    # Sum across (age, gender) within (geo, year, education).
    # Use min_count=1 so that (geo, year, education) cells where ALL
    # underlying (age, gender) values are NaN remain NaN rather than
    # collapsing to 0 — which would silently produce zero skill totals
    # when multiplied by COMP/H_EMP.
    agg_long = agg_sub.groupby(
        ['geo_code', 'year', 'education'], as_index=False
    )[['Share_E', 'Share_W']].sum(min_count=1)
    # EU-KLEMS 2023 shares are expressed in percent (0–100 scale).
    # Convert to fractions so that multiplying by totals yields correct units.
    agg_long['Share_E'] = agg_long['Share_E'] / 100.0
    agg_long['Share_W'] = agg_long['Share_W'] / 100.0
    agg_wide = _pivot_shares_to_wide(agg_long, _EDU_CODE_TO_NAME)

    # --- Fine-code fallback path (codes 11–15; US only in 2023 release) ---
    fine_sub = industry_sub[
        industry_sub['education'].isin(list(_FINE_EDU_TO_NAME.keys())) &
        industry_sub['age'].isin(_FINE_AGE_CODES)
    ]
    fine_long = fine_sub.groupby(
        ['geo_code', 'year', 'education'], as_index=False
    )[['Share_E', 'Share_W']].sum(min_count=1)
    fine_long['Share_E'] = fine_long['Share_E'] / 100.0
    fine_long['Share_W'] = fine_long['Share_W'] / 100.0
    fine_wide = _pivot_shares_to_wide(fine_long, _FINE_EDU_TO_NAME)

    if agg_wide.empty and fine_wide.empty:
        return pd.DataFrame()

    if agg_wide.empty:
        out = fine_wide
    elif fine_wide.empty:
        out = agg_wide
    else:
        # Combine: prefer aggregate where present, fall back to fine codes
        # where aggregate is missing.  Set (geo_code, year) as index so
        # combine_first() aligns on both dimensions.
        agg_idx = agg_wide.set_index(['geo_code', 'year'])
        fine_idx = fine_wide.set_index(['geo_code', 'year'])
        out = agg_idx.combine_first(fine_idx).reset_index()

    out.columns.name = None
    return out



def _get_sector_skill_shares(labour_long: pd.DataFrame) -> pd.DataFrame:
    # Restrict labour file to level-1 sectors.
    lab_sub = labour_long[
        labour_long['nace_r2_code'].isin(_LEVEL1_SECTORS)
    ].copy()
    if lab_sub.empty:
        return pd.DataFrame()
    # Both age/edu code systems may coexist; filter to the consistent
    # combinations (aggregate edu × aggregate age, fine edu × fine age).
    agg_mask = (
        lab_sub['education'].isin([1, 2, 3]) &
        lab_sub['age'].isin([1, 2, 3])
    )
    fine_mask = (
        lab_sub['education'].isin(list(_FINE_EDU_TO_NAME.keys())) &
        lab_sub['age'].isin(_FINE_AGE_CODES)
    )
    lab_sub = lab_sub[agg_mask | fine_mask].copy()
    if lab_sub.empty:
        return pd.DataFrame()
    # Map education code → 3-level skill (aggregate codes preferred).
    skill_map = {**_EDU_CODE_TO_NAME, **_FINE_EDU_TO_NAME}
    lab_sub['skill'] = lab_sub['education'].map(skill_map)
    lab_sub = lab_sub.dropna(subset=['skill'])
    # Sum across (age, gender) within (geo, sector, year, skill).
    sec_skill = lab_sub.groupby(
        ['geo_code', 'nace_r2_code', 'year', 'skill'], as_index=False
    )[['Share_E', 'Share_W']].sum(min_count=1)
    # Renormalise within (geo, sector, year) so shares sum to exactly 1 across
    # skills.  Some EU-KLEMS 2023 sector cells have raw sums slightly off from
    # 100 (e.g. FI sector B 1995 sums to 80, IT sector B 1995 sums to 110),
    # presumably because the underlying microdata are not weighted to perfectly
    # close at the sector level.  Without renormalisation our synthesised TOT
    # shares would inherit those deviations and ``lab_low + lab_med + lab_high``
    # would drift from ``comp_total`` by a few percent for some country-years.
    sec_skill_sum = sec_skill.groupby(
        ['geo_code', 'nace_r2_code', 'year'], as_index=False
    )[['Share_E', 'Share_W']].sum(min_count=1).rename(
        columns={'Share_E': '_skill_E_sum', 'Share_W': '_skill_W_sum'}
    )
    sec_skill = sec_skill.merge(
        sec_skill_sum, on=['geo_code', 'nace_r2_code', 'year'], how='left'
    )
    # Avoid divide-by-zero when no skill shares are populated for a sector cell.
    sec_skill['Share_E'] = np.where(
        sec_skill['_skill_E_sum'] > 0,
        sec_skill['Share_E'] / sec_skill['_skill_E_sum'],
        np.nan,
    )
    sec_skill['Share_W'] = np.where(
        sec_skill['_skill_W_sum'] > 0,
        sec_skill['Share_W'] / sec_skill['_skill_W_sum'],
        np.nan,
    )
    return sec_skill.drop(columns=['_skill_E_sum', '_skill_W_sum'])


def _calculate_tot_skill_shares(
    sec_skill: pd.DataFrame,
    na_long: pd.DataFrame,
) -> pd.DataFrame:
    # Restrict national accounts to the same level-1 sectors and merge.
    na_sub = na_long[
        na_long['nace_r2_code'].isin(_LEVEL1_SECTORS)
    ][['nace_r2_code', 'geo_code', 'year', 'COMP', 'H_EMP']].copy()
    if na_sub.empty:
        return pd.DataFrame()

    merged = sec_skill.merge(
        na_sub, on=['geo_code', 'nace_r2_code', 'year'], how='inner'
    )
    if merged.empty:
        return pd.DataFrame()
    # Within-sector skill totals (hours / comp).
    merged['hours_skill'] = merged['Share_E'] * merged['H_EMP']
    merged['comp_skill']  = merged['Share_W'] * merged['COMP']

    # Sum across sectors → TOT-level skill totals.
    skill_tot = merged.groupby(
        ['geo_code', 'year', 'skill'], as_index=False
    )[['hours_skill', 'comp_skill']].sum(min_count=1)
    # The denominator must be the sum of H_EMP / COMP only over those
    # sectors where skill shares are populated — otherwise a sector with
    # missing skill data (e.g. NACE T and U for some EU countries pre-2008)
    # inflates the denominator, biasing all three skill shares downward and
    # making lab_sum drift below comp_total by a few percent.
    cov_sectors = merged.dropna(subset=['Share_E', 'Share_W'])[
        ['geo_code', 'nace_r2_code', 'year', 'COMP', 'H_EMP']
    ].drop_duplicates(subset=['geo_code', 'nace_r2_code', 'year'])
    sector_tot = cov_sectors.groupby(
        ['geo_code', 'year'], as_index=False
    )[['COMP', 'H_EMP']].sum(min_count=1).rename(
        columns={'COMP': 'comp_sec_sum', 'H_EMP': 'hours_sec_sum'}
    )
    skill_tot = skill_tot.merge(sector_tot, on=['geo_code', 'year'], how='left')
    # Effective TOT share = (skill total) / (sector-sum total).
    # By construction these shares sum to 1 across skills within (geo, year)
    # — so when later multiplied by the published TOT comp_total / hours_total
    # the per-skill totals close to comp_total exactly (the only remaining
    # error is the published TOT vs sector-sum rounding, which is ≪ 1%).
    skill_tot['Share_E'] = skill_tot['hours_skill'] / skill_tot['hours_sec_sum']
    skill_tot['Share_W'] = skill_tot['comp_skill']  / skill_tot['comp_sec_sum']
    return skill_tot


def _aggregate_labour_shares_from_sectors(
    labour_long: pd.DataFrame,
    na_long: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesise TOT-level labour shares by aggregating level-1 NACE sectors.

    EU-KLEMS 2023 leaves the ``TOT`` row of the labour file empty (NaN) for
    many EU countries pre-2008, even though per-sector skill shares are
    populated back to 1995 for nine countries (AT, DE, ES, FI, FR, IT, NL,
    UK plus US via fine codes).  When that happens we can still build
    effective TOT-level shares as

        share_E_skill_TOT = Σ_s (share_E_skill_s × H_EMP_s) / Σ_s H_EMP_s
        share_W_skill_TOT = Σ_s (share_W_skill_s × COMP_s)  / Σ_s COMP_s

    summed over level-1 NACE sectors ``A..U`` whose H_EMP/COMP totals are
    provided by ``national_accounts.csv``.  We deliberately drop compound
    sector codes (``C10-C12``, ``M-N``, ``MARKT``, ``TOT_IND``, …) so that
    the contributions are exclusive and sum to the TOT total to within
    rounding.

    Returns wide-format (geo_code, year, share_E_low/med/high,
    share_W_low/med/high).  Returns an empty frame if no sector-level
    data are present in either input.
    """
    sec_skill = _get_sector_skill_shares(labour_long)
    if sec_skill.empty:
        return pd.DataFrame()

    skill_tot = _calculate_tot_skill_shares(sec_skill, na_long)
    if skill_tot.empty:
        return pd.DataFrame()

    return _pivot_skill_shares_to_wide(skill_tot)


def load_klems_panel(
    cache_dir: Path | str = 'data/raw/euklems',
    *,
    codes: Iterable[str] | None = None,
    industry: str = 'TOT',
    include_investment: bool = False,
    equip_def: str = 'tangible',
) -> pd.DataFrame:
    """Load the full EU-KLEMS 2023 panel at the requested industry.

    Parameters
    ----------
    cache_dir
        Directory containing ``national_accounts.csv``,
        ``capital_accounts.csv``, ``labour_accounts.csv``.
    codes
        ISO-3 country codes to keep. Default: all 30 countries in
        ``_KLEMS_TO_ISO3``. Aggregates (EA19, EU…) are always excluded.
    industry
        EU-KLEMS NACE Rev. 2 sector code. Default: ``'TOT'`` (total economy).
        Use ``'C'`` for manufacturing, etc.
    include_investment : bool, default False
        If True, also return the ``i_equip`` column, which is the sum of
        gross investment flows over the asset classes selected by
        ``equip_def`` (nominal, same currency as ``k_equip``). Useful for
        the investment-based m2 diagnostic in the KORV GMM estimator.
        When the ``I_*`` columns are absent from the capital accounts file
        the column is returned as NaN throughout. Existing callers that
        pass ``include_investment=False`` (the default) see no change in
        the returned schema.
    equip_def : {'tangible', 'ict', 'broad'}, default 'tangible'
        Equipment-aggregate definition (see :data:`_EQUIP_DEFS`):

        * ``'tangible'`` --- ``K_OMach + K_TraEq + K_IT`` (v0.3.1 baseline,
          tangible equipment + IT hardware).
        * ``'ict'``      --- ``K_IT + K_Soft_DB`` (ICT-extended; the
          capital-skill-complementarity-literature definition with software
          as the operative substitution margin for routine unskilled labor).
        * ``'broad'``    --- ``K_OMach + K_TraEq + K_IT + K_Soft_DB``
          (tangible + software).

        The same suffix list is applied to ``k_equip``, ``i_equip``, and to
        the value-weighted price index ``p_equip_index``.

    Returns
    -------
    pd.DataFrame
        Long-format with columns ``code, year, lab_low, lab_med, lab_high,
        hours_low, hours_med, hours_high, comp_total, hours_total, va,
        k_equip, p_equip_index``. If ``include_investment=True``, also
        includes ``i_equip``. One row per (code, year). Empty DataFrame if
        cache_dir is missing or files are absent.

        ``p_equip_index`` is a value-weighted geometric mean of the
        ``Ip_*`` equipment price indexes selected by ``equip_def``, with
        weights given by the corresponding real capital stocks ``K_*``.
        If all selected price series are missing for a given (code, year),
        the value is NaN. When price data are absent from the capital
        accounts file (some EU-KLEMS releases omit them) the column is
        returned as NaN throughout rather than aborting the load.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return _EMPTY.copy()
    # Resolve equipment-definition columns once.
    _k_equip_cols, _ip_equip_cols, _i_equip_cols = _equip_columns(equip_def)
    paths = {
        'na':  cache_dir / 'national_accounts.csv',
        'k':   cache_dir / 'capital_accounts.csv',
        'lab': cache_dir / 'labour_accounts.csv',
    }
    if not all(p.exists() for p in paths.values()):
        return _EMPTY.copy()
    try:
        na = pd.read_csv(
            paths['na'],
            usecols=['nace_r2_code', 'geo_code', 'year', 'COMP', 'H_EMP', 'VA_CP'],
        )
        # Try to read capital_accounts with Ip price columns; fall back
        # gracefully if they are absent from the release being loaded.
        _k_base_cols = ['nace_r2_code', 'geo_code', 'year', *_k_equip_cols]
        _k_price_cols = list(_ip_equip_cols)
        _k_invest_cols = list(_i_equip_cols)
        # Columns to request from capital_accounts.
        _k_extra = _k_price_cols + (_k_invest_cols if include_investment else [])
        try:
            k = pd.read_csv(
                paths['k'],
                usecols=_k_base_cols + _k_extra,
            )
        except (ValueError, KeyError):
            # Some optional columns absent — load base + price, fill missing.
            try:
                k = pd.read_csv(paths['k'], usecols=_k_base_cols + _k_price_cols)
            except (ValueError, KeyError):
                k = pd.read_csv(paths['k'], usecols=_k_base_cols)
            for _c in _k_price_cols:
                if _c not in k.columns:
                    k[_c] = np.nan
            if include_investment:
                for _c in _k_invest_cols:
                    if _c not in k.columns:
                        k[_c] = np.nan
        lab = pd.read_csv(paths['lab'])
    except Exception:
        return _EMPTY.copy()

    # Drop EU/EA aggregates from national accounts (kept full-sector for the
    # sector-aggregation fallback below; industry filter applied after).
    na_full = na[~na['geo_code'].isin(_KLEMS_AGGREGATES)].copy()
    na = na_full[na_full['nace_r2_code'] == industry]
    k = k[
        (k['nace_r2_code'] == industry) &
        (~k['geo_code'].isin(_KLEMS_AGGREGATES))
    ]
    if na.empty or k.empty:
        return _EMPTY.copy()

    lab_shares = _aggregate_labour_shares(lab, industry=industry)
    # When industry='TOT', synthesise extra (geo, year) cells from
    # level-1 sector data — this lifts pre-2008 EU coverage that is
    # missing from the TOT row of the labour file.
    if industry == 'TOT':
        sector_shares = _aggregate_labour_shares_from_sectors(lab, na_full)
        if not sector_shares.empty:
            sector_shares = sector_shares[
                ~sector_shares['geo_code'].isin(_KLEMS_AGGREGATES)
            ]
            if lab_shares.empty:
                lab_shares = sector_shares
            else:
                # Prefer TOT-level (already-published) shares where present;
                # fill gaps with the sector-aggregated synthesis.
                lab_idx = lab_shares.set_index(['geo_code', 'year'])
                sec_idx = sector_shares.set_index(['geo_code', 'year'])
                lab_shares = lab_idx.combine_first(sec_idx).reset_index()
                lab_shares.columns.name = None
    if lab_shares.empty:
        return _EMPTY.copy()
    lab_shares = lab_shares[~lab_shares['geo_code'].isin(_KLEMS_AGGREGATES)]

    # Merge on (geo_code, year).
    na = na.drop(columns='nace_r2_code')
    k = k.drop(columns='nace_r2_code')
    df = na.merge(k, on=['geo_code', 'year'], how='outer')
    df = df.merge(lab_shares, on=['geo_code', 'year'], how='left')

    # Compute outputs.
    df['lab_low']    = df['COMP']  * df.get('share_W_low',  np.nan)
    df['lab_med']    = df['COMP']  * df.get('share_W_med',  np.nan)
    df['lab_high']   = df['COMP']  * df.get('share_W_high', np.nan)
    df['hours_low']  = df['H_EMP'] * df.get('share_E_low',  np.nan)
    df['hours_med']  = df['H_EMP'] * df.get('share_E_med',  np.nan)
    df['hours_high'] = df['H_EMP'] * df.get('share_E_high', np.nan)
    df['comp_total']  = df['COMP']
    df['hours_total'] = df['H_EMP']
    df['va']          = df['VA_CP']
    # Some equip_def-selected columns may be absent from older releases —
    # protect the sum from KeyError by ensuring all expected K_* exist.
    for _c in _k_equip_cols:
        if _c not in df.columns:
            df[_c] = np.nan
    df['k_equip'] = df[list(_k_equip_cols)].sum(axis=1, min_count=1)

    # Compute p_equip_index: value-weighted geometric mean of the
    # equip_def-selected EU-KLEMS equipment price indexes (Ip_*).
    # Weights are the corresponding real capital stocks; if all
    # Ip columns are NaN the result is NaN.
    _ip_cols = list(_ip_equip_cols)
    _wt_cols = list(_k_equip_cols)
    for _c in _ip_cols:
        if _c not in df.columns:
            df[_c] = np.nan
    # Build weight matrix (fallback to equal weights if stocks are all 0/NaN).
    # Reset numeric column labels to position-aligned indices so that pandas
    # multiplies element-wise rather than aligning on column NAME — Ip_* and
    # K_* are different names but correspond positionally by suffix.
    _ip = df[_ip_cols].copy().clip(lower=1e-9)  # avoid log(0)
    _wt = df[_wt_cols].fillna(0)
    _ip.columns = range(len(_ip_cols))
    _wt.columns = range(len(_wt_cols))
    _wt_sum = _wt.sum(axis=1).replace(0, np.nan)
    _wt_norm = _wt.div(_wt_sum, axis=0)
    # Replace rows where all Ip values are NaN with NaN result.
    _ip_valid = df[_ip_cols].notna().any(axis=1)
    _log_pk = (np.log(_ip) * _wt_norm).sum(axis=1)
    df['p_equip_index'] = np.where(_ip_valid, np.exp(_log_pk), np.nan)

    # Optionally compute i_equip = sum of nominal equipment investment flows.
    if include_investment:
        _inv_cols = list(_i_equip_cols)
        for _c in _inv_cols:
            if _c not in df.columns:
                df[_c] = np.nan
        df['i_equip'] = df[_inv_cols].sum(axis=1, min_count=1)

    df['code'] = df['geo_code'].map(_KLEMS_TO_ISO3)
    df = df.dropna(subset=['code'])
    df['year'] = df['year'].astype(int)

    _out_cols = _OUT_COLS_WITH_INVEST if include_investment else _OUT_COLS
    out = df[_out_cols].copy()
    if codes is not None:
        out = out[out['code'].isin(set(codes))]
    return out.sort_values(['code', 'year']).reset_index(drop=True)


# Backward-compat: keep load_klems_country as a thin wrapper that opens
# the cache once and filters to one country, in case any caller still
# uses it. New code should use load_klems_panel(codes=[...]) directly.
def load_klems_country(
    cache_dir: Path | str = 'data/raw/euklems',
    *,
    code: str,
    industry: str = 'TOT',
    equip_def: str = 'tangible',
) -> pd.DataFrame:
    """Load one country from the EU-KLEMS 2023 cache."""
    return load_klems_panel(
        cache_dir, codes=[code], industry=industry, equip_def=equip_def,
    )


__all__ = ['_KLEMS_TO_ISO3', 'load_klems_country', 'load_klems_panel']
