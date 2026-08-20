"""Transforms on a :func:`~puremacro.fetch.qna_panel` frame.

:func:`~puremacro.fetch.qna_panel` returns the three products of a quarterly
national accounts build — current-price levels, implicit deflators and volume
measures — tied together, component by component, by

.. math:: \\text{nominal} = \\text{real} \\times \\text{deflator} / 100.

Everything here operates on that frame and keeps that relationship exact:

* :func:`qna_rebase` moves every country onto a **common price reference
  year**, which the source does not do (each country references its own).
* :func:`qna_identity` scores the expenditure identity in both nominal and
  volume terms, which is how the **non-additivity of chain-linked volumes**
  stops being folklore and becomes a number.
* :func:`qna_contributions` decomposes real GDP growth into the contribution
  of each component, using previous-period prices as chain-linking requires.

None of these touch the network: hand them a live panel or a frozen CSV of one
and they behave identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Expenditure components and the sign with which they enter
#: :math:`Y = C_{hh} + C_{gov} + I + X - M`. ``capform`` (gross capital
#: formation) rather than ``inv`` (GFCF) is the term the identity actually
#: uses: it is GFCF *plus the change in inventories*, and dropping inventories
#: turns an exact identity into a wrong one.
IDENTITY_TERMS: dict[str, int] = {
    "cons_hh": +1, "cons_gov": +1, "capform": +1, "exports": +1, "imports": -1,
}


def _codes(panel: pd.DataFrame) -> list[str]:
    return list(panel.index.get_level_values("code").unique())


def _require_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError(
            "expected a (code, date)-indexed qna_panel frame; got an index "
            f"with {getattr(panel.index, 'nlevels', 1)} level(s). Pass the "
            "wide frame, not the long=True form.")


def _base_names(panel: pd.DataFrame) -> list[str]:
    """Component names present as current-price columns."""
    return [c for c in panel.columns
            if not c.endswith("_defl") and not c.endswith("_real")]


def qna_rebase(panel: pd.DataFrame, year: int | str,
               *, strict: bool = False) -> pd.DataFrame:
    """Re-reference every country's deflators to 100 in ``year``.

    The OECD references each country's volumes to that country's own base
    year — 2017 for the United States, 2018 for Mexico, 2020 for Spain — so
    the deflator columns of a raw panel are **not comparable levels across
    countries**, and a chart of ``inv_defl`` is a chart of three different
    conventions. This puts them all on one reference year.

    The volume columns are rescaled by the same factor, so real levels come
    out in ``year`` prices and

    .. math:: \\text{nominal} = \\text{real} \\times \\text{deflator} / 100

    still holds exactly, component by component.

    This is a **re-referencing**, not a re-basing: it multiplies each series
    by one scalar per country. It changes the units the level is quoted in
    and nothing else — every growth rate, every ratio of two quarters and
    every chain link is untouched. Re-*basing* a chain-linked volume, which
    would mean recomputing the links from a different year's price weights,
    is not something a published panel lets you do at all.

    Parameters
    ----------
    panel
        A :func:`~puremacro.fetch.qna_panel` frame, ``(code, date)``-indexed.
    year
        Reference year, e.g. ``2015``. The scaling factor is the **annual
        average** of the component's own deflator over that year's quarters,
        which is what "100 in 2015" means in the accounts.
    strict
        Raise if a country has no observations in ``year``. Default ``False``
        leaves such a country's columns as ``NaN`` — an emerging-market series
        that starts in 2019 simply cannot be referenced to 2015, and dropping
        the whole country over it is usually worse than a visible hole.

    Returns
    -------
    pandas.DataFrame
        Same shape and column order as ``panel``. Current-price columns are
        returned unchanged — they are what they are — and ``panel.attrs`` is
        carried over with ``price_ref_year`` updated in the metadata records.

    Examples
    --------
    >>> panel = qna_panel(["USA", "ESP"], real=True)     # doctest: +SKIP
    >>> flat = qna_rebase(panel, 2015)                   # doctest: +SKIP
    >>> flat.groupby("code")["gdp_defl"].apply(          # doctest: +SKIP
    ...     lambda s: s[s.index.get_level_values("date").year == 2015].mean())
    code
    ESP    100.0
    USA    100.0
    """
    _require_panel(panel)
    year = int(year)
    out = panel.copy()
    dates = out.index.get_level_values("date")
    in_year = dates.year == year

    # A country is "missing" when it has no quarter in the reference year at
    # all — a series starting in 2019 cannot be referenced to 2015. A country
    # that merely lacks *one component* there keeps its other columns rebased
    # and gets NaN only in that component, which is the useful behaviour.
    covered = set(out.loc[in_year].index.get_level_values("code").unique())
    missing = sorted(c for c in _codes(panel) if c not in covered)

    code_of = pd.Series(out.index.get_level_values("code"), index=out.index)
    for name in _base_names(panel):
        defl_col, real_col = f"{name}_defl", f"{name}_real"
        if defl_col not in out.columns:
            continue
        # One factor per country per component: the component's own deflator
        # averaged over the reference year's quarters.
        factor = (out.loc[in_year, defl_col].groupby(level="code").mean() / 100.0)
        factor = factor.replace(0.0, np.nan)
        aligned = code_of.map(factor)
        out[defl_col] = out[defl_col] / aligned
        if real_col in out.columns:
            out[real_col] = out[real_col] * aligned
    if missing and strict:
        raise ValueError(
            f"no {year} observations to reference on for: {', '.join(missing)}. "
            "Pass strict=False to leave them NaN, or pick an earlier year.")

    meta = [dict(r) for r in panel.attrs.get("meta", ())]
    for rec in meta:
        if rec.get("code") not in missing:
            rec["price_ref_year"] = float(year)
    out.attrs = dict(panel.attrs)
    out.attrs["meta"] = tuple(meta)
    out.attrs["price_ref_year"] = year
    if missing:
        out.attrs["rebase_missing"] = tuple(missing)
    return out


def qna_identity(panel: pd.DataFrame, *, as_share: bool = True) -> pd.DataFrame:
    """Score :math:`Y = C_{hh} + C_{gov} + I + X - M` on both sides of the panel.

    Two very different numbers come out, and telling them apart is the whole
    point of holding current prices as the primitive:

    ``nominal_*``
        The published **statistical discrepancy**, plus whatever seasonal
        adjustment added to it. In current prices the identity is an
        accounting fact, so on *unadjusted* data this is only the residual the
        statistical office could not reconcile — often exactly zero, because
        many offices force it to close. On the adjusted data a panel actually
        holds there is a second term: each series is adjusted on its own, so
        the adjusted components need not sum to adjusted GDP. It shows up as a
        small residual alternating in sign quarter to quarter (Brazil and
        Romania are clear cases here) rather than as a level shift, which is
        how to tell the two apart.

    ``real_*``
        The **chain-linking gap**, present only when the panel was built with
        ``real=True``. Chain-linked volumes are not additive: each component
        is deflated with its own price weights, so the components do not add
        to GDP, and the gap widens the further a quarter sits from the price
        reference year. This is not a data error and it does not go away.
        It is why growth is decomposed with :func:`qna_contributions` rather
        than by adding up volumes.

    Parameters
    ----------
    panel
        A :func:`~puremacro.fetch.qna_panel` frame.
    as_share
        Report gaps as a percentage of GDP (default) rather than in millions
        of national currency, so countries are comparable.

    Returns
    -------
    pandas.DataFrame
        One row per country, indexed by ``code``:
        ``nominal_mean``, ``nominal_absmax``, ``real_mean``, ``real_absmax``,
        ``real_last``, ``n_obs``. Real columns are ``NaN`` when the panel
        carries no volume measures.

    Examples
    --------
    >>> qna_identity(qna_panel(["USA"], real=True))      # doctest: +SKIP
          nominal_mean  nominal_absmax  real_mean  real_absmax  real_last  n_obs
    code
    USA           0.00            0.00       0.01         0.09       0.03    122
    """
    _require_panel(panel)
    have_real = all(f"{n}_real" in panel.columns for n in IDENTITY_TERMS)
    have_real = have_real and "gdp_real" in panel.columns

    def _gap(suffix: str) -> pd.Series:
        gdp = panel[f"gdp{suffix}"]
        total = sum(sign * panel[f"{name}{suffix}"]
                    for name, sign in IDENTITY_TERMS.items())
        gap = gdp - total
        return 100.0 * gap / gdp if as_share else gap

    nom = _gap("")
    rows = pd.DataFrame({
        "nominal_mean": nom.groupby(level="code").mean(),
        "nominal_absmax": nom.abs().groupby(level="code").max(),
    })
    if have_real:
        rl = _gap("_real")
        rows["real_mean"] = rl.groupby(level="code").mean()
        rows["real_absmax"] = rl.abs().groupby(level="code").max()
        rows["real_last"] = rl.groupby(level="code").apply(
            lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    else:
        rows["real_mean"] = rows["real_absmax"] = rows["real_last"] = np.nan
    rows["n_obs"] = panel["gdp"].groupby(level="code").count()
    rows.attrs["units"] = "% of GDP" if as_share else panel.attrs.get("units", "")
    return rows


def qna_contributions(panel: pd.DataFrame, *, annualise: bool = False,
                      periods: int = 1) -> pd.DataFrame:
    """Contributions of each expenditure component to **real** GDP growth.

    Because chain-linked volumes do not add up (see :func:`qna_identity`),
    the share of a component in real GDP is not its weight in real growth.
    The correct weight is its **previous-period nominal share**, which is the
    same thing as valuing this period's volume change at last period's prices:

    .. math::

        \\text{contrib}_{i,t}
          = \\underbrace{\\left(\\frac{Q_{i,t}}{Q_{i,t-1}} - 1\\right)}_{\\text{volume growth}}
            \\times \\underbrace{\\frac{P_{i,t-1} Q_{i,t-1}}{P_{Y,t-1} Q_{Y,t-1}}}_{\\text{last period's nominal share}}

    which needs all three products at once — the volumes for the growth rate,
    the current-price levels for the weight. Imports enter negatively.

    The contributions sum to real GDP growth up to a ``residual``, returned as
    its own column rather than quietly spread over the components. Two things
    land in it, and they are worth telling apart before reading too much into
    a large one:

    * the **chain-linking gap** — small and slow-moving in normal quarters
      (hundredths of a point for the United States), spiking when relative
      prices move violently, as in 2020;
    * the country's **statistical discrepancy** — if the published components
      do not add to published GDP in current prices, they cannot add to it in
      contributions either. Mexico's runs near a point of GDP and drives a
      residual an order of magnitude above Spain's. :func:`qna_identity` on
      the same panel says which of the two you are looking at.

    Parameters
    ----------
    panel
        A :func:`~puremacro.fetch.qna_panel` frame built with ``real=True``.
    annualise
        Express as annualised rates, ``(1 + g)**(4 / periods) - 1``, the
        convention US releases use. Default is the plain period rate.
    periods
        Growth horizon in quarters: ``1`` for quarter on quarter (default),
        ``4`` for the same quarter a year earlier.

    Returns
    -------
    pandas.DataFrame
        ``(code, date)``-indexed, in percentage points:
        one column per component of :data:`IDENTITY_TERMS`, plus ``gdp``
        (measured real GDP growth) and ``residual`` = ``gdp`` minus the sum of
        the contributions.

    Raises
    ------
    ValueError
        If the panel carries no ``_real`` columns — rebuild it with
        ``qna_panel(..., real=True)``.

    Examples
    --------
    >>> panel = qna_panel(["USA"], real=True)                 # doctest: +SKIP
    >>> qna_contributions(panel).loc["USA"].tail(1).round(2)  # doctest: +SKIP
                cons_hh  cons_gov  capform  exports  imports   gdp  residual
    date
    2026-04-01     0.36      -0.03     0.04     0.10    -0.20  0.27      0.00
    """
    _require_panel(panel)
    if periods < 1:
        raise ValueError(f"periods must be a positive number of quarters, got {periods}")
    needed = ["gdp_real"] + [f"{n}_real" for n in IDENTITY_TERMS]
    absent = [c for c in needed if c not in panel.columns]
    if absent:
        raise ValueError(
            "qna_contributions needs volume measures; missing "
            f"{', '.join(absent)}. Rebuild with qna_panel(..., real=True).")

    by_code = panel.groupby(level="code", sort=False)
    gdp_nom_lag = by_code["gdp"].shift(periods)

    out = {}
    for name, sign in IDENTITY_TERMS.items():
        growth = by_code[f"{name}_real"].pct_change(periods, fill_method=None)
        weight = by_code[name].shift(periods) / gdp_nom_lag
        out[name] = sign * 100.0 * growth * weight

    gdp_growth = 100.0 * by_code["gdp_real"].pct_change(periods, fill_method=None)
    res = pd.DataFrame(out, index=panel.index)
    if annualise:
        # Annualise the *aggregate*, then rescale the parts by the same
        # factor: annualising each contribution on its own would not add up.
        ann = 100.0 * ((1.0 + gdp_growth / 100.0) ** (4.0 / periods) - 1.0)
        scale = (ann / gdp_growth).replace([np.inf, -np.inf], np.nan)
        res = res.mul(scale, axis=0)
        gdp_growth = ann
    res["gdp"] = gdp_growth
    res["residual"] = gdp_growth - res[list(IDENTITY_TERMS)].sum(axis=1, min_count=1)
    res.attrs["units"] = "percentage points of real GDP growth"
    res.attrs["periods"] = periods
    res.attrs["annualised"] = bool(annualise)
    return res


__all__ = ["qna_rebase", "qna_identity", "qna_contributions", "IDENTITY_TERMS"]
