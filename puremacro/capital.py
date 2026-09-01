r"""Perpetual-inventory capital stocks and TFP from a quarterly national-accounts panel.

The package could already *estimate* production functions — :mod:`puremacro.korv_gmm`
fits a capital-skill-complementarity CES, :func:`puremacro.labor_share.gollin_adjusted_ls`
corrects the labour share — but it could not *build* the capital input either of them
needs. This module closes that gap using only what :func:`puremacro.fetch.qna_panel`
already returns:

.. math::

    K_{i,t} = (1-\delta_i) K_{i,t-1} + I_{i,t-1}

one stock per asset, from gross fixed capital formation split into equipment,
structures, dwellings and intellectual property products (``assets=True``), and then a
Solow residual against hours worked (``labor=True``).

Two choices here are load-bearing and both are easy to get wrong.

**Depreciation is converted geometrically, never linearly.** A quarterly rate is
:math:`1-(1-\delta_a)^{1/4}`, not :math:`\delta_a/4`. The linear shortcut understates
depreciation and so biases the steady-state stock *up* — by +5.3% for equipment and
+8.5% for IPP at zero growth, which is larger than most of the effects a user of this
module is trying to measure.

**The aggregate is a Törnqvist index of capital services, not a sum of stocks.**
Chain-linked volumes are not additive away from their reference year, so
:math:`\sum_i K_i` depends on the price reference year the OECD happened to choose:
re-reference the same panel with :func:`puremacro.fetch.qna_rebase` and the summed stock
moves by up to 1.0%. The Törnqvist index, which aggregates *growth rates* with rental-cost
weights, is invariant to machine precision (measured: 2.6e-15). ``aggregate="sum"`` is
available and is the right choice for the four reference areas publishing fixed-base
volumes (ARG, IDN, MEX, ZAF), where additivity holds by construction.

Coverage, measured against the cached OECD responses in this checkout rather than
asserted: 34 of 49 reference areas publish all four asset classes as volumes, 33 of those
also publish them in current prices (Colombia does not, so it has no deflators and cannot
enter a services index), and 29 additionally publish the hours and employment that
:func:`qna_tfp` needs.

**What this module will not tell you honestly.** The initial stock has to be assumed, and
the assumption decays at the asset's own depreciation rate. For equipment
(:math:`\delta=0.13`) and IPP (0.20) that is fast — a 50% error in :math:`K_0` is under
0.4% within the sample. For structures (0.03) and especially dwellings (0.011) it is not:
the same error is still worth a median 10.1% and 16.4% at the end of a 30-year panel, and
the burn-in needed to clear it exceeds the entire published history for most countries.
Their *growth rates* are usable; their *levels* are an assumption wearing a number's
clothes. Every result therefore carries ``k0_sensitivity``, and you should look at it.

References
----------
Harberger, A. (1978). Perspectives on capital and technology in less developed countries.
Hulten, C. & Wykoff, F. (1981). The estimation of economic depreciation using vintage
    asset prices. *Journal of Econometrics* 15(3).
Jorgenson, D. (1963). Capital theory and investment behavior. *American Economic Review*
    53(2). — the user-cost formula the Törnqvist weights use.
OECD (2009). *Measuring Capital*, 2nd edition. — the geometric-depreciation convention.
Solow, R. (1957). Technical change and the aggregate production function. *Review of
    Economics and Statistics* 39(3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Asset column in a ``qna_panel(assets=True)`` frame -> annual geometric depreciation
#: rate. Equipment matches the BEA rate :mod:`puremacro.korv_gmm` already assumes
#: (``delta_e=0.13``), so a stock built here can be handed straight to it. Structures and
#: dwellings follow Hulten-Wykoff; the IPP rate is a weighted blend of R&D and software,
#: which on the cached data split roughly 44/43 with the remainder in originals and
#: mineral exploration.
PIM_DELTAS: dict[str, float] = {
    "inv_equip":  0.130,
    "inv_struct": 0.030,
    "inv_dwell":  0.011,
    "inv_ipp":    0.200,
}

#: The asset columns this module builds stocks for, in a fixed order.
PIM_ASSETS: tuple[str, ...] = ("inv_equip", "inv_struct", "inv_dwell", "inv_ipp")

#: Stock column produced for each asset.
_STOCK = {a: "k_" + a.removeprefix("inv_") for a in PIM_ASSETS}

_QUARTERS_PER_YEAR = 4


@dataclass(frozen=True)
class CapitalResult:
    """Per-country capital stocks, the aggregate, and what to distrust about them.

    ``stocks`` is the panel: a ``(code, date)``-indexed frame carrying one column per
    asset (``k_equip``, ``k_struct``, ``k_dwell``, ``k_ipp``), the aggregate ``k``, and
    the diagnostics described below. The remaining fields are per-country scalars.
    """

    stocks: pd.DataFrame
    deltas: dict[str, float]
    delta_quarterly: dict[str, float]
    k0_method: str
    aggregate: str
    growth: dict[str, float] = field(default_factory=dict)
    burn_in: dict[str, int] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    n_obs: int = 0

    def summary(self) -> pd.DataFrame:
        """One row per country: the assumptions made and how much they matter."""
        rows = []
        for code in self.names:
            g = self.stocks.loc[code]
            rows.append({
                "code": code,
                "trend_g_annual": self.growth.get(code, np.nan),
                "burn_in_q": self.burn_in.get(code, 0),
                "k0_sensitivity": float(g["k0_sensitivity"].iloc[-1])
                if "k0_sensitivity" in g and len(g) else np.nan,
                "coverage_pct": float(g["coverage_pct"].median())
                if "coverage_pct" in g else np.nan,
                "n_obs": int(len(g)),
            })
        return pd.DataFrame(rows)


def _quarterly_delta(annual: float) -> float:
    """Geometric conversion. ``annual/4`` is the tempting wrong answer."""
    if not (0.0 <= annual <= 1.0):
        raise ValueError(f"annual depreciation must be between 0 and 1, got {annual}")
    return 1.0 - (1.0 - annual) ** (1.0 / _QUARTERS_PER_YEAR)


def _trend_growth(gdp_real: pd.Series) -> float:
    """Geometric mean quarterly growth of real GDP, floored at zero.

    Harberger's initial stock divides by ``g + delta``. Letting a country's own asset
    growth in there is what makes it explode: on the cached panel six asset-country cells
    have ``g + delta_q`` within 0.005 of zero, which silently produces an initial stock
    worth centuries of investment rather than raising anything. Trend *GDP* growth is
    safe — the minimum across the 51 cached reference areas is Italy at +0.66%/yr, and
    none is negative — and the floor makes it safe by construction rather than by luck.
    """
    s = gdp_real.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return 0.0
    g = (s.iloc[-1] / s.iloc[0]) ** (1.0 / (len(s) - 1)) - 1.0
    return float(max(g, 0.0))


def _pim(inv: np.ndarray, dq: float, k0: float, mid: bool) -> np.ndarray:
    """The recursion itself.

    ``K[t]`` is the stock **predetermined at t**: investment made in ``t-1`` is in place
    and has not yet depreciated. That is the convention a Solow residual wants, and it is
    a quarter away from the end-of-period net stock national statistical offices publish
    — worth knowing before comparing levels against an official series.
    """
    half = (1.0 - dq) ** 0.5 if mid else 1.0
    k = np.empty(len(inv) + 1, dtype=float)
    # K_0 is scaled too, or `mid` is not the pure level factor it claims to be.
    k[0] = half * k0
    for t, i_t in enumerate(inv):
        k[t + 1] = (1.0 - dq) * k[t] + half * (0.0 if not np.isfinite(i_t) else i_t)
    return k[:-1]


def qna_capital(panel: pd.DataFrame, *,
                deltas: dict[str, float] | None = None,
                r: float = 0.04,
                k0: str = "harberger",
                k0_window: int = 20,
                aggregate: str = "tornqvist",
                capital_gains: str = "none",
                timing: str = "predetermined",
                strict: bool = False) -> CapitalResult:
    r"""Build per-asset capital stocks from a ``qna_panel(assets=True, real=True)`` frame.

    Parameters
    ----------
    panel
        A :func:`puremacro.fetch.qna_panel` result built with ``assets=True`` and
        ``real=True``. ``aggregate="tornqvist"`` additionally needs the ``_defl``
        columns, which that call also returns.
    deltas
        Annual geometric depreciation per asset; defaults to :data:`PIM_DELTAS`.
        Converted to quarterly as :math:`1-(1-\delta)^{1/4}`.
    r
        Annual real required return in the rental price. Only used by
        ``aggregate="tornqvist"``.
    k0
        ``"harberger"`` (default) sets :math:`K_0 = I_0/(g+\delta_q)` with ``g`` the
        country's floored trend real GDP growth and :math:`I_0` the mean of the first
        ``k0_window`` quarters. ``"zero"`` starts from nothing, which is honest and
        wrong for a long time.
    aggregate
        ``"tornqvist"`` (default) aggregates growth rates with rental-cost weights and is
        invariant to the price reference year. ``"sum"`` adds the stocks, which is exact
        only for fixed-base volumes (ARG, IDN, MEX, ZAF) and drifts up to 1% elsewhere.
    capital_gains
        ``"none"`` (default) uses a rental price of :math:`P_i(r_q+\delta_{q,i})`.
        ``"expost"`` subtracts realised asset-price inflation, which is Jorgenson's
        formula and which goes *negative* in 38% of quarters for dwellings on this data:
        the hurdle :math:`r_q+\delta_q` is 0.0128 against a median quarterly price-change
        standard deviation of 0.0158, so a one-sigma move flips the sign. Use it only
        with smoothed deflators.
    timing
        ``"predetermined"`` (default) or ``"mid"``, which places investment mid-quarter.
        ``"mid"`` is a pure level factor of :math:`(1-\delta_q)^{1/2}` — provided
        :math:`K_0` is scaled with it, which this does.
    strict
        Raise instead of skipping a country that is missing an asset or a deflator.

    Returns
    -------
    CapitalResult
        See :class:`CapitalResult`. The ``stocks`` frame carries ``k0_sensitivity``, the
        proportional change in the aggregate from a +50% shock to every :math:`K_0` —
        read it before quoting a level.
    """
    if aggregate not in ("tornqvist", "sum"):
        raise ValueError(f"aggregate must be 'tornqvist' or 'sum', got {aggregate!r}")
    if capital_gains not in ("none", "expost"):
        raise ValueError(f"capital_gains must be 'none' or 'expost', got {capital_gains!r}")
    if timing not in ("predetermined", "mid"):
        raise ValueError(f"timing must be 'predetermined' or 'mid', got {timing!r}")
    if k0 not in ("harberger", "zero"):
        raise ValueError(f"k0 must be 'harberger' or 'zero', got {k0!r}")

    deltas = dict(PIM_DELTAS if deltas is None else deltas)
    dq = {a: _quarterly_delta(d) for a, d in deltas.items()}
    r_q = (1.0 + r) ** (1.0 / _QUARTERS_PER_YEAR) - 1.0
    mid = timing == "mid"

    assets = [a for a in PIM_ASSETS if a in deltas]
    need_defl = aggregate == "tornqvist"

    out, growth, burn = [], {}, {}
    for code, g in panel.groupby(level="code", sort=True):
        res = _process_country(code, g, assets, strict, need_defl, k0_window, k0, dq, mid, r_q, aggregate, capital_gains)
        if res is not None:
            f, gr, b = res
            growth[code] = gr
            burn[code] = b
            out.append(f)
    stocks = (pd.concat(out).sort_index() if out
              else pd.DataFrame(index=pd.MultiIndex.from_arrays(
                  [[], []], names=["code", "date"])))
    names = tuple(sorted({c for c, _ in stocks.index})) if len(stocks) else ()
    res = CapitalResult(stocks=stocks, deltas=deltas, delta_quarterly=dq,
                        k0_method=k0, aggregate=aggregate, growth=growth,
                        burn_in=burn, names=names, n_obs=int(len(stocks)))
    stocks.attrs["deltas"] = deltas
    stocks.attrs["k0_method"] = k0
    return res



def _process_country(code: str, g: pd.DataFrame, assets: list[str], strict: bool,
                     need_defl: bool, k0_window: int, k0: str, dq: dict[str, float],
                     mid: bool, r_q: float, aggregate: str, capital_gains: str) -> tuple[pd.DataFrame, float, int] | None:
    g = g.droplevel("code").sort_index()
    vols = {a: f"{a}_real" for a in assets}
    missing = [a for a in assets if vols[a] not in g or g[vols[a]].dropna().empty]
    if missing:
        if strict:
            raise ValueError(f"{code}: no volume for {', '.join(missing)}")
        return None
    if need_defl:
        no_defl = [a for a in assets
                   if f"{a}_defl" not in g or g[f"{a}_defl"].dropna().empty]
        if no_defl:
            if strict:
                raise ValueError(
                    f"{code}: aggregate='tornqvist' needs deflators, missing "
                    f"{', '.join(no_defl)}. Colombia publishes asset volumes and no "
                    f"current prices; use aggregate='sum' for it.")
            return None

    gg = _trend_growth(g["gdp_real"]) if "gdp_real" in g else 0.0
    growth_val = float((1.0 + gg) ** _QUARTERS_PER_YEAR - 1.0)

    frame = pd.DataFrame(index=g.index)
    shocked = {}
    for a in assets:
        inv = g[vols[a]].to_numpy(dtype=float)
        start = float(np.nanmean(inv[:k0_window])) if np.isfinite(inv[:k0_window]).any() else 0.0
        k_init = 0.0 if k0 == "zero" else start / max(gg + dq[a], 1e-6)
        frame[_STOCK[a]] = _pim(inv, dq[a], k_init, mid)
        shocked[a] = _pim(inv, dq[a], 1.5 * k_init, mid)

    frame = _aggregate(frame, g, assets, dq, r_q, aggregate, capital_gains)
    shock_frame = pd.DataFrame(
        {_STOCK[a]: shocked[a] for a in assets}, index=g.index)
    shock_frame = _aggregate(shock_frame, g, assets, dq, r_q, aggregate,
                             capital_gains)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["k0_sensitivity"] = np.abs(shock_frame["k"] / frame["k"] - 1.0)

    frame["coverage_pct"] = _coverage(g, assets)
    # quarters until a 50% K0 error is worth under 1% of the aggregate
    below = np.flatnonzero(frame["k0_sensitivity"].to_numpy() < 0.01)
    burn_val = int(below[0]) if below.size else int(len(frame))

    frame["code"] = code
    return frame.reset_index().set_index(["code", "date"]), growth_val, burn_val

def _aggregate(frame: pd.DataFrame, g: pd.DataFrame, assets: list[str],
               dq: dict[str, float], r_q: float, how: str,
               capital_gains: str) -> pd.DataFrame:
    """Attach the aggregate ``k`` to a frame of per-asset stocks."""
    cols = [_STOCK[a] for a in assets]
    if how == "sum":
        frame["k"] = frame[cols].sum(axis=1)
        return frame

    # Jorgensonian rental price, then a Törnqvist index of the growth rates. Aggregating
    # growth rather than levels is what makes this invariant to the price reference year.
    shares, dlogs = {}, {}
    for a in assets:
        p = g[f"{a}_defl"].to_numpy(dtype=float)
        uc = p * (r_q + dq[a])
        if capital_gains == "expost":
            infl = np.concatenate([[np.nan], np.diff(np.log(np.where(p > 0, p, np.nan)))])
            uc = p * (r_q + dq[a] - infl)
        k = frame[_STOCK[a]].to_numpy(dtype=float)
        shares[a] = uc * k
        with np.errstate(divide="ignore", invalid="ignore"):
            dlogs[a] = np.concatenate([[np.nan], np.diff(np.log(np.where(k > 0, k, np.nan)))])

    value = np.vstack([shares[a] for a in assets])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = value / np.nansum(value, axis=0)
    w = np.where(np.isfinite(w), w, 0.0)
    # Törnqvist: the two-period average share weights each asset's growth.
    w_bar = np.concatenate([np.full((len(assets), 1), np.nan),
                            0.5 * (w[:, 1:] + w[:, :-1])], axis=1)
    dl = np.vstack([dlogs[a] for a in assets])
    dlog_k = np.nansum(w_bar * dl, axis=0)
    dlog_k[0] = 0.0
    level = np.exp(np.cumsum(np.where(np.isfinite(dlog_k), dlog_k, 0.0)))
    # Anchor on the first quarter whose summed stock is strictly positive, so the index
    # lands in currency units. Not simply the first quarter: under k0="zero" every stock
    # starts at 0, and anchoring there would multiply the whole index by zero.
    totals = frame[cols].sum(axis=1).to_numpy(dtype=float)
    positive = np.flatnonzero(np.isfinite(totals) & (totals > 0.0))
    if positive.size == 0:
        frame["k"] = np.nan
        return frame
    j = int(positive[0])
    frame["k"] = level / level[j] * totals[j]
    return frame


def _coverage(g: pd.DataFrame, assets: list[str]) -> np.ndarray:
    """Current-price share of total GFCF the four asset classes actually account for.

    Australia's four components cover about 71% of its published total, with no further
    asset code that closes the gap — which is a scope defect in the source rather than a
    rounding issue, and a reason to distrust an Australian capital aggregate.
    """
    have = [a for a in assets if a in g]
    if not have or "inv" not in g:
        return np.full(len(g), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (g[have].sum(axis=1) / g["inv"]).to_numpy(dtype=float) * 100.0


@dataclass(frozen=True)
class TFPResult:
    """A Solow residual and the pieces it was built from.

    ``tfp`` is a ``(code, date)``-indexed frame with ``log_tfp`` (the residual),
    ``tfp`` (its exponential, indexed to 100 in each country's first usable quarter),
    the ``labor_share`` used, and the two input contributions, so the decomposition can
    be checked rather than taken on trust.
    """

    tfp: pd.DataFrame
    share: str
    share_mode: str
    labor: str
    labor_share: dict[str, float] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    n_obs: int = 0

    def summary(self) -> pd.DataFrame:
        """One row per country: the labour share used and average TFP growth."""
        rows = []
        for code in self.names:
            g = self.tfp.loc[code]["log_tfp"].dropna()
            ann = (g.iloc[-1] - g.iloc[0]) / max(len(g) - 1, 1) * 4 * 100 if len(g) > 1 else np.nan
            rows.append({"code": code,
                         "labor_share": self.labor_share.get(code, np.nan),
                         "tfp_growth_pct_yr": ann,
                         "n_obs": int(len(g))})
        return pd.DataFrame(rows)


def qna_tfp(panel: pd.DataFrame, capital: CapitalResult | None = None, *,
            share: str = "gollin", share_mode: str = "mean",
            labor: str = "hours", output: str = "va_factor",
            strict: bool = False, **capital_kw) -> TFPResult:
    r"""Solow residual from a national-accounts panel and its capital stock.

    .. math:: \log A_t = \log Y_t - s_L \log L_t - (1-s_L) \log K_t

    Parameters
    ----------
    panel
        A :func:`puremacro.fetch.qna_panel` result built with ``assets=True``,
        ``labor=True``, ``income=True`` and ``real=True``. ``output="va"`` also needs
        ``output=True``.
    capital
        A :func:`qna_capital` result to reuse. Built here from ``panel`` if omitted,
        passing through any extra keyword arguments.
    share
        ``"gollin"`` (default) applies the Gollin (2002) correction, which scales the
        unadjusted share by the reciprocal of the employee share of the workforce and so
        needs the employment split ``labor=True`` provides. ``"unadjusted"`` is
        ``comp_emp / Y``, which reads low for countries with much self-employment for
        reasons that have nothing to do with how employees are paid.
    share_mode
        ``"mean"`` (default) holds the share fixed at its country mean, as the Solow
        decomposition assumes. ``"varying"`` uses it quarter by quarter.
    labor
        ``"hours"`` (default) or ``"emp"``. Hours is the better input measure and the
        margin that carries European recessions; heads are available for more countries.
    output
        ``"va_factor"`` (default) is gross value added at **factor cost**
        (``gdp_income - taxes_prod_imp_net``, i.e. :math:`D1 + B2A3G`) deflated by the
        GDP deflator — the denominator that matches a factor-share decomposition, and
        available for the United States, which publishes nothing in the by-activity
        output flow. ``"va"`` uses ``va_total_real``; ``"gdp"`` uses ``gdp_real``.

    Returns
    -------
    TFPResult
        See :class:`TFPResult`.

    Notes
    -----
    South Africa is excluded under ``output="va_factor"``: it publishes GDP and
    compensation of employees but neither ``taxes_prod_imp_net`` nor ``surplus_mixed``.
    """
    if share not in ("gollin", "unadjusted"):
        raise ValueError(f"share must be 'gollin' or 'unadjusted', got {share!r}")
    if share_mode not in ("mean", "varying"):
        raise ValueError(f"share_mode must be 'mean' or 'varying', got {share_mode!r}")
    if labor not in ("hours", "emp"):
        raise ValueError(f"labor must be 'hours' or 'emp', got {labor!r}")
    if output not in ("va_factor", "va", "gdp"):
        raise ValueError(f"output must be 'va_factor', 'va' or 'gdp', got {output!r}")

    cap = qna_capital(panel, **capital_kw) if capital is None else capital
    rows, share_by_code = [], {}
    for code in cap.names:
        g = panel.loc[code].sort_index()
        k = cap.stocks.loc[code]["k"]
        y = _output(g, output)
        lab = g.get(labor)
        if y is None or lab is None or y.dropna().empty or lab.dropna().empty:
            if strict:
                raise ValueError(f"{code}: no {output} or no {labor}")
            continue
        s_l = _labor_share(g, share)
        if s_l is None or s_l.dropna().empty:
            if strict:
                raise ValueError(f"{code}: cannot form a {share} labour share")
            continue
        s_use = float(s_l.mean()) if share_mode == "mean" else s_l
        share_by_code[code] = float(np.mean(s_use)) if share_mode == "varying" else s_use

        with np.errstate(divide="ignore", invalid="ignore"):
            ly, ll, lk = np.log(y), np.log(lab), np.log(k)
        contrib_l = s_use * ll
        contrib_k = (1.0 - s_use) * lk
        log_tfp = ly - contrib_l - contrib_k
        frame = pd.DataFrame({
            "log_tfp": log_tfp,
            "labor_share": s_use if share_mode == "varying" else float(s_use),
            "contrib_labor": contrib_l,
            "contrib_capital": contrib_k,
        }, index=g.index)
        ok = frame["log_tfp"].dropna()
        if ok.empty:
            continue
        frame["tfp"] = np.exp(frame["log_tfp"] - ok.iloc[0]) * 100.0
        frame["code"] = code
        rows.append(frame.reset_index().set_index(["code", "date"]))

    tfp = (pd.concat(rows).sort_index() if rows
           else pd.DataFrame(index=pd.MultiIndex.from_arrays(
               [[], []], names=["code", "date"])))
    names = tuple(sorted({c for c, _ in tfp.index})) if len(tfp) else ()
    return TFPResult(tfp=tfp, share=share, share_mode=share_mode, labor=labor,
                     labor_share=share_by_code, names=names, n_obs=int(len(tfp)))


def _output(g: pd.DataFrame, how: str) -> pd.Series | None:
    """Real output, in the measure the caller asked for."""
    if how == "gdp":
        return g.get("gdp_real")
    if how == "va":
        return g.get("va_total_real")
    # value added at factor cost: D1 + B2A3G, deflated by the GDP deflator so it is a
    # volume comparable with the capital and labour inputs.
    if "gdp_income" not in g or "taxes_prod_imp_net" not in g or "gdp_defl" not in g:
        return None
    nominal = g["gdp_income"] - g["taxes_prod_imp_net"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return nominal / g["gdp_defl"] * 100.0


def _labor_share(g: pd.DataFrame, how: str) -> pd.Series | None:
    """Labour share of value added at factor cost."""
    if "comp_emp" not in g or "gdp_income" not in g or "taxes_prod_imp_net" not in g:
        return None
    va = g["gdp_income"] - g["taxes_prod_imp_net"]
    with np.errstate(divide="ignore", invalid="ignore"):
        unadj = (g["comp_emp"] / va).replace([np.inf, -np.inf], np.nan)
    if how == "unadjusted":
        return unadj.clip(upper=1.0)
    if "emp_employees" not in g or "emp_selfemp" not in g:
        return None
    # Gollin Adjustment 2: scale the unadjusted share by the reciprocal of the employee
    # share of the workforce, i.e. impute to the self-employed the same average labour
    # income an employee earns. `mixed_income` is in that function's input schema but is
    # never read, so it is not required here either.
    workforce = g["emp_employees"] + g["emp_selfemp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (workforce / g["emp_employees"]).replace([np.inf, -np.inf], np.nan)
    return (unadj * scale).clip(upper=1.0)


__all__ = ["qna_capital", "qna_tfp", "CapitalResult", "TFPResult",
           "PIM_DELTAS", "PIM_ASSETS"]
