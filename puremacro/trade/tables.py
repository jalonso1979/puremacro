"""Dedicated LaTeX Table Generation and Formatting Engine for puremacro.trade.

This module provides high-level table generation routines that extract macroeconomic
aggregates from ScenarioBatchResult, format publication-ready pandas DataFrames, and
emit identical LaTeX tabular code matching the paper's canonical tables:
- selected_country_impacts.tex
- mean_by_scenario.tex
- weighted_mean_by_scenario.tex
- selected_country_impacts_with_ROW.tex

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero dev-dependencies in the runtime path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from puremacro.trade._results import ScenarioBatchResult
from puremacro.trade.data import CANONICAL_COUNTRY_CODES, EU_COUNTRY_CODES
from puremacro.trade.geary_khamis import restore_capital_formation


# ---------------------------------------------------------------------------
# DataFrame Table Builders
# ---------------------------------------------------------------------------

def generate_selected_country_table(
    batch_res: ScenarioBatchResult,
    countries: Sequence[str] | None = None,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Generate publication DataFrame of macroeconomic impacts on selected economies.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved baseline and counterfactual trade equilibrium results.
    countries : Sequence[str] | None, default None
        List of country codes to include. Defaults to ['CAN', 'CHN', 'EU_', 'MEX', 'USA'].
        Supports 'EU_' and 'EUR' as the GDP-weighted aggregate of EU-27 member states.
    round_digits : int, default 2
        Number of decimal places to round numerical values.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame indexed by ('Section', 'Country') with columns for each
        counterfactual scenario and a 'Base' column for baseline levels.
    """
    if countries is None:
        c_list = ["CAN", "CHN", "EU_", "MEX", "USA"]
    else:
        c_list = list(countries)

    all_countries = (
        list(batch_res.country_codes)
        if batch_res.country_codes
        else list(batch_res.real_gdp_table.index)
    )
    if not batch_res.scenarios or not all_countries:
        idx = pd.MultiIndex.from_tuples([], names=["Section", "Country"])
        return pd.DataFrame(index=idx)

    is_eu = np.array([c in EU_COUNTRY_CODES for c in all_countries], dtype=bool)

    # Extract baseline consumption weights for EU aggregation
    base_res = batch_res.scenarios.get(batch_res.baseline_scenario)
    matlab_compat = bool(batch_res.metadata.get("matlab_compat", True))
    if base_res is not None and base_res.c_sol is not None:
        c_base = restore_capital_formation(
            base_res.c_sol, base_res.XN_sol, matlab_compat=matlab_compat
        )
        gdp_b = np.sum(c_base[0] if c_base.ndim == 3 else c_base, axis=0).ravel()
    else:
        gdp_b = np.ones(len(all_countries))

    sum_eu_gdp = float(np.sum(gdp_b[is_eu]))
    n_eu = int(np.sum(is_eu))
    w_eu = gdp_b[is_eu] / sum_eu_gdp if sum_eu_gdp > 0 else np.ones(n_eu) / max(n_eu, 1)

    cf_scens = [s for s in batch_res.scenarios.keys() if s != batch_res.baseline_scenario]
    if not cf_scens:
        cf_scens = list(batch_res.scenarios.keys())

    def _get_vals(metric_df: pd.DataFrame, c_code: str) -> list[float]:
        if c_code in ("EUR", "EU_"):
            if np.any(is_eu) and not metric_df.empty:
                eu_sub = metric_df.iloc[is_eu]
                return [float(np.sum(w_eu * eu_sub[s])) for s in cf_scens]
            return [0.0] * len(cf_scens)
        if c_code in metric_df.index:
            return [float(metric_df.loc[c_code, s]) for s in cf_scens]
        return [0.0] * len(cf_scens)

    rows: list[dict[str, Any]] = []

    # Section 1: GDP growth (%)
    for c in c_list:
        vals = _get_vals(batch_res.real_gdp_table, c)
        row: dict[str, Any] = {"Section": "GDP growth (%)", "Country": c}
        for s, v in zip(cf_scens, vals):
            row[s] = round(v, round_digits)
        row["Base"] = np.nan
        rows.append(row)

    # Section 2: Inflation (%)
    infl_table = (
        (batch_res.cpi_table - 1.0) * 100.0
        if not batch_res.cpi_table.empty
        else pd.DataFrame()
    )
    for c in c_list:
        vals = _get_vals(infl_table, c)
        row_inf: dict[str, Any] = {"Section": "Inflation (%)", "Country": c}
        for s, v in zip(cf_scens, vals):
            row_inf[s] = round(v, round_digits)
        row_inf["Base"] = np.nan
        rows.append(row_inf)

    # Section 3: Net exports / GDP (%)
    xn_gdp_df = batch_res.to_frame("xn_over_gdp")
    for c in c_list:
        vals = _get_vals(xn_gdp_df, c)
        row_xn: dict[str, Any] = {"Section": "Net exports / GDP (%)", "Country": c}
        for s, v in zip(cf_scens, vals):
            row_xn[s] = round(v, round_digits)
        if batch_res.baseline_scenario in xn_gdp_df.columns:
            if c in ("EUR", "EU_") and np.any(is_eu):
                base_xn_eu = xn_gdp_df.iloc[is_eu][batch_res.baseline_scenario]
                row_xn["Base"] = round(float(np.sum(w_eu * base_xn_eu)), round_digits)
            elif c in xn_gdp_df.index:
                row_xn["Base"] = round(
                    float(xn_gdp_df.loc[c, batch_res.baseline_scenario]), round_digits
                )
            else:
                row_xn["Base"] = np.nan
        else:
            row_xn["Base"] = np.nan
        rows.append(row_xn)

    return pd.DataFrame(rows).set_index(["Section", "Country"])


def generate_mean_by_scenario_table(
    batch_res: ScenarioBatchResult,
    round_digits: int = 3,
) -> pd.DataFrame:
    """Generate DataFrame of unweighted cross-country means and standard deviations.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved baseline and counterfactual trade equilibrium results.
    round_digits : int, default 3
        Number of decimal places for formatted 'mean (std)' strings.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'Variable' with scenario columns containing formatted strings.
    """
    cf_scens = [s for s in batch_res.scenarios.keys() if s != batch_res.baseline_scenario]
    if not cf_scens:
        cf_scens = list(batch_res.scenarios.keys())

    gdp_df = batch_res.real_gdp_table[cf_scens]
    infl_df = ((batch_res.cpi_table - 1.0) * 100.0)[cf_scens]
    xn_df = batch_res.to_frame("xn_over_gdp")[cf_scens]

    records: list[dict[str, Any]] = []
    for label, metric_df in [
        ("GDP growth (%)", gdp_df),
        ("Inflation (%)", infl_df),
        ("XN over GDP (%)", xn_df),
    ]:
        row: dict[str, Any] = {"Variable": label}
        for s in cf_scens:
            m = float(np.mean(metric_df[s]))
            sd = float(np.std(metric_df[s], ddof=1))
            row[s] = f"{m:.{round_digits}f} ({sd:.{round_digits}f})"
        records.append(row)

    return pd.DataFrame(records).set_index("Variable")


def generate_weighted_mean_by_scenario_table(
    batch_res: ScenarioBatchResult,
    round_digits: int = 3,
) -> pd.DataFrame:
    """Generate DataFrame of GDP-weighted cross-country means and standard deviations.

    Replicates MATLAB Resultadosl.m:337-365. Weights are derived from baseline
    consumption across all countries.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved baseline and counterfactual trade equilibrium results.
    round_digits : int, default 3
        Number of decimal places for formatted 'mean (std)' strings.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'Variable' with scenario columns containing formatted strings.
    """
    cf_scens = [s for s in batch_res.scenarios.keys() if s != batch_res.baseline_scenario]
    if not cf_scens:
        cf_scens = list(batch_res.scenarios.keys())

    all_countries = (
        list(batch_res.country_codes)
        if batch_res.country_codes
        else list(batch_res.real_gdp_table.index)
    )
    base_res = batch_res.scenarios.get(batch_res.baseline_scenario)
    matlab_compat = bool(batch_res.metadata.get("matlab_compat", True))
    if base_res is not None and base_res.c_sol is not None:
        c_base = restore_capital_formation(
            base_res.c_sol, base_res.XN_sol, matlab_compat=matlab_compat
        )
        gdp_b = np.sum(c_base[0] if c_base.ndim == 3 else c_base, axis=0).ravel()
    else:
        gdp_b = np.ones(len(all_countries))

    w = gdp_b / np.sum(gdp_b)

    gdp_df = batch_res.real_gdp_table[cf_scens]
    infl_df = ((batch_res.cpi_table - 1.0) * 100.0)[cf_scens]
    xn_df = batch_res.to_frame("xn_over_gdp")[cf_scens]

    records: list[dict[str, Any]] = []
    for label, metric_df in [
        ("GDP growth (%)", gdp_df),
        ("Inflation (%)", infl_df),
        ("XN over GDP (%)", xn_df),
    ]:
        row: dict[str, Any] = {"Variable": label}
        for s in cf_scens:
            x = np.asarray(metric_df[s], dtype=float)
            wm = float(np.sum(w * x))
            wsd = float(np.sqrt(np.sum(w * (x - wm) ** 2)))
            row[s] = f"{wm:.{round_digits}f} ({wsd:.{round_digits}f})"
        records.append(row)

    return pd.DataFrame(records).set_index("Variable")


# ---------------------------------------------------------------------------
# LaTeX Table String Formatters
# ---------------------------------------------------------------------------

def to_latex_selected_country_table(
    batch_res: ScenarioBatchResult,
    countries: Sequence[str] | None = None,
    round_digits: int = 2,
    legacy_compat: bool = True,
) -> str:
    """Render selected country impacts table as an identical LaTeX string.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved trade equilibrium results.
    countries : Sequence[str] | None, default None
        List of country codes. Default ['CAN', 'CHN', 'EU_', 'MEX', 'USA'].
    round_digits : int, default 2
        Decimal places for floating point values.
    legacy_compat : bool, default True
        If True, reproduces the exact verbatim text of MATLAB Resultadosl.m
        matching headlinePaper/selected_country_impacts.tex byte-for-byte
        (including the 7-column net exports row). The legacy layout only
        applies to the canonical 5-scenario batch; a batch with any other
        number of counterfactual scenarios always gets the clean layout with
        the ``& Base`` column, because the MATLAB text is undefined for it.
        If False, outputs clean 7-column LaTeX headers (& Base) to prevent
        LaTeX overfull alignment tab errors.

    Returns
    -------
    str
        LaTeX table string.
    """
    if countries is None:
        c_list = ["CAN", "CHN", "EU_", "MEX", "USA"]
    else:
        c_list = list(countries)

    tbl = generate_selected_country_table(
        batch_res, countries=c_list, round_digits=round_digits
    )
    cf_scens = [col for col in tbl.columns if col != "Base"]
    n_scen = len(cf_scens)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"% zebra striping via xcolor")
    lines.append(r"\rowcolors{2}{gray!15}{white}")

    use_legacy = legacy_compat and (n_scen == 5)
    col_count = n_scen if use_legacy else (n_scen + 1)
    lines.append(r"\begin{tabular}{l" + "r" * col_count + "}")
    lines.append(r"\toprule")

    hdr_scens = " & ".join([s.replace("_", r"\_") for s in cf_scens])
    if use_legacy:
        lines.append(f"Country & {hdr_scens} \\\\")
    else:
        lines.append(f"Country & {hdr_scens} & Base \\\\")

    lines.append(r"\midrule")

    # Section 1: GDP growth (%)
    sec1_span = (n_scen + 1) if use_legacy else (col_count + 1)
    lines.append(
        r"    \multicolumn{" + str(sec1_span) + r"}{l}{\textbf{GDP growth (\%)}} \\"
    )
    for c in c_list:
        row_vals = [tbl.loc[("GDP growth (%)", c), s] for s in cf_scens]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        if not use_legacy:
            row_str += " &       "
        row_str += r" \\"
        lines.append(row_str)

    # Section 2: Inflation (%)
    lines.append(r"    \midrule")
    lines.append(
        r"    \multicolumn{" + str(sec1_span) + r"}{l}{\textbf{Inflation (\%)}} \\"
    )
    for c in c_list:
        row_vals = [tbl.loc[("Inflation (%)", c), s] for s in cf_scens]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        if not use_legacy:
            row_str += " &       "
        row_str += r" \\"
        lines.append(row_str)

    # Section 3: Net exports / GDP (%)
    lines.append(r"    \midrule")
    lines.append(
        r"    \multicolumn{" + str(sec1_span) + r"}{l}{\textbf{Net exports / GDP (\%)}} \\"
    )
    for c in c_list:
        row_vals = [tbl.loc[("Net exports / GDP (%)", c), s] for s in cf_scens]
        base_val = tbl.loc[("Net exports / GDP (%)", c), "Base"]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        row_str += f" & {base_val:6.{round_digits}f}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    # Note: Unicode \u2010 hyphen in tariff‐shock matches paper LaTeX
    lines.append(
        "  \\caption{Impact (% change) of tariff\u2010shock scenarios on selected economies}"
    )
    lines.append(r"  \label{tab:selected_country_impacts}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


to_latex_selected_country = to_latex_selected_country_table


def to_latex_mean_by_scenario_table(
    batch_res: ScenarioBatchResult,
    round_digits: int = 3,
    legacy_compat: bool = True,
) -> str:
    """Render unweighted cross-country mean impact table as LaTeX threeparttable string.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved trade equilibrium results.
    round_digits : int, default 3
        Number of decimal places.
    legacy_compat : bool, default True
        If True, reproduces the exact verbatim text of MATLAB Resultadosl.m
        matching headlinePaper/mean_by_scenario.tex byte-for-byte (including double
        backslash footnote formatting).

    Returns
    -------
    str
        LaTeX table string.
    """
    cf_scens = [s for s in batch_res.scenarios.keys() if s != batch_res.baseline_scenario]
    if not cf_scens:
        cf_scens = list(batch_res.scenarios.keys())

    gdp_df = batch_res.real_gdp_table[cf_scens]
    infl_df = ((batch_res.cpi_table - 1.0) * 100.0)[cf_scens]
    xn_df = batch_res.to_frame("xn_over_gdp")[cf_scens]

    mean_gdp = [float(np.mean(gdp_df[s])) for s in cf_scens]
    std_gdp = [float(np.std(gdp_df[s], ddof=1)) for s in cf_scens]

    mean_inf = [float(np.mean(infl_df[s])) for s in cf_scens]
    std_inf = [float(np.std(infl_df[s], ddof=1)) for s in cf_scens]

    mean_xn = [float(np.mean(xn_df[s])) for s in cf_scens]
    std_xn = [float(np.std(xn_df[s], ddof=1)) for s in cf_scens]

    if legacy_compat:
        # Matches MATLAB Resultadosl.m literal escaping
        footnote = (
            r"\\textit{Notes:} Columns 1–5 report cross-country means (and standard deviations) "
            r"under five tariff-shock scenarios: "
            r"\\textbf{t10} = uniform 10\\% tariff increase; "
            r"\\textbf{t10\\_25} = graduated schedule 10\\%–25\\%; "
            r"\\textbf{t10\\_54} = 10\\%–54\\%; "
            r"\\textbf{t10\\_125} = 10\\%–125\\%; "
            r"\\textbf{t10\\_145} = 10\\%–145\\%."
        )
    else:
        footnote = (
            r"\textit{Notes:} Columns 1–5 report cross-country means (and standard deviations) "
            r"under five tariff-shock scenarios: "
            r"\textbf{t10} = uniform 10\% tariff increase; "
            r"\textbf{t10\_25} = graduated schedule 10\%–25\%; "
            r"\textbf{t10\_54} = 10\%–54\%; "
            r"\textbf{t10\_125} = 10\%–125\%; "
            r"\textbf{t10\_145} = 10\%–145\%."
        )

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\caption{Mean values by tariff-shock scenario}")
    lines.append(r"\label{tab:means_by_scenario}")
    lines.append("")
    lines.append(r"\begin{tabular}{l" + "r" * len(cf_scens) + "}")
    lines.append(r"\toprule")

    hdr = "      Variable "
    for s in cf_scens:
        hdr += " & " + s.replace("_", r"\_")
    hdr += r" \\"
    lines.append(hdr)
    lines.append(r"      \midrule")

    # GDP growth row
    row_gdp = r"GDP~growth~(\%)"
    for m_val, s_val in zip(mean_gdp, std_gdp):
        row_gdp += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_gdp += r" \\"
    lines.append(row_gdp)

    # Inflation row
    row_inf = r"Inflation~(\%)   "
    for m_val, s_val in zip(mean_inf, std_inf):
        row_inf += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_inf += r"\\" if legacy_compat else r" \\"
    lines.append(row_inf)

    # XN over GDP row
    row_xn = r"XN~over~GDP~(\%) "
    for m_val, s_val in zip(mean_xn, std_xn):
        row_xn += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_xn += r"\\" if legacy_compat else r" \\"
    lines.append(row_xn)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\begin{tablenotes}")
    lines.append(f"\\footnotesize {footnote}")
    lines.append(r"\end{tablenotes}")
    lines.append("")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


to_latex_mean_by_scenario = to_latex_mean_by_scenario_table


def to_latex_weighted_mean_by_scenario_table(
    batch_res: ScenarioBatchResult,
    round_digits: int = 3,
    legacy_compat: bool = True,
) -> str:
    """Render GDP-weighted cross-country mean impact table as LaTeX threeparttable string.

    Replicates MATLAB Resultadosl.m:366-418 and weighted_mean_by_scenario.tex.
    """
    cf_scens = [s for s in batch_res.scenarios.keys() if s != batch_res.baseline_scenario]
    if not cf_scens:
        cf_scens = list(batch_res.scenarios.keys())

    all_countries = (
        list(batch_res.country_codes)
        if batch_res.country_codes
        else list(batch_res.real_gdp_table.index)
    )
    base_res = batch_res.scenarios.get(batch_res.baseline_scenario)
    matlab_compat = bool(batch_res.metadata.get("matlab_compat", True))
    if base_res is not None and base_res.c_sol is not None:
        c_base = restore_capital_formation(
            base_res.c_sol, base_res.XN_sol, matlab_compat=matlab_compat
        )
        gdp_b = np.sum(c_base[0] if c_base.ndim == 3 else c_base, axis=0).ravel()
    else:
        gdp_b = np.ones(len(all_countries))

    w = gdp_b / np.sum(gdp_b)

    gdp_df = batch_res.real_gdp_table[cf_scens]
    infl_df = ((batch_res.cpi_table - 1.0) * 100.0)[cf_scens]
    xn_df = batch_res.to_frame("xn_over_gdp")[cf_scens]

    wm_gdp, wsd_gdp = [], []
    wm_inf, wsd_inf = [], []
    wm_xn, wsd_xn = [], []

    for s in cf_scens:
        x_g = np.asarray(gdp_df[s], dtype=float)
        m_g = float(np.sum(w * x_g))
        wm_gdp.append(m_g)
        wsd_gdp.append(float(np.sqrt(np.sum(w * (x_g - m_g) ** 2))))

        x_i = np.asarray(infl_df[s], dtype=float)
        m_i = float(np.sum(w * x_i))
        wm_inf.append(m_i)
        wsd_inf.append(float(np.sqrt(np.sum(w * (x_i - m_i) ** 2))))

        x_x = np.asarray(xn_df[s], dtype=float)
        m_x = float(np.sum(w * x_x))
        wm_xn.append(m_x)
        wsd_xn.append(float(np.sqrt(np.sum(w * (x_x - m_x) ** 2))))

    if legacy_compat:
        # Note: \u2010 (hyphen) and \u2013 (en-dash) match Resultadosl.m exactly
        footnote = (
            "\\\\textit{Notes:} Columns 1\u20135 report GDP\u2010weighted cross\u2010country means "
            "(and standard deviations) under five tariff\u2010shock scenarios: "
            "\\\\textbf{t10} = uniform 10\\\\% tariff increase; "
            "\\\\textbf{t10\\\\_25} = graduated schedule 10\\\\%\u201325\\\\%; "
            "\\\\textbf{t10\\\\_54} = 10\\\\%\u201354\\\\%; "
            "\\\\textbf{t10\\\\_125} = 10\\\\%\u2013125\\\\%; "
            "\\\\textbf{t10\\\\_145} = 10\\\\%\u2013145\\\\%."
        )
    else:
        footnote = (
            r"\textit{Notes:} Columns 1–5 report GDP-weighted cross-country means "
            r"(and standard deviations) under five tariff-shock scenarios: "
            r"\textbf{t10} = uniform 10\% tariff increase; "
            r"\textbf{t10\_25} = graduated schedule 10\%–25\%; "
            r"\textbf{t10\_54} = 10\%–54\%; "
            r"\textbf{t10\_125} = 10\%–125\%; "
            r"\textbf{t10\_145} = 10\%–145\%."
        )

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{threeparttable}")
    lines.append("    \\caption{GDP\u2010weighted means by tariff\u2010shock scenario}")
    lines.append(r"    \label{tab:weighted_means_by_scenario}")
    lines.append("")
    lines.append(r"    \begin{tabular}{l" + "r" * len(cf_scens) + "}")
    lines.append(r"      \toprule")

    hdr = "      Variable "
    for s in cf_scens:
        hdr += " & " + s.replace("_", r"\_")
    hdr += r" \\"
    lines.append(hdr)
    lines.append(r"      \midrule")

    row_gdp = r"      GDP~growth~(\%)  "
    for m_val, s_val in zip(wm_gdp, wsd_gdp):
        row_gdp += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_gdp += r" \\"
    lines.append(row_gdp)

    row_inf = r"      Inflation~(\%)   "
    for m_val, s_val in zip(wm_inf, wsd_inf):
        row_inf += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_inf += r" \\"
    lines.append(row_inf)

    row_xn = r"      XN~over~GDP~(\%) "
    for m_val, s_val in zip(wm_xn, wsd_xn):
        row_xn += f" & {m_val:6.{round_digits}f}\\,(\\scriptsize {s_val:4.{round_digits}f})"
    row_xn += r" \\"
    lines.append(row_xn)

    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append("")
    lines.append(r"    \begin{tablenotes}")
    lines.append(f"      \\footnotesize {footnote}")
    lines.append(r"    \end{tablenotes}")
    lines.append("")
    lines.append(r"  \end{threeparttable}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


to_latex_weighted_mean_by_scenario = to_latex_weighted_mean_by_scenario_table


def to_latex_selected_country_with_row(
    batch_res: ScenarioBatchResult,
    round_digits: int = 2,
    legacy_compat: bool = True,
) -> str:
    """Render selected country impacts table with ROW and EUR as a LaTeX string.

    Replicates MATLAB Resultadosl.m:574-629 and selected_country_impacts_with_ROW.tex.
    ``legacy_compat=True`` reproduces that text only for the canonical 5-scenario
    batch; any other number of counterfactual scenarios always gets the clean
    layout with the ``& Base`` column.
    """
    targets = ["CAN", "CHN", "EUR", "MEX", "USA", "ROW"]
    tbl = generate_selected_country_table(
        batch_res, countries=targets, round_digits=round_digits
    )
    cf_scens = [col for col in tbl.columns if col != "Base"]
    n_scen = len(cf_scens)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    use_legacy = legacy_compat and (n_scen == 5)
    col_count = n_scen if use_legacy else (n_scen + 1)
    span = 7 if use_legacy else (col_count + 1)
    lines.append(r"\rowcolors{2}{gray!15}{white}")
    lines.append(r"\begin{tabular}{l" + "r" * col_count + "}")
    lines.append(r"\toprule")

    hdr_scens = " & ".join([s.replace("_", r"\_") for s in cf_scens])
    if use_legacy:
        lines.append(f"Country & {hdr_scens} \\\\")
    else:
        lines.append(f"Country & {hdr_scens} & Base \\\\")
    lines.append(r"\midrule")

    # Section 1: GDP growth (%)
    lines.append(r"  \multicolumn{" + str(span) + r"}{l}{\textbf{GDP growth (\%)}} \\")
    for c in targets:
        row_vals = [tbl.loc[("GDP growth (%)", c), s] for s in cf_scens]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        if not use_legacy:
            row_str += " &       "
        row_str += r" \\"
        lines.append(row_str)

    # Section 2: Inflation (%)
    lines.append(r"  \midrule")
    lines.append(r"  \multicolumn{" + str(span) + r"}{l}{\textbf{Inflation (\%)}} \\")
    for c in targets:
        row_vals = [tbl.loc[("Inflation (%)", c), s] for s in cf_scens]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        if not use_legacy:
            row_str += " &       "
        row_str += r" \\"
        lines.append(row_str)

    # Section 3: Net exports / GDP (%)
    lines.append(r"  \midrule")
    lines.append(r"  \multicolumn{" + str(span) + r"}{l}{\textbf{Net exports / GDP (\%)}} \\")
    for c in targets:
        row_vals = [tbl.loc[("Net exports / GDP (%)", c), s] for s in cf_scens]
        base_val = tbl.loc[("Net exports / GDP (%)", c), "Base"]
        row_str = f"    {c:<8s} "
        for v in row_vals:
            row_str += f" & {v:6.{round_digits}f}"
        row_str += f" & {base_val:6.{round_digits}f}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def export_latex_tables(
    batch_res: ScenarioBatchResult,
    output_dir: str | Path,
    prefix: str = "",
    legacy_compat: bool = True,
) -> dict[str, Path]:
    """Export all publication LaTeX tables to output directory.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved trade equilibrium results.
    output_dir : str | Path
        Directory path where .tex files will be written.
    prefix : str, default ''
        Optional filename prefix.
    legacy_compat : bool, default True
        Whether to enforce exact character parity against MATLAB reference output.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping table identifier to its Path on disk:
        {'selected_country_impacts', 'mean_by_scenario', 'weighted_mean_by_scenario',
         'selected_country_impacts_with_row'}.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1. selected_country_impacts.tex
    p_sel = out_dir / f"{prefix}selected_country_impacts.tex"
    p_sel.write_text(
        to_latex_selected_country(batch_res, legacy_compat=legacy_compat),
        encoding="utf-8",
    )
    paths["selected_country_impacts"] = p_sel

    # 2. mean_by_scenario.tex
    p_mean = out_dir / f"{prefix}mean_by_scenario.tex"
    p_mean.write_text(
        to_latex_mean_by_scenario(batch_res, legacy_compat=legacy_compat),
        encoding="utf-8",
    )
    paths["mean_by_scenario"] = p_mean

    # 3. weighted_mean_by_scenario.tex
    p_wmean = out_dir / f"{prefix}weighted_mean_by_scenario.tex"
    p_wmean.write_text(
        to_latex_weighted_mean_by_scenario(batch_res, legacy_compat=legacy_compat),
        encoding="utf-8",
    )
    paths["weighted_mean_by_scenario"] = p_wmean

    # 4. selected_country_impacts_with_ROW.tex
    p_row = out_dir / f"{prefix}selected_country_impacts_with_ROW.tex"
    p_row.write_text(
        to_latex_selected_country_with_row(batch_res, legacy_compat=legacy_compat),
        encoding="utf-8",
    )
    paths["selected_country_impacts_with_row"] = p_row

    return paths


__all__ = [
    "generate_selected_country_table",
    "generate_mean_by_scenario_table",
    "generate_weighted_mean_by_scenario_table",
    "to_latex_selected_country_table",
    "to_latex_selected_country",
    "to_latex_mean_by_scenario_table",
    "to_latex_mean_by_scenario",
    "to_latex_weighted_mean_by_scenario_table",
    "to_latex_weighted_mean_by_scenario",
    "to_latex_selected_country_with_row",
    "export_latex_tables",
]
