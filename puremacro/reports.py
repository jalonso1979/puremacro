"""Markdown / LaTeX / Typst output formatters for puremacro results.

Aimed at iPad researchers writing in Working Copy / Pages / Quarto:
turn a result dict (from any LP, VAR, GARCH, etc. function) into a
publication-ready table without leaving the notebook.

Also exposes light dataclass wrappers for the most common result
shapes so users get tab-completion in iPad IDEs and a uniform
``.to_markdown()`` method.

Every renderer here is what the result classes' ``to_markdown`` /
``to_latex`` / ``to_typst`` methods call, so the three guarantees below
hold package-wide:

* LaTeX special characters (``\\ & % $ # _ { } ~ ^``) are escaped in
  headers and cells, so a column named ``lo_95%`` compiles.
* Typst markup characters (``* _ # $ @ [ ] < > ~`` and the backslash) are
  escaped inside content cells, so ``0.452***`` stays literal.
* Floats are printed with at most six decimals unless ``digits`` is
  given, so ``0.30000000000000004`` never reaches a manuscript. An
  unnamed ``RangeIndex`` (pandas' default row numbering) is not
  rendered as a column; any other index is.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Escaping and cell formatting
# ---------------------------------------------------------------------------
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Characters that start markup inside a Typst content block ``[...]``.
_TYPST_SPECIALS = set("\\*_#$@[]<>~`")


def latex_escape(text: str) -> str:
    """Escape every LaTeX special character in ``text``."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def typst_escape(text: str) -> str:
    """Escape every character that Typst would read as markup in ``text``."""
    return "".join(("\\" + ch) if ch in _TYPST_SPECIALS else ch for ch in str(text))


def _fmt_cell(v: Any, digits: int | None = None) -> str:
    """One cell as text: NaN/None -> '', floats to ``digits`` decimals.

    With ``digits=None`` a float is printed positionally with at most six
    decimals (trailing zeros dropped), and values below 1e-4 in magnitude
    switch to ``%g`` so they are not rendered as ``0``.
    """
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if np.isnan(x):
            return ""
        if not np.isfinite(x):
            return str(x)
        if digits is not None:
            return f"{x:.{digits}f}"
        if x == 0.0:
            return "0"
        if abs(x) < 1e-4:
            return f"{x:.6g}"
        return np.format_float_positional(x, precision=6, unique=True, trim="-")
    if np.ndim(v) == 0:
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
    return str(v)


def _prepare(df: pd.DataFrame, index: bool | None) -> pd.DataFrame:
    """Return ``df`` with the index promoted to a column when it carries information.

    ``index=None`` (the default) keeps the index unless it is an unnamed
    ``RangeIndex`` — pandas' default row numbering, which would otherwise
    appear as a meaningless ``index`` column. ``True`` / ``False`` force
    either behaviour.
    """
    if index is None:
        index = not (isinstance(df.index, pd.RangeIndex) and df.index.name is None)
    return df.reset_index() if index else df


def _cells(df: pd.DataFrame, digits: int | None) -> tuple[list[str], list[list[str]]]:
    cols = [str(c) for c in df.columns]
    rows = [[_fmt_cell(v, digits) for v in row]
            for row in df.itertuples(index=False, name=None)]
    return cols, rows


def _df_to_markdown(df: pd.DataFrame, index: bool | None = None, *,
                    digits: int | None = None) -> str:
    """Self-contained markdown renderer (avoids the ``tabulate`` dependency).

    Supported internal API: result containers across puremacro (for example
    ``puremacro.trade.caliendo_parro`` and ``puremacro.models.trade_policy``)
    render their ``to_markdown``/``to_latex``/``to_typst`` tables with
    :func:`_df_to_markdown`, :func:`_df_to_latex` and :func:`_df_to_typst`;
    their signatures are kept stable for those callers.

    Parameters
    ----------
    index : bool, optional
        Include the index as the first column. Default: yes, unless it is
        an unnamed ``RangeIndex``.
    digits : int, optional
        Fixed number of decimals for float cells; default at most six.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    cols = [c.replace("|", r"\|") for c in cols]
    rows = [[c.replace("|", r"\|") for c in r] for r in rows]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]

    def fmt(vals: Sequence[str]) -> str:
        return "| " + " | ".join(v.rjust(w) for v, w in zip(vals, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(cols), sep]
    for r in rows:
        out.append(fmt(r))
    return "\n".join(out)


def _df_to_latex(df: pd.DataFrame, index: bool | None = None, *,
                 digits: int | None = None) -> str:
    """Self-contained LaTeX ``tabular`` renderer.

    Every header and cell is passed through :func:`latex_escape`, so
    ``%``, ``&``, ``_`` and the other specials compile as literals.
    Parameters as for :func:`_df_to_markdown`.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    align = "l" + "r" * max(len(cols) - 1, 0)
    header = " & ".join(latex_escape(c) for c in cols) + " \\\\\n"
    body = "".join(
        " & ".join(latex_escape(c) for c in r) + " \\\\\n" for r in rows
    )
    return (
        "\\begin{tabular}{" + align + "}\n"
        + header
        + "\\hline\n"
        + body
        + "\\end{tabular}"
    )


def _df_to_typst(df: pd.DataFrame, index: bool | None = None, *,
                 digits: int | None = None) -> str:
    """Self-contained Typst ``#table`` renderer.

    Cell text is passed through :func:`typst_escape`, so significance
    stars and underscores are literal rather than emphasis markup.
    Parameters as for :func:`_df_to_markdown`.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    n_cols = len(cols)
    header_cells = [f"  [* {typst_escape(c)} *]" for c in cols]
    data_cells = [f"  [{typst_escape(v)}]" for r in rows for v in r]
    all_cells = ",\n".join(header_cells + data_cells)
    return f"#table(\n  columns: {n_cols},\n{all_cells},\n)"


def _p_stars(p: float) -> str:
    """Conventional academic significance stars."""
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------
def coef_table(
    beta: np.ndarray,
    se: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    fmt: str = "markdown",
    digits: int = 3,
    include_t: bool = True,
    include_p: bool = True,
    include_ci: bool = True,
    alpha: float = 0.05,
    stars: bool = False,
) -> str:
    """Render a coefficient table.

    Parameters
    ----------
    beta : (k,) — point estimates.
    se   : (k,) — standard errors.
    names : optional regressor names; defaults to ``x0, x1, ...``.
    fmt  : ``"markdown"``, ``"latex"``, ``"typst"``, or ``"plain"`` (whitespace).
    digits : int, default 3
        Decimals shown for every numeric column.
    include_t, include_p, include_ci : bool
        Add the t statistic, the two-sided normal p-value and the
        ``lo_<level>%`` / ``hi_<level>%`` confidence bounds.
    alpha : float, default 0.05
        Size of the confidence interval (``1 - alpha`` coverage).
    stars : bool, default False
        If True, append significance stars (* p<0.10, ** p<0.05, *** p<0.01) to coef.

    Returns
    -------
    String table. In LaTeX and Typst every special character in the
    names and headers is escaped, so the output compiles as-is.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    se = np.asarray(se, dtype=float).ravel()
    k = len(beta)
    if names is None:
        names = [f"x{i}" for i in range(k)]
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for i in range(k):
        tt = beta[i] / se[i] if se[i] > 0 else np.nan
        p_val = 2.0 * norm.sf(abs(tt)) if not np.isnan(tt) else np.nan
        coef_str = f"{beta[i]:.{digits}f}"
        if stars:
            coef_str += _p_stars(p_val)
        row = {"variable": names[i],
               "coef": coef_str if stars else beta[i],
               "se": se[i]}
        if include_t:
            row["t"] = tt
        if include_p:
            row["p"] = p_val
        if include_ci:
            row[f"lo_{int((1-alpha)*100)}%"] = beta[i] - z * se[i]
            row[f"hi_{int((1-alpha)*100)}%"] = beta[i] + z * se[i]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variable")
    if not stars:
        df = df.round(digits)
    else:
        # round numeric columns only
        num_cols = [c for c in df.columns if c != "coef"]
        df[num_cols] = df[num_cols].round(digits)
    if fmt == "markdown":
        return _df_to_markdown(df, digits=digits)
    if fmt == "latex":
        return _df_to_latex(df, digits=digits)
    if fmt == "typst":
        return _df_to_typst(df, digits=digits)
    return df.to_string()


def irf_to_dataframe(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> pd.DataFrame:
    """Convert IRF matrices to formatted pandas DataFrame."""
    point = np.asarray(point, dtype=float)
    if point.ndim == 3:
        # If user passes (H+1, n_resp, n_shock), pick the first shock by default.
        point = point[:, :, 0]
        if lower is not None:
            lower = np.asarray(lower, dtype=float)[:, :, 0]
        if upper is not None:
            upper = np.asarray(upper, dtype=float)[:, :, 0]
    H_plus, n = point.shape
    if h_axis is None:
        h_axis = range(H_plus)
    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]

    rows = []
    for h_idx, h in enumerate(h_axis):
        row: Dict[str, Union[int, str]] = {"h": h}
        for i, name in enumerate(var_names):
            cell = f"{point[h_idx, i]:+.{digits}f}"
            if lower is not None and upper is not None:
                cell += (f"  [{lower[h_idx, i]:+.{digits}f}, "
                         f"{upper[h_idx, i]:+.{digits}f}]")
            row[name] = cell
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def irf_to_markdown(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a markdown table.

    If ``lower`` / ``upper`` are given, columns alternate point estimate
    and bracketed band, e.g. ``+0.123 [-0.045, +0.301]``.
    """
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_markdown(df)


def irf_to_latex(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a LaTeX tabular."""
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_latex(df)


def irf_to_typst(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a Typst table."""
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_typst(df)


def summary_to_dataframe(d: dict) -> pd.DataFrame:
    """Format flat result dictionary into a 2-column DataFrame."""
    rows = []
    for key, val in d.items():
        if isinstance(val, (int, float, np.floating, np.integer)):
            rows.append({"key": key, "value": f"{val:.4g}"})
        elif isinstance(val, str):
            rows.append({"key": key, "value": val})
        elif isinstance(val, np.ndarray):
            rows.append({"key": key, "value": f"ndarray{tuple(val.shape)}"})
        elif isinstance(val, (pd.Series, pd.DataFrame)):
            rows.append({"key": key, "value": f"{type(val).__name__}{tuple(val.shape)}"})
        elif isinstance(val, (list, tuple)):
            rows.append({"key": key, "value": f"{type(val).__name__} (len {len(val)})"})
        else:
            rows.append({"key": key, "value": str(val)[:40]})
    return pd.DataFrame(rows, columns=["key", "value"])


def summary_to_markdown(d: dict, *, title: str = "Result") -> str:
    """Quick markdown summary for a flat result dict.

    Numeric scalars become a key/value table. NumPy arrays and pandas
    objects are summarised by shape.
    """
    df = summary_to_dataframe(d)
    return f"### {title}\n\n" + _df_to_markdown(df, index=False)


def summary_to_latex(d: dict, *, title: str = "Result") -> str:
    """Quick LaTeX tabular summary for a flat result dict.

    The title is emitted as a ``%`` comment line above the table; keys
    and values are escaped, so ``share_%`` renders as ``share\\_\\%``.
    """
    df = summary_to_dataframe(d)
    return f"% {title}\n" + _df_to_latex(df, index=False)


def summary_to_typst(d: dict, *, title: str = "Result") -> str:
    """Quick Typst table summary for a flat result dict."""
    df = summary_to_dataframe(d)
    return f"// {title}\n" + _df_to_typst(df, index=False)


# ---------------------------------------------------------------------------
# Lightweight dataclass wrappers
# ---------------------------------------------------------------------------
@dataclass
class IRFResult:
    """Wraps a triple (point, lower, upper) of IRF arrays (H+1, n_resp, n_shock).

    ``to_markdown`` / ``to_latex`` / ``to_typst`` all render the same
    table (one column per response variable, first shock, bands in
    brackets) and all accept ``digits=`` and ``h_axis=``.
    """
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    var_names: tuple = ()
    method: str = ""
    alpha: float = 0.10

    def to_frame(self, shock_idx: int = 0) -> pd.DataFrame:
        point = np.asarray(self.point, dtype=float)
        if point.ndim == 3:
            p_slice = point[:, :, shock_idx]
            l_slice = self.lower[:, :, shock_idx] if self.lower is not None else None
            u_slice = self.upper[:, :, shock_idx] if self.upper is not None else None
        else:
            p_slice = point
            l_slice = self.lower
            u_slice = self.upper
        H_plus, n = p_slice.shape
        names = list(self.var_names) if self.var_names else [f"y{i}" for i in range(n)]
        data = {"h": np.arange(H_plus)}
        for i, name in enumerate(names):
            data[name] = p_slice[:, i]
            if l_slice is not None and u_slice is not None:
                data[f"{name}_lower"] = l_slice[:, i]
                data[f"{name}_upper"] = u_slice[:, i]
        return pd.DataFrame(data).set_index("h")

    def to_markdown(self, **kwargs) -> str:
        return irf_to_markdown(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def to_latex(self, **kwargs) -> str:
        return irf_to_latex(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def to_typst(self, **kwargs) -> str:
        return irf_to_typst(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def at(self, h: int):
        """Return a (n_resp, n_shock) slice at horizon h with bands."""
        return {
            "point": self.point[h],
            "lower": None if self.lower is None else self.lower[h],
            "upper": None if self.upper is None else self.upper[h],
        }


@dataclass
class LPResult:
    """Wraps the DataFrame returned by ``lp_hac``, ``panel_lp``, etc."""
    df: pd.DataFrame
    method: str = "lp"
    n_obs: int | None = None
    alpha: float = 0.10

    def __getitem__(self, key):
        return self.df[key]

    def __len__(self):
        return len(self.df)

    @property
    def horizons(self):
        return self.df.index if self.df.index.name == "h" else self.df["h"].values

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.df, **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.df, **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.df, **kwargs)


df_to_markdown = _df_to_markdown
df_to_latex = _df_to_latex
df_to_typst = _df_to_typst


def _extract_result_dataframe(result_or_df: Any) -> pd.DataFrame:
    if isinstance(result_or_df, pd.DataFrame):
        return result_or_df.copy()
    if isinstance(result_or_df, pd.Series):
        return result_or_df.to_frame()
    for attr in ("to_dataframe", "to_frame", "simulated_path", "df", "summary"):
        val = getattr(result_or_df, attr, None)
        if callable(val):
            try:
                out = val()
                if isinstance(out, pd.DataFrame):
                    return out.copy()
            except Exception:
                continue
        elif isinstance(val, pd.DataFrame):
            return val.copy()
    if isinstance(result_or_df, dict):
        return summary_to_dataframe(result_or_df)
    raise TypeError(f"Cannot extract DataFrame from object of type {type(result_or_df).__name__}")


def result_to_interactive_html(
    result_or_df: Any,
    *,
    title: str = "PureMacro Interactive Explorer",
    subtitle: str | None = None,
    audit_badge: str | None = None,
    digits: int = 4,
) -> str:
    """Render a self-contained, zero-server HTML5 + inline SVG explorer for any PureMacro result or DataFrame."""
    import html as _html
    import json as _json

    df = _extract_result_dataframe(result_or_df)
    if audit_badge is None:
        conv = getattr(result_or_df, "converged", None)
        if conv is True:
            audit_badge = "PASS"
        elif conv is False:
            audit_badge = "FAIL"

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        # Try coercing numeric columns if formatted as strings
        coerced = df.apply(pd.to_numeric, errors="coerce")
        num_df = coerced.dropna(axis=1, how="all")

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2"]
    n_rows = max(1, len(df))
    svg_w, svg_h = 760, 260
    pad_l, pad_r, pad_t, pad_b = 56, 24, 20, 32
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    polylines: list[str] = []
    checkboxes: list[str] = []
    if num_df.shape[1] > 0 and len(num_df) > 0:
        vals = num_df.to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        y_min = float(np.min(finite_vals)) if finite_vals.size > 0 else -1.0
        y_max = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if abs(y_max - y_min) < 1e-12:
            y_min -= 1.0
            y_max += 1.0

        def _y_px(v: float) -> float:
            return pad_t + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

        def _x_px(i: int) -> float:
            if len(num_df) <= 1:
                return pad_l + 0.5 * plot_w
            return pad_l + plot_w * (i / (len(num_df) - 1))

        if y_min <= 0.0 <= y_max:
            y0 = _y_px(0.0)
            polylines.append(
                f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{svg_w - pad_r}" y2="{y0:.1f}" '
                f'stroke="#888" stroke-dasharray="4,3" stroke-width="1" />'
            )

        for col_idx, col_name in enumerate(num_df.columns):
            color = palette[col_idx % len(palette)]
            safe_col = _html.escape(str(col_name))
            pts = []
            for r_i, v in enumerate(num_df[col_name].to_numpy(dtype=float)):
                if np.isfinite(v):
                    pts.append(f"{_x_px(r_i):.1f},{_y_px(float(v)):.1f}")
            pts_str = " ".join(pts)
            polylines.append(
                f'<polyline class="pm-series" data-series="{safe_col}" '
                f'fill="none" stroke="{color}" stroke-width="2.2" points="{pts_str}">'
                f"<title>{safe_col}</title></polyline>"
            )
            checkboxes.append(
                f'<label style="margin-right:12px;font-size:13px;cursor:pointer;">'
                f'<input type="checkbox" class="pm-toggle" data-series="{safe_col}" checked '
                f'onchange="pmToggleSeries(this)" /> '
                f'<span style="color:{color};font-weight:600;">■</span> {safe_col}</label>'
            )

    badge_html = ""
    if audit_badge:
        b_up = audit_badge.upper()
        b_col = "#1b7f37" if "PASS" in b_up else ("#b35900" if "WARN" in b_up else "#b91c1c")
        badge_html = (
            f'<span class="pm-audit-badge" style="background:{b_col};color:#fff;'
            f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">'
            f"{_html.escape(audit_badge)}</span>"
        )

    sub_html = f'<p style="margin:4px 0 0 0;color:#555;font-size:13px;">{_html.escape(subtitle)}</p>' if subtitle else ""

    df_prep = _prepare(df, None)
    cols, rows = _cells(df_prep, digits)
    th_html = "".join(
        f'<th data-col="{_html.escape(c)}" style="padding:6px 10px;border-bottom:2px solid #ccc;text-align:right;">'
        f"{_html.escape(c)}</th>"
        for c in cols
    )
    tr_list: list[str] = []
    for r_idx, r in enumerate(rows):
        tds = "".join(
            f'<td data-col="{_html.escape(cols[c_i])}" style="padding:5px 10px;border-bottom:1px solid #eee;text-align:right;">'
            f"{_html.escape(cell)}</td>"
            for c_i, cell in enumerate(r)
        )
        tr_list.append(f'<tr class="pm-row" data-row="{r_idx}">{tds}</tr>')

    payload_json = _json.dumps({"columns": cols, "row_count": len(rows)})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{_html.escape(title)}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:16px;color:#111;">
<div class="pm-explorer" style="max-width:860px;border:1px solid #ddd;border-radius:8px;padding:16px;background:#fff;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <h3 style="margin:0;font-size:18px;">{_html.escape(title)}</h3>
      {sub_html}
    </div>
    {badge_html}
  </div>
  <div class="pm-controls" style="margin-bottom:10px;padding:8px 10px;background:#f8f9fa;border-radius:6px;">
    <div style="margin-bottom:6px;">{"".join(checkboxes)}</div>
    <div style="font-size:12px;color:#444;">
      Horizon / Row Cutoff: <input type="range" id="pm-horizon-slider" min="1" max="{n_rows}" value="{n_rows}" oninput="pmFilterRows(this.value)" />
      <span id="pm-horizon-label">{n_rows} / {n_rows}</span>
    </div>
  </div>
  <svg class="pm-chart" viewBox="0 0 {svg_w} {svg_h}" style="width:100%;height:auto;border:1px solid #eee;border-radius:4px;background:#fcfcfc;margin-bottom:12px;">
    {"".join(polylines)}
  </svg>
  <div style="overflow-x:auto;max-height:320px;">
    <table class="pm-table" style="width:100%;border-collapse:collapse;font-size:12.5px;">
      <thead><tr>{th_html}</tr></thead>
      <tbody>{"".join(tr_list)}</tbody>
    </table>
  </div>
  <script type="application/json" id="pm-metadata">{payload_json}</script>
  <script>
  function pmToggleSeries(cb) {{
    var s = cb.getAttribute('data-series');
    var show = cb.checked;
    document.querySelectorAll('.pm-series[data-series="' + s + '"]').forEach(function(el) {{
      el.style.display = show ? '' : 'none';
    }});
    document.querySelectorAll('[data-col="' + s + '"]').forEach(function(el) {{
      el.style.display = show ? '' : 'none';
    }});
  }}
  function pmFilterRows(val) {{
    var maxRow = parseInt(val, 10);
    var lbl = document.getElementById('pm-horizon-label');
    if (lbl) lbl.textContent = maxRow + ' / {n_rows}';
    document.querySelectorAll('.pm-row').forEach(function(tr) {{
      var idx = parseInt(tr.getAttribute('data-row'), 10);
      tr.style.display = (idx < maxRow) ? '' : 'none';
    }});
  }}
  </script>
</div>
</body>
</html>"""


@dataclass(frozen=True)
class _ExtractedSeries:
    """Internal canonical single-series IRF representation prior to alignment."""

    label: str
    source_key: str
    group: str | None
    h: np.ndarray
    point: np.ndarray
    se: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ci_level: float | None
    band_kind: str
    is_cumulative: bool
    method: str
    n_obs: int | float | None = None
    vcov: np.ndarray | None = None
    supt_crit: float | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _detect_coef_labels(columns: Any) -> list[str]:
    cols = [str(c) for c in columns]
    if "beta" in cols:
        return [""]
    labels = [c[5:] for c in cols if c.startswith("beta_")]
    if labels:
        return labels
    if "point" in cols or "response" in cols:
        return [""]
    return []


def _format_tau(tau: float) -> str:
    if abs(tau * 100.0 - round(tau * 100.0)) < 1e-6:
        return f"tau={tau:.2f}"
    return f"tau={tau:g}"


def _resolve_selector(sel: Any, key: str) -> Any:
    if isinstance(sel, Mapping):
        return sel.get(key)
    return sel


def _resolve_axis_index(
    sel: str | int | None,
    names: Sequence[str] | None,
    size: int,
    axis_label: str,
    spec_key: str,
) -> tuple[int, str | None]:
    if sel is None:
        if size > 1:
            warn_msg = (
                f"{spec_key!r}: multi-variable IRF ({axis_label} size={size}) without explicit "
                f"{axis_label}=; defaulting to index 0."
            )
            return 0, warn_msg
        return 0, None
    if isinstance(sel, (int, np.integer)):
        idx = int(sel)
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError(
                f"{spec_key!r}: {axis_label} index {sel} out of range for size {size}."
            )
        return idx, None
    if names is not None and len(names) > 0:
        name_list = [str(n) for n in names]
        if str(sel) not in name_list:
            raise ValueError(
                f"{spec_key!r}: {axis_label} {sel!r} not found in available names {name_list}."
            )
        return name_list.index(str(sel)), None
    raise ValueError(
        f"{spec_key!r} has no var_names metadata; pass integer {axis_label} index instead of {sel!r}."
    )


def _extract_ndarray_pair(
    arr: np.ndarray | None,
    t_idx: int,
    s_idx: int,
) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a.copy()
    if a.ndim == 2:
        return a[:, t_idx].copy()
    if a.ndim == 3:
        return a[:, t_idx, s_idx].copy()
    raise ValueError(f"Unsupported IRF array rank ndim={a.ndim}; expected 1, 2, or 3.")


def _extract_array_triplet(
    key: str,
    pt_raw: np.ndarray,
    lo_raw: np.ndarray | None,
    hi_raw: np.ndarray | None,
    var_names: Sequence[str] | None,
    target: str | int | None,
    shock: str | int | None,
    warn_list: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    pt_arr = np.asarray(pt_raw, dtype=float)
    size_t = pt_arr.shape[1] if pt_arr.ndim >= 2 else 1
    size_s = pt_arr.shape[2] if pt_arr.ndim >= 3 else 1
    t_idx, w1 = _resolve_axis_index(target, var_names or None, size_t, "target", key)
    s_idx, w2 = _resolve_axis_index(shock, var_names or None, size_s, "shock", key)
    if pt_arr.ndim >= 2 and w1:
        warn_list.append(w1)
    if pt_arr.ndim >= 3 and w2:
        warn_list.append(w2)
    pt = _extract_ndarray_pair(pt_arr, t_idx, s_idx)
    assert pt is not None
    lo = (
        _extract_ndarray_pair(np.asarray(lo_raw, dtype=float), t_idx, s_idx)
        if lo_raw is not None
        else np.full(len(pt), np.nan, dtype=float)
    )
    hi = (
        _extract_ndarray_pair(np.asarray(hi_raw, dtype=float), t_idx, s_idx)
        if hi_raw is not None
        else np.full(len(pt), np.nan, dtype=float)
    )
    assert lo is not None and hi is not None
    return pt, lo, hi, t_idx, s_idx


def _scalar_crit_value(crit: Any, t_idx: int, s_idx: int) -> float | None:
    if crit is None:
        return None
    arr = np.asarray(crit, dtype=float)
    if arr.ndim == 2:
        return float(arr[t_idx, s_idx])
    if arr.ndim == 1:
        return float(arr[t_idx])
    return float(arr)


def _extract_from_lp_dataframe(
    df: pd.DataFrame,
    *,
    key: str,
    ci_level: float | None,
    method: str,
    n_obs_default: int | float | None,
    is_cumulative: bool,
    vcov: np.ndarray | None,
    band_kind_hint: str | None,
    supt_crit: float | None,
    extra_meta: dict[str, Any],
    spec_regime: str | None,
    spec_quantile: float | None,
    expand_groups: bool,
) -> list[_ExtractedSeries]:
    attrs = dict(getattr(df, "attrs", {}) or {})
    if supt_crit is None and "supt_crit" in attrs:
        supt_crit = float(attrs["supt_crit"])
    if band_kind_hint is None:
        band_kind_hint = attrs.get("band_kind")

    if "h" not in df.columns and df.index.name == "h":
        df = pd.DataFrame(df.reset_index())
    elif "h" not in df.columns:
        raise ValueError(f"{key!r}: LP/IRF DataFrame is missing required 'h' column.")
    else:
        df = pd.DataFrame(df.reset_index(drop=True))

    def _finish_rows(
        sub: pd.DataFrame,
        *,
        label: str,
        group: str | None,
        pt_col: str,
        se_col: str | None,
        lo_col: str | None,
        hi_col: str | None,
        warn_list: list[str],
    ) -> _ExtractedSeries:
        ordered = sub.sort_values("h") if len(sub) > 1 else sub
        if ordered["h"].duplicated().any():
            raise ValueError(
                f"{key!r}: duplicate horizons {ordered['h'].tolist()} found for single series "
                f"{label!r}; pass an explicit group selector or deduplicate rows."
            )
        h_arr = np.asarray(ordered["h"], dtype=int)
        pt_arr = np.asarray(ordered[pt_col], dtype=float)
        if se_col and se_col in ordered.columns:
            se_arr = np.asarray(ordered[se_col], dtype=float)
        else:
            se_arr = np.full(len(ordered), np.nan, dtype=float)
        if lo_col and lo_col in ordered.columns and hi_col and hi_col in ordered.columns:
            lo_arr = np.asarray(ordered[lo_col], dtype=float)
            hi_arr = np.asarray(ordered[hi_col], dtype=float)
        else:
            lo_arr = np.full(len(ordered), np.nan, dtype=float)
            hi_arr = np.full(len(ordered), np.nan, dtype=float)

        has_bands = bool(np.any(np.isfinite(lo_arr)) and np.any(np.isfinite(hi_arr)))
        if not has_bands:
            bk = "none"
            resolved_ci = None
        else:
            bk = str(band_kind_hint) if band_kind_hint else ("sup-t" if supt_crit is not None else "pointwise")
            resolved_ci = ci_level

        n_obs_val: int | float | None = n_obs_default
        if "n_obs" in ordered.columns:
            finite_n = pd.to_numeric(ordered["n_obs"], errors="coerce").dropna()
            if not finite_n.empty:
                n_obs_val = int(finite_n.iloc[0])

        return _ExtractedSeries(
            label=label,
            source_key=key,
            group=group,
            h=h_arr,
            point=pt_arr,
            se=se_arr,
            lower=lo_arr,
            upper=hi_arr,
            ci_level=resolved_ci,
            band_kind=bk,
            is_cumulative=is_cumulative,
            method=method,
            n_obs=n_obs_val,
            vcov=vcov,
            supt_crit=supt_crit,
            extra_meta=dict(extra_meta),
            warnings=warn_list,
        )

    # Quantile LP layout: stacked rows by tau
    if "tau" in df.columns and "beta" in df.columns:
        taus = sorted(float(t) for t in pd.Series(df["tau"]).dropna().unique().tolist())
        if not taus:
            raise ValueError(f"{key!r}: 'tau' column contains no finite quantiles.")
        if spec_quantile is not None:
            matches = [t for t in taus if abs(t - float(spec_quantile)) <= 1e-6]
            if not matches:
                raise ValueError(f"{key!r}: requested quantile={spec_quantile} not found in {taus}.")
            chosen_taus = [matches[0]]
            suffix_mode = False
        elif expand_groups:
            chosen_taus = taus
            suffix_mode = True
        elif len(taus) == 1:
            chosen_taus = taus
            suffix_mode = False
        else:
            med = [t for t in taus if abs(t - 0.50) <= 1e-6]
            if med:
                chosen_taus = [med[0]]
                suffix_mode = False
                auto_warn = (
                    f"{key!r}: multiple quantiles {taus} present without quantile= or "
                    "expand_groups=True; defaulting to median tau=0.50."
                )
            else:
                raise ValueError(
                    f"{key!r} has multiple quantiles {taus} without median tau=0.50; "
                    "pass quantile=... or expand_groups=True."
                )

        out: list[_ExtractedSeries] = []
        for idx_t, t_val in enumerate(chosen_taus):
            mask = np.isclose(pd.to_numeric(df["tau"], errors="coerce").to_numpy(dtype=float), t_val, atol=1e-6)
            sub = df.loc[mask]
            tag = _format_tau(t_val)
            lbl = f"{key} [{tag}]" if suffix_mode else key
            w = [auto_warn] if ("auto_warn" in locals() and idx_t == 0) else []
            out.append(
                _finish_rows(
                    sub,
                    label=lbl,
                    group=tag,
                    pt_col="beta",
                    se_col="se" if "se" in df.columns else None,
                    lo_col="lo" if "lo" in df.columns else ("lower" if "lower" in df.columns else None),
                    hi_col="hi" if "hi" in df.columns else ("upper" if "upper" in df.columns else None),
                    warn_list=w,
                )
            )
        return out

    coef_labels = _detect_coef_labels(df.columns)
    if not coef_labels:
        raise ValueError(f"{key!r}: could not locate beta/point/response columns in {list(df.columns)}.")

    # Single-coefficient or generic IRF dataframe
    if coef_labels == [""]:
        if "beta" in df.columns:
            pt_col = "beta"
            se_col = "se" if "se" in df.columns else None
            lo_col = "lo" if "lo" in df.columns else ("lower" if "lower" in df.columns else None)
            hi_col = "hi" if "hi" in df.columns else ("upper" if "upper" in df.columns else None)
        elif "point" in df.columns:
            pt_col = "point"
            se_col = "se" if "se" in df.columns else None
            lo_col = "lo" if "lo" in df.columns else ("lower" if "lower" in df.columns else None)
            hi_col = "hi" if "hi" in df.columns else ("upper" if "upper" in df.columns else None)
        else:
            pt_col = "response"
            se_col = None
            lo_col = "lo" if "lo" in df.columns else ("lower" if "lower" in df.columns else None)
            hi_col = "hi" if "hi" in df.columns else ("upper" if "upper" in df.columns else None)
        return [
            _finish_rows(
                df,
                label=key,
                group=None,
                pt_col=pt_col,
                se_col=se_col,
                lo_col=lo_col,
                hi_col=hi_col,
                warn_list=[],
            )
        ]

    # Multi-regime / grouped LPResult (beta_H, beta_L, beta_pos, beta_neg, ...)
    if spec_regime is not None:
        if spec_regime not in coef_labels:
            raise ValueError(f"{key!r}: regime={spec_regime!r} not found in available groups {coef_labels}.")
        chosen_groups = [spec_regime]
        suffix_mode = False
    elif expand_groups:
        chosen_groups = coef_labels
        suffix_mode = True
    elif len(coef_labels) == 1:
        chosen_groups = coef_labels
        suffix_mode = False
    else:
        raise ValueError(
            f"{key!r} contains multiple groups/regimes {coef_labels}; pass regime=... or expand_groups=True."
        )

    out_groups: list[_ExtractedSeries] = []
    for grp in chosen_groups:
        lbl = f"{key} [{grp}]" if suffix_mode else key
        out_groups.append(
            _finish_rows(
                df,
                label=lbl,
                group=grp,
                pt_col=f"beta_{grp}",
                se_col=f"se_{grp}" if f"se_{grp}" in df.columns else None,
                lo_col=f"lo_{grp}" if f"lo_{grp}" in df.columns else None,
                hi_col=f"hi_{grp}" if f"hi_{grp}" in df.columns else None,
                warn_list=[],
            )
        )
    return out_groups


def _extract_irf_series(
    key: str,
    obj: Any,
    *,
    target: str | int | None,
    shock: str | int | None,
    regime: str | None,
    quantile: float | None,
    ci: float | None,
    expand_groups: bool,
) -> tuple[list[_ExtractedSeries], bool]:
    """Normalize supported LP/VAR/DSGE/report IRF containers into canonical series."""
    warn_list: list[str] = []
    used_ci_selector = False

    # 1. LPDiDResult
    if hasattr(obj, "estimates") and hasattr(obj, "n_events") and hasattr(obj, "pre_window"):
        est = obj.estimates
        alpha = getattr(obj, "alpha", None)
        ci_lvl = (1.0 - float(alpha)) if alpha is not None else getattr(est, "ci_level", None)
        weights = getattr(obj, "weights", "equal")
        n_units = getattr(obj, "n_units", None)
        n_events = getattr(obj, "n_events", None)
        meta = {
            "n_units": n_units,
            "n_events": n_events,
            "weights": weights,
            "cluster": getattr(obj, "cluster", None),
        }
        return _extract_from_lp_dataframe(
            est,
            key=key,
            ci_level=ci_lvl,
            method=f"lp_did({weights})",
            n_obs_default=n_units,
            is_cumulative=bool(getattr(est, "is_cumulative", False)),
            vcov=getattr(est, "vcov", None),
            band_kind_hint=None,
            supt_crit=None,
            extra_meta=meta,
            spec_regime=regime,
            spec_quantile=quantile,
            expand_groups=expand_groups,
        ), used_ci_selector

    # 2. puremacro.reports.LPResult (dataclass wrapper around df)
    if not isinstance(obj, pd.DataFrame) and hasattr(obj, "df") and isinstance(getattr(obj, "df", None), pd.DataFrame) and hasattr(obj, "method"):
        alpha = getattr(obj, "alpha", None)
        ci_lvl = (1.0 - float(alpha)) if alpha is not None else None
        return _extract_from_lp_dataframe(
            obj.df,
            key=key,
            ci_level=ci_lvl,
            method=str(getattr(obj, "method", "LP")),
            n_obs_default=getattr(obj, "n_obs", None),
            is_cumulative=bool(getattr(obj, "is_cumulative", False)),
            vcov=getattr(obj, "vcov", None),
            band_kind_hint=None,
            supt_crit=None,
            extra_meta={},
            spec_regime=regime,
            spec_quantile=quantile,
            expand_groups=expand_groups,
        ), used_ci_selector

    # 3. puremacro.lp.LPResult / SmoothLPResult / plain pd.DataFrame
    if isinstance(obj, pd.DataFrame):
        ci_lvl = getattr(obj, "ci_level", None)
        if ci_lvl is None and getattr(obj, "alpha", None) is not None:
            ci_lvl = 1.0 - float(obj.alpha)
        spec_dict = getattr(obj, "_spec", None)
        if ci_lvl is None and isinstance(spec_dict, Mapping) and "ci_level" in spec_dict:
            ci_lvl = spec_dict["ci_level"]
        n_obs_default: int | float | None = getattr(obj, "n_obs", None)
        if n_obs_default is None and isinstance(spec_dict, Mapping):
            n_by_h = spec_dict.get("n_obs_by_horizon")
            if isinstance(n_by_h, Mapping) and n_by_h:
                n_obs_default = min(int(v) for v in n_by_h.values())
        method_val = getattr(obj, "method", None)
        if not method_val and isinstance(spec_dict, Mapping):
            method_val = spec_dict.get("method") or spec_dict.get("estimator")
        return _extract_from_lp_dataframe(
            obj,
            key=key,
            ci_level=float(ci_lvl) if ci_lvl is not None else None,
            method=str(method_val or "LP"),
            n_obs_default=n_obs_default,
            is_cumulative=bool(getattr(obj, "is_cumulative", False)),
            vcov=getattr(obj, "vcov", None),
            band_kind_hint=None,
            supt_crit=None,
            extra_meta={},
            spec_regime=regime,
            spec_quantile=quantile,
            expand_groups=expand_groups,
        ), used_ci_selector

    # 4. SupTBandResult
    if hasattr(obj, "center") and hasattr(obj, "crit_value") and hasattr(obj, "lower") and hasattr(obj, "upper"):
        center_arr = np.asarray(obj.center, dtype=float)
        var_names_opt = [str(v) for v in (getattr(obj, "var_names", ()) or ())]
        pt, lo, hi, t_idx, s_idx = _extract_array_triplet(
            key,
            center_arr,
            np.asarray(obj.lower, dtype=float),
            np.asarray(obj.upper, dtype=float),
            var_names_opt,
            target,
            shock,
            warn_list,
        )
        scale_val = getattr(obj, "scale", None)
        se_arr = _extract_ndarray_pair(np.asarray(scale_val, dtype=float), t_idx, s_idx) if scale_val is not None else None
        if se_arr is None:
            se_arr = np.full(len(pt), np.nan, dtype=float)
        alpha = getattr(obj, "alpha", None)
        crit_f = _scalar_crit_value(getattr(obj, "crit_value", None), t_idx, s_idx)
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=np.arange(len(pt), dtype=int),
                point=pt,
                se=se_arr,
                lower=lo,
                upper=hi,
                ci_level=(1.0 - float(alpha)) if alpha is not None else None,
                band_kind="sup-t",
                is_cumulative=False,
                method=str(getattr(obj, "method", "sup-t")),
                supt_crit=crit_f,
                warnings=warn_list,
            )
        ], used_ci_selector

    # 5. BayesianIRFResult
    if hasattr(obj, "median") and hasattr(obj, "bands") and hasattr(obj, "variables") and hasattr(obj, "shocks"):
        shocks = list(getattr(obj, "shocks", ()) or ())
        variables = list(getattr(obj, "variables", ()) or ())
        med_obj = obj.median
        if isinstance(med_obj, Mapping):
            shock_keys = list(med_obj.keys())
            if shock is None:
                if len(shock_keys) == 1:
                    shock_name = shock_keys[0]
                else:
                    raise ValueError(
                        f"{key!r} is a BayesianIRFResult with multiple shocks {shock_keys}; pass shock=..."
                    )
            elif isinstance(shock, (int, np.integer)):
                idx_s = int(shock)
                shock_name = shocks[idx_s] if shocks else shock_keys[idx_s]
            else:
                shock_name = str(shock)
            if shock_name not in med_obj:
                raise ValueError(f"{key!r}: shock {shock_name!r} not in {shock_keys}.")
            med_df = med_obj[shock_name]
        else:
            med_df = med_obj
            if shock is None:
                shock_name = shocks[0] if shocks else ""
            elif isinstance(shock, (int, np.integer)):
                shock_name = shocks[int(shock)]
            else:
                shock_name = str(shock)

        cols = [str(c) for c in med_df.columns]
        if target is None:
            if variables and str(variables[0]) in cols:
                t_col = str(variables[0])
            else:
                t_col = cols[0]
            if len(cols) > 1:
                warn_list.append(
                    f"{key!r}: multi-variable BayesianIRFResult without target=; defaulting to {t_col!r}."
                )
        elif isinstance(target, (int, np.integer)):
            t_col = cols[int(target)]
        else:
            t_col = str(target)
            if t_col not in cols:
                raise ValueError(f"{key!r}: target {t_col!r} not in {cols}.")

        pt = np.asarray(med_df[t_col], dtype=float)
        if np.issubdtype(med_df.index.dtype, np.integer):
            h_arr = np.asarray(med_df.index, dtype=int)
        else:
            h_arr = np.arange(len(pt), dtype=int)

        raw_bands = obj.bands or {}
        if isinstance(raw_bands, Mapping) and shock_name in raw_bands and isinstance(raw_bands[shock_name], Mapping):
            band_map = raw_bands[shock_name]
        else:
            band_map = raw_bands

        avail_cis: list[tuple[float, Any]] = []
        for k_b in band_map.keys():
            try:
                avail_cis.append((float(k_b), k_b))
            except (TypeError, ValueError):
                continue

        lo_arr = np.full(len(pt), np.nan, dtype=float)
        hi_arr = np.full(len(pt), np.nan, dtype=float)
        resolved_ci: float | None = None
        band_kind = "none"

        if avail_cis:
            if ci is not None:
                used_ci_selector = True
                matched = [pair for pair in avail_cis if abs(pair[0] - float(ci)) <= 1e-4]
                if not matched:
                    raise ValueError(
                        f"{key!r}: requested credible band ci={ci} not in available bands {[p[0] for p in avail_cis]}."
                    )
                chosen_ci, chosen_key = matched[0]
            else:
                if ci is None and len(avail_cis) > 0:
                    used_ci_selector = False
                default_90 = [pair for pair in avail_cis if abs(pair[0] - 0.90) <= 1e-4]
                chosen_ci, chosen_key = default_90[0] if default_90 else sorted(avail_cis, key=lambda x: x[0])[-1]
            lo_df_or_map, hi_df_or_map = band_map[chosen_key]
            lo_df = lo_df_or_map[shock_name] if isinstance(lo_df_or_map, Mapping) else lo_df_or_map
            hi_df = hi_df_or_map[shock_name] if isinstance(hi_df_or_map, Mapping) else hi_df_or_map
            lo_arr = np.asarray(lo_df[t_col], dtype=float)
            hi_arr = np.asarray(hi_df[t_col], dtype=float)
            resolved_ci = chosen_ci
            band_kind = "credible"
        elif ci is not None:
            raise ValueError(f"{key!r}: BayesianIRFResult has no stored credible bands for ci={ci}.")

        det_rate = getattr(obj, "determinacy_rate", None)
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=h_arr,
                point=pt,
                se=np.full(len(pt), np.nan, dtype=float),
                lower=lo_arr,
                upper=hi_arr,
                ci_level=resolved_ci,
                band_kind=band_kind,
                is_cumulative=False,
                method="Bayesian-IRF",
                extra_meta={"determinacy_rate": det_rate} if det_rate is not None else {},
                warnings=warn_list,
            )
        ], used_ci_selector

    # 6. SVARResult (has ndarray .irf and .var_names; note VarEstimateResult has callable .irf())
    if hasattr(obj, "irf") and not callable(getattr(obj, "irf", None)) and hasattr(obj, "var_names"):
        irf_arr = np.asarray(obj.irf, dtype=float)
        var_names = [str(v) for v in (getattr(obj, "var_names", ()) or ())]
        ci_lo_raw = getattr(obj, "ci_lower", None)
        ci_hi_raw = getattr(obj, "ci_upper", None)
        gk_lo_raw = getattr(obj, "gk_lower", None)
        gk_hi_raw = getattr(obj, "gk_upper", None)
        has_gk = gk_lo_raw is not None and gk_hi_raw is not None
        if ci_lo_raw is not None and ci_hi_raw is not None:
            lo_raw, hi_raw, bk = np.asarray(ci_lo_raw), np.asarray(ci_hi_raw), "pointwise"
        elif has_gk:
            lo_raw, hi_raw, bk = np.asarray(gk_lo_raw), np.asarray(gk_hi_raw), "gk"
        else:
            lo_raw, hi_raw, bk = None, None, "none"
        pt, lo, hi, _, _ = _extract_array_triplet(
            key, irf_arr, lo_raw, hi_raw, var_names, target, shock, warn_list
        )
        meta = getattr(obj, "metadata", None) or {}
        alpha = meta.get("alpha") if isinstance(meta, Mapping) else None
        ci_lvl = (1.0 - float(alpha)) if (alpha is not None and bk != "none") else None
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=np.arange(len(pt), dtype=int),
                point=pt,
                se=np.full(len(pt), np.nan, dtype=float),
                lower=lo,
                upper=hi,
                ci_level=ci_lvl,
                band_kind=bk,
                is_cumulative=False,
                method=str(getattr(obj, "method", "SVAR")),
                extra_meta={"has_gk_bands": has_gk},
                warnings=warn_list,
            )
        ], used_ci_selector

    # 7. ProxySVARResult / attribute objects with irf_point
    if hasattr(obj, "irf_point"):
        pt_raw = np.asarray(obj.irf_point, dtype=float)
        var_names = [str(v) for v in (getattr(obj, "var_names", ()) or ())]
        lo_raw = getattr(obj, "irf_lower", None)
        hi_raw = getattr(obj, "irf_upper", None)
        pt, lo, hi, _, _ = _extract_array_triplet(
            key, pt_raw, lo_raw, hi_raw, var_names, target, shock, warn_list
        )
        has_b = lo_raw is not None and hi_raw is not None
        ci_attr = getattr(obj, "ci", None)
        alpha_attr = getattr(obj, "alpha", None)
        ci_lvl = (
            float(ci_attr) if (ci_attr is not None and has_b)
            else ((1.0 - float(alpha_attr)) if (alpha_attr is not None and has_b) else None)
        )
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=np.arange(len(pt), dtype=int),
                point=pt,
                se=np.full(len(pt), np.nan, dtype=float),
                lower=lo,
                upper=hi,
                ci_level=ci_lvl,
                band_kind="pointwise" if has_b else "none",
                is_cumulative=False,
                method=str(getattr(obj, "method", type(obj).__name__)),
                warnings=warn_list,
            )
        ], used_ci_selector

    # 8. puremacro.reports.IRFResult (has point, lower, upper, var_names, alpha)
    if hasattr(obj, "point") and hasattr(obj, "alpha") and not isinstance(obj, Mapping):
        pt_raw = np.asarray(obj.point, dtype=float)
        var_names = [str(v) for v in (getattr(obj, "var_names", ()) or ())]
        lo_raw = getattr(obj, "lower", None)
        hi_raw = getattr(obj, "upper", None)
        pt, lo, hi, _, _ = _extract_array_triplet(
            key, pt_raw, lo_raw, hi_raw, var_names, target, shock, warn_list
        )
        has_b = bool(np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)))
        alpha = getattr(obj, "alpha", None)
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=np.arange(len(pt), dtype=int),
                point=pt,
                se=np.full(len(pt), np.nan, dtype=float),
                lower=lo,
                upper=hi,
                ci_level=(1.0 - float(alpha)) if (alpha is not None and has_b) else None,
                band_kind="pointwise" if has_b else "none",
                is_cumulative=False,
                method=str(getattr(obj, "method", "IRF")),
                warnings=warn_list,
            )
        ], used_ci_selector

    # 9. VarEstimateResult (has .to_frame(target, shock), .Sigma, and .names/.var_names)
    if (
        callable(getattr(obj, "to_frame", None))
        and hasattr(obj, "Sigma")
        and (hasattr(obj, "var_names") or hasattr(obj, "names"))
    ):
        raw_vnames = getattr(obj, "var_names", None) or getattr(obj, "names", None)
        var_names = [str(v) for v in raw_vnames] if raw_vnames else []
        n_vars = len(var_names) if var_names else (
            int(obj.Sigma.shape[0]) if hasattr(obj, "Sigma") else 1
        )
        t_idx, w1 = _resolve_axis_index(target, var_names or None, n_vars, "target", key)
        s_idx, w2 = _resolve_axis_index(shock, var_names or None, n_vars, "shock", key)
        if w1:
            warn_list.append(w1)
        if w2:
            warn_list.append(w2)
        t_arg: str | int = var_names[t_idx] if var_names else t_idx
        s_arg: str | int = var_names[s_idx] if var_names else s_idx
        df_v = obj.to_frame(t_arg, s_arg)
        res_list = _extract_from_lp_dataframe(
            df_v,
            key=key,
            ci_level=None,
            method=str(getattr(obj, "method", "VAR")),
            n_obs_default=getattr(obj, "n_obs", None),
            is_cumulative=False,
            vcov=None,
            band_kind_hint="none",
            supt_crit=None,
            extra_meta={},
            spec_regime=None,
            spec_quantile=None,
            expand_groups=False,
        )
        res_list[0].warnings.extend(warn_list)
        return res_list, used_ci_selector

    # 10. Mapping dicts (bootstrap_bands / irf_point dicts)
    if isinstance(obj, Mapping):
        pt_key = "point" if "point" in obj else ("irf_point" if "irf_point" in obj else None)
        if pt_key is None:
            raise ValueError(
                f"{key!r}: dict IRF must contain 'point' (or 'irf_point'); got keys {list(obj.keys())}."
            )
        pt_raw = np.asarray(obj[pt_key], dtype=float)
        var_names = [str(v) for v in obj.get("var_names", ())]
        lo_key = "lower" if "lower" in obj else ("irf_lower" if "irf_lower" in obj else None)
        hi_key = "upper" if "upper" in obj else ("irf_upper" if "irf_upper" in obj else None)
        lo_raw = np.asarray(obj[lo_key]) if (lo_key and obj.get(lo_key) is not None) else None
        hi_raw = np.asarray(obj[hi_key]) if (hi_key and obj.get(hi_key) is not None) else None
        pt, lo, hi, t_idx, s_idx = _extract_array_triplet(
            key, pt_raw, lo_raw, hi_raw, var_names, target, shock, warn_list
        )
        if lo_raw is not None and hi_raw is not None:
            crit = obj.get("crit_value")
            bk = str(obj.get("band") or ("sup-t" if crit is not None else "pointwise"))
            if "alpha" in obj and obj["alpha"] is not None:
                ci_lvl = 1.0 - float(obj["alpha"])
            elif "ci" in obj and obj["ci"] is not None:
                ci_lvl = float(obj["ci"])
            else:
                ci_lvl = None
            crit_f = _scalar_crit_value(crit, t_idx, s_idx)
        else:
            bk = "none"
            ci_lvl = None
            crit_f = None

        h_arr = np.asarray(obj["h"], dtype=int) if "h" in obj else np.arange(len(pt), dtype=int)
        return [
            _ExtractedSeries(
                label=key,
                source_key=key,
                group=None,
                h=h_arr,
                point=pt,
                se=np.full(len(pt), np.nan, dtype=float),
                lower=lo,
                upper=hi,
                ci_level=ci_lvl,
                band_kind=bk,
                is_cumulative=False,
                method=str(obj.get("method", f"dict({bk})")),
                supt_crit=crit_f,
                warnings=warn_list,
            )
        ], used_ci_selector

    raise TypeError(
        f"{key!r}: unsupported IRF container type {type(obj).__name__}."
    )


def _cumulate_series(series: _ExtractedSeries, warn_list: list[str]) -> _ExtractedSeries:
    if series.is_cumulative:
        raise ValueError(
            f"{series.label!r} is already cumulative; cumulative=True cannot cumulate it twice."
        )
    if len(series.h) == 0:
        raise ValueError(f"{series.label!r}: cannot compute cumulative IRF on empty horizons.")
    order = np.argsort(series.h, kind="mergesort")
    h_sorted = series.h[order]
    if len(h_sorted) > 1 and not np.array_equal(np.diff(h_sorted), np.ones(len(h_sorted) - 1, dtype=int)):
        raise ValueError(
            f"{series.label!r} has non-consecutive horizons {h_sorted.tolist()}; "
            "cumulative=True requires consecutive unit-step horizons (step=1)."
        )

    pt_sorted = series.point[order]
    cum_pt = np.cumsum(pt_sorted)
    H = len(h_sorted)

    if series.vcov is not None:
        vcov_arr = np.asarray(series.vcov, dtype=float)
        if vcov_arr.shape == (H, H):
            vcov_ord = vcov_arr[np.ix_(order, order)]
            tril = np.tril(np.ones((H, H), dtype=float))
            cum_cov = tril @ vcov_ord @ tril.T
            cum_se = np.sqrt(np.maximum(np.diag(cum_cov), 0.0))
            ci_use = series.ci_level if series.ci_level is not None else 0.90
            z = float(norm.ppf(1.0 - (1.0 - ci_use) / 2.0))
            cum_lo = cum_pt - z * cum_se
            cum_hi = cum_pt + z * cum_se
            return _ExtractedSeries(
                label=series.label,
                source_key=series.source_key,
                group=series.group,
                h=h_sorted,
                point=cum_pt,
                se=cum_se,
                lower=cum_lo,
                upper=cum_hi,
                ci_level=series.ci_level,
                band_kind=series.band_kind,
                is_cumulative=True,
                method=series.method,
                n_obs=series.n_obs,
                vcov=cum_cov,
                supt_crit=series.supt_crit,
                extra_meta=dict(series.extra_meta),
                warnings=list(series.warnings),
            )

    warn_msg = (
        f"{series.label!r}: cumulative SEs/bands unavailable without cross-horizon vcov; "
        "omitting cumulative interval bounds."
    )
    warn_list.append(warn_msg)
    warnings.warn(warn_msg, UserWarning, stacklevel=3)
    nan_arr = np.full(H, np.nan, dtype=float)
    return _ExtractedSeries(
        label=series.label,
        source_key=series.source_key,
        group=series.group,
        h=h_sorted,
        point=cum_pt,
        se=nan_arr.copy(),
        lower=nan_arr.copy(),
        upper=nan_arr.copy(),
        ci_level=None,
        band_kind="none",
        is_cumulative=True,
        method=series.method,
        n_obs=series.n_obs,
        vcov=None,
        supt_crit=series.supt_crit,
        extra_meta=dict(series.extra_meta),
        warnings=list(series.warnings),
    )


def _slice_series_to_horizons(series: _ExtractedSeries, target_h: Sequence[int]) -> _ExtractedSeries:
    pos_by_h = {int(h_val): i for i, h_val in enumerate(series.h)}
    idx = np.array([pos_by_h[int(hv)] for hv in target_h], dtype=int)
    vcov_slice = (
        series.vcov[np.ix_(idx, idx)]
        if (series.vcov is not None and series.vcov.shape == (len(series.h), len(series.h)))
        else None
    )
    return _ExtractedSeries(
        label=series.label,
        source_key=series.source_key,
        group=series.group,
        h=series.h[idx],
        point=series.point[idx],
        se=series.se[idx],
        lower=series.lower[idx],
        upper=series.upper[idx],
        ci_level=series.ci_level,
        band_kind=series.band_kind,
        is_cumulative=series.is_cumulative,
        method=series.method,
        n_obs=series.n_obs,
        vcov=vcov_slice,
        supt_crit=series.supt_crit,
        extra_meta=dict(series.extra_meta),
        warnings=list(series.warnings),
    )


@dataclass(frozen=True)
class IRFComparisonResult:
    """Aligned multi-specification impulse-response comparison result.

    Attributes
    ----------
    table : pd.DataFrame
        Aligned long-format horizon comparison table with columns
        ``[spec, h, point, se, lower, upper, diff_vs_baseline, ci_level,
        band_kind, is_cumulative, method]``.
    metrics : pd.DataFrame
        Per-specification summary metrics (peak horizon and response, cumulative
        sum, max absolute diff & RMSE vs baseline, significance count, sign
        agreement, interval metadata, and sample size).
    baseline : str
        Reference specification label used for ``diff_vs_baseline`` and metrics.
    horizons : tuple[int, ...]
        Aligned integer horizon grid shared across all compared specifications.
    warnings : tuple[str, ...]
        Informational and compatibility warnings emitted during comparison.
    errors : dict[str, str]
        Mapping of skipped specification labels to failure messages (e.g. from
        :func:`puremacro.lp.lp_sensitivity_grid` with ``on_error="skip"``).
    digits : int
        Default rounding precision for table export methods.
    """

    table: pd.DataFrame
    metrics: pd.DataFrame
    baseline: str
    horizons: tuple[int, ...]
    warnings: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    digits: int = 3

    def to_frame(
        self,
        layout: Literal["wide", "coef_se", "metrics"] = "wide",
        digits: int | None = None,
    ) -> pd.DataFrame:
        """Return comparison table as a pandas DataFrame in the requested layout."""
        d = self.digits if digits is None else int(digits)
        if layout == "metrics":
            return self.metrics.copy()

        specs = list(dict.fromkeys(self.table["spec"].tolist()))
        if layout == "wide":
            out = pd.DataFrame({"h": list(self.horizons)})
            for spec in specs:
                sub = self.table[self.table["spec"] == spec].set_index("h").reindex(list(self.horizons))
                out[f"{spec}_point"] = sub["point"].to_numpy(dtype=float)
                out[f"{spec}_se"] = sub["se"].to_numpy(dtype=float)
                out[f"{spec}_lower"] = sub["lower"].to_numpy(dtype=float)
                out[f"{spec}_upper"] = sub["upper"].to_numpy(dtype=float)
            return out

        if layout == "coef_se":
            by_spec = {
                spec: self.table[self.table["spec"] == spec].set_index("h").reindex(list(self.horizons))
                for spec in specs
            }
            rows: list[dict[str, str]] = []
            for h_val in self.horizons:
                row_pt: dict[str, str] = {"h": str(int(h_val))}
                row_se: dict[str, str] = {"h": ""}
                for spec in specs:
                    rec = by_spec[spec].loc[h_val]
                    pt = float(rec["point"])
                    se_val = float(rec["se"])
                    lo_val = float(rec["lower"])
                    hi_val = float(rec["upper"])
                    if np.isfinite(pt) and np.isfinite(se_val) and se_val > 0.0:
                        p_val = float(2.0 * norm.sf(abs(pt / se_val)))
                        stars = _p_stars(p_val)
                        row_pt[spec] = f"{pt:.{d}f}{stars}"
                        row_se[spec] = f"({se_val:.{d}f})"
                    elif np.isfinite(pt) and np.isfinite(lo_val) and np.isfinite(hi_val):
                        row_pt[spec] = f"{pt:.{d}f}"
                        row_se[spec] = f"[{lo_val:.{d}f}, {hi_val:.{d}f}]"
                    elif np.isfinite(pt):
                        row_pt[spec] = f"{pt:.{d}f}"
                        row_se[spec] = ""
                    else:
                        row_pt[spec] = ""
                        row_se[spec] = ""
                rows.append(row_pt)
                rows.append(row_se)
            return pd.DataFrame(rows)

        raise ValueError(
            f"Unknown layout {layout!r}; expected one of 'wide', 'coef_se', 'metrics'."
        )

    def to_dataframe(
        self,
        layout: Literal["wide", "coef_se", "metrics"] = "wide",
        digits: int | None = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout, digits=digits)

    def to_manuscript_dataframe(self, digits: int | None = None) -> pd.DataFrame:
        """Return stacked AER/JME-style ``coef / (se)`` manuscript DataFrame."""
        return self.to_frame(layout="coef_se", digits=digits)

    def summary(self) -> str:
        """Return a concise multi-line text summary of the comparison."""
        h_span = f"[{min(self.horizons)}..{max(self.horizons)}]" if self.horizons else "[]"
        lines = [
            f"IRF Comparison ({len(self.metrics)} specs, baseline={self.baseline!r}, horizons={h_span})",
            self.metrics.to_string(index=False),
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.errors:
            lines.append(f"Skipped errors ({len(self.errors)}):")
            lines.extend(f"  - {k}: {v}" for k, v in self.errors.items())
        return "\n".join(lines)

    def to_markdown(
        self,
        layout: Literal["wide", "coef_se", "metrics"] = "coef_se",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as a GitHub-Flavored Markdown table."""
        d = self.digits if digits is None else int(digits)
        return _df_to_markdown(self.to_frame(layout=layout, digits=d), digits=d)

    def to_latex(
        self,
        layout: Literal["wide", "coef_se", "metrics"] = "coef_se",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as a publication LaTeX ``tabular``."""
        d = self.digits if digits is None else int(digits)
        return _df_to_latex(self.to_frame(layout=layout, digits=d), digits=d)

    def to_typst(
        self,
        layout: Literal["wide", "coef_se", "metrics"] = "coef_se",
        digits: int | None = None,
    ) -> str:
        """Render the comparison table as a Typst ``#table(...)``."""
        d = self.digits if digits is None else int(digits)
        return _df_to_typst(self.to_frame(layout=layout, digits=d), digits=d)

    def plot(
        self,
        show_bands: bool = True,
        ax: Axes | None = None,
        *,
        title: str = "IRF Comparison",
        ylabel: str = "Response",
    ) -> Figure | Axes:
        """Overlay compared impulse responses using lazy Matplotlib plotting.

        Returns the Matplotlib ``Figure`` when ``ax is None``, or ``ax`` when an
        existing ``Axes`` is provided.
        """
        from puremacro.plot import plot_irf_multi

        specs = list(dict.fromkeys(self.table["spec"].tolist()))
        frames: dict[str, pd.DataFrame] = {}
        for spec in specs:
            sub = self.table[self.table["spec"] == spec].sort_values("h")
            df_s = pd.DataFrame(
                {
                    "h": sub["h"].to_numpy(dtype=int),
                    "beta": sub["point"].to_numpy(dtype=float),
                }
            )
            has_finite_bands = (
                bool(sub["lower"].notna().any())
                and bool(sub["upper"].notna().any())
            )
            if show_bands and has_finite_bands:
                df_s["lo"] = sub["lower"].to_numpy(dtype=float)
                df_s["hi"] = sub["upper"].to_numpy(dtype=float)
            frames[spec] = df_s

        fig = plot_irf_multi(
            frames,
            title=title,
            ylabel=ylabel,
            show_bands=show_bands,
            ax=ax,
        )
        if ax is not None:
            return ax
        return fig


def compare_irfs(
    results: Mapping[str, Any],
    *,
    baseline: str | None = None,
    horizon_align: Literal["exact", "intersection"] = "exact",
    target: str | int | Mapping[str, str | int] | None = None,
    shock: str | int | Mapping[str, str | int] | None = None,
    regime: str | Mapping[str, str] | None = None,
    quantile: float | Mapping[str, float] | None = None,
    ci: float | None = None,
    expand_groups: bool = False,
    cumulative: bool = False,
    allow_mixed_cumulative: bool = False,
    band_compat: Literal["warn", "raise", "ignore"] = "warn",
    digits: int = 3,
) -> IRFComparisonResult:
    """Compare impulse responses across LP, VAR, SVAR, and DSGE estimators.

    Parameters
    ----------
    results : Mapping[str, Any]
        Ordered mapping from specification label to IRF result container
        (``LPResult``, ``SmoothLPResult``, ``LPDiDResult``, ``SVARResult``,
        ``BayesianIRFResult``, ``VarEstimateResult``, ``SupTBandResult``,
        ``reports.IRFResult``, ``reports.LPResult``, or ``bootstrap_bands`` dict).
    baseline : str, optional
        Reference specification label for difference and RMSE metrics. Defaults
        to the first extracted specification.
    horizon_align : {"exact", "intersection"}, default "exact"
        ``"exact"`` rejects mismatched horizons with an informative error listing
        each specification's horizon range; ``"intersection"`` restricts all
        series to their shared horizon intersection.
    target, shock, regime, quantile : selector or per-spec mapping, optional
        Selectors for 3-D/multi-variable IRFs and multi-regime/quantile LPs.
        Per-spec mappings override selectively; single-coefficient LPs ignore
        ``regime`` and ``quantile`` gracefully when mixed with multi-group LPs.
    ci : float, optional
        Credible interval level to select on multi-band objects such as
        :class:`~puremacro.dsge._results.BayesianIRFResult`. Emits a warning if
        supplied when no multi-band object is present.
    expand_groups : bool, default False
        When ``True``, expands multi-regime or multi-quantile results into
        separate labeled entries (e.g. ``Spec [H]``, ``Spec [L]``).
    cumulative : bool, default False
        Accumulate non-cumulative IRFs over consecutive unit-step horizons.
        Uses exact cross-horizon ``vcov`` when available (``SmoothLPResult``)
        and omits cumulative SEs with a warning otherwise.
    allow_mixed_cumulative : bool, default False
        Allow mixing already-cumulative IRFs with level IRFs when ``cumulative=False``.
    band_compat : {"warn", "raise", "ignore"}, default "warn"
        How to handle mismatched ``ci_level`` or ``band_kind`` across series.
    digits : int, default 3
        Default decimal places for table formatting methods.
    """
    if not isinstance(results, Mapping) or len(results) == 0:
        raise ValueError("results must be a non-empty mapping of label -> IRF result.")
    if horizon_align not in ("exact", "intersection"):
        raise ValueError(f"Unknown horizon_align={horizon_align!r}; expected 'exact' or 'intersection'.")
    if band_compat not in ("warn", "raise", "ignore"):
        raise ValueError(f"Unknown band_compat={band_compat!r}; expected 'warn', 'raise', or 'ignore'.")

    extracted: list[_ExtractedSeries] = []
    warn_list: list[str] = []
    any_ci_selector_used = False

    for raw_key, obj in results.items():
        key = str(raw_key)
        series_batch, used_ci = _extract_irf_series(
            key,
            obj,
            target=_resolve_selector(target, key),
            shock=_resolve_selector(shock, key),
            regime=_resolve_selector(regime, key),
            quantile=_resolve_selector(quantile, key),
            ci=ci,
            expand_groups=expand_groups,
        )
        any_ci_selector_used = any_ci_selector_used or used_ci
        for s in series_batch:
            if s.warnings:
                for w in s.warnings:
                    warn_list.append(w)
                    warnings.warn(w, UserWarning, stacklevel=2)
            extracted.append(s)

    if ci is not None and not any_ci_selector_used:
        ci_msg = (
            f"ci={ci} only selects stored credible bands on multi-band objects "
            "(such as BayesianIRFResult); frequentist LP/VAR intervals were kept as estimated."
        )
        warn_list.append(ci_msg)
        warnings.warn(ci_msg, UserWarning, stacklevel=2)

    cum_flags = [s.is_cumulative for s in extracted]
    if any(cum_flags) and not all(cum_flags) and not allow_mixed_cumulative and not cumulative:
        raise ValueError(
            "Cannot mix already-cumulative IRFs with non-cumulative IRFs; "
            "pass allow_mixed_cumulative=True or compare consistent transformations."
        )

    if cumulative:
        extracted = [_cumulate_series(s, warn_list) for s in extracted]

    # Horizon alignment
    h_sets = [set(int(v) for v in s.h.tolist()) for s in extracted]
    common_h = sorted(set.intersection(*h_sets)) if h_sets else []
    all_equal_h = all(hs == h_sets[0] for hs in h_sets)
    if horizon_align == "exact" and not all_equal_h:
        spans = ", ".join(
            f"{s.label}: [{int(np.min(s.h))}..{int(np.max(s.h))}] (n={len(s.h)})"
            if len(s.h) > 0 else f"{s.label}: []"
            for s in extracted
        )
        raise ValueError(
            f"Horizons do not match across specs ({spans}). "
            "Pass horizon_align='intersection' to align onto common horizons."
        )
    if not common_h:
        raise ValueError("No common horizons exist across specifications.")
    if horizon_align == "intersection" and not all_equal_h:
        align_msg = (
            f"Aligned mismatched horizons to common intersection [{common_h[0]}..{common_h[-1]}] "
            f"({len(common_h)} horizons)."
        )
        warn_list.append(align_msg)
        warnings.warn(align_msg, UserWarning, stacklevel=2)

    aligned = [_slice_series_to_horizons(s, common_h) for s in extracted]
    labels = [s.label for s in aligned]

    # Band compatibility audit
    sigs = {
        (None if s.ci_level is None else round(float(s.ci_level), 4), s.band_kind)
        for s in aligned
    }
    if len(sigs) > 1:
        desc = ", ".join(
            f"{s.label}: {s.band_kind}@{f'{s.ci_level:.0%}' if s.ci_level is not None else 'NA'}"
            for s in aligned
        )
        compat_msg = f"Confidence bands differ across compared specs ({desc})."
        if band_compat == "raise":
            raise ValueError(compat_msg)
        if band_compat == "warn":
            warn_list.append(compat_msg)
            warnings.warn(compat_msg, UserWarning, stacklevel=2)

    # Resolve baseline label
    if baseline is None:
        base_label = labels[0]
    elif baseline in labels:
        base_label = baseline
    else:
        matching_expanded = [s.label for s in aligned if s.source_key == baseline]
        if matching_expanded:
            base_label = matching_expanded[0]
        else:
            raise ValueError(f"Unknown baseline {baseline!r}; available specs: {labels}.")

    by_label = {s.label: s for s in aligned}
    base_series = by_label[base_label]

    table_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for s in aligned:
        diff = s.point - base_series.point
        for i, h_val in enumerate(common_h):
            table_rows.append(
                {
                    "spec": s.label,
                    "h": int(h_val),
                    "point": float(s.point[i]),
                    "se": float(s.se[i]),
                    "lower": float(s.lower[i]),
                    "upper": float(s.upper[i]),
                    "diff_vs_baseline": float(diff[i]),
                    "ci_level": float(s.ci_level) if s.ci_level is not None else np.nan,
                    "band_kind": s.band_kind,
                    "is_cumulative": bool(s.is_cumulative),
                    "method": s.method,
                }
            )

        finite_pt = np.isfinite(s.point)
        if np.any(finite_pt):
            masked_abs = np.where(finite_pt, np.abs(s.point), -np.inf)
            peak_pos = int(np.argmax(masked_abs))
            peak_h = int(s.h[peak_pos])
            peak_resp = float(s.point[peak_pos])
            cum_sum = float(np.nansum(s.point))
        else:
            peak_h = int(common_h[0])
            peak_resp = np.nan
            cum_sum = np.nan

        finite_diff = diff[np.isfinite(diff)]
        max_abs_diff = float(np.max(np.abs(finite_diff))) if finite_diff.size else np.nan
        rmse = float(np.sqrt(np.mean(finite_diff ** 2))) if finite_diff.size else np.nan

        band_valid = np.isfinite(s.lower) & np.isfinite(s.upper)
        n_sig = int(np.sum((s.lower[band_valid] > 0.0) | (s.upper[band_valid] < 0.0)))

        nz_mask = np.isfinite(base_series.point) & np.isfinite(s.point) & (np.abs(base_series.point) > 1e-12)
        if np.any(nz_mask):
            sign_agree = float(np.mean(np.sign(s.point[nz_mask]) == np.sign(base_series.point[nz_mask])))
        else:
            sign_agree = 1.0 if s.label == base_label else np.nan

        metric_rows.append(
            {
                "spec": s.label,
                "baseline": s.label == base_label,
                "peak_h": peak_h,
                "peak_response": peak_resp,
                "cum_sum": cum_sum,
                "max_abs_diff_vs_base": max_abs_diff,
                "rmse_vs_base": rmse,
                "n_sig": n_sig,
                "sign_agreement": sign_agree,
                "ci_level": float(s.ci_level) if s.ci_level is not None else np.nan,
                "band_kind": s.band_kind,
                "is_cumulative": bool(s.is_cumulative),
                "method": s.method,
                "n_obs": s.n_obs if s.n_obs is not None else np.nan,
            }
        )

    return IRFComparisonResult(
        table=pd.DataFrame(table_rows),
        metrics=pd.DataFrame(metric_rows),
        baseline=base_label,
        horizons=tuple(int(h_v) for h_v in common_h),
        warnings=tuple(dict.fromkeys(warn_list)),
        errors={},
        digits=int(digits),
    )


__all__ = [
    "coef_table",
    "compare_irfs",
    "df_to_latex",
    "df_to_markdown",
    "df_to_typst",
    "irf_to_dataframe",
    "irf_to_latex",
    "irf_to_markdown",
    "irf_to_typst",
    "latex_escape",
    "typst_escape",
    "summary_to_dataframe",
    "summary_to_latex",
    "summary_to_markdown",
    "summary_to_typst",
    "result_to_interactive_html",
    "IRFComparisonResult",
    "IRFResult",
    "LPResult",
]
