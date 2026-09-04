"""Markdown / LaTeX output formatters for puremacro results.

Aimed at iPad researchers writing in Working Copy / Pages / Quarto:
turn a result dict (from any LP, VAR, GARCH, etc. function) into a
publication-ready table without leaving the notebook.

Also exposes light dataclass wrappers for the most common result
shapes so users get tab-completion in iPad IDEs and a uniform
``.to_markdown()`` method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


def _df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Self-contained markdown renderer (avoids the ``tabulate`` dependency)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = list(df.columns.astype(str))
    rows = [[("" if pd.isna(v) else str(v)) for v in df[c]] for c in cols]
    rows = list(map(list, zip(*rows)))  # transpose to row-major
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    fmt = lambda vals: "| " + " | ".join(
        v.rjust(w) for v, w in zip(vals, widths)
    ) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(cols), sep]
    for r in rows:
        out.append(fmt(r))
    return "\n".join(out)


def _df_to_latex(df: pd.DataFrame, index: bool = True) -> str:
    """Self-contained LaTeX tabular renderer."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = list(df.columns.astype(str))
    align = "l" + "r" * (len(cols) - 1)
    rows: List[List[str]] = list(map(list, zip(*[
        [("" if pd.isna(v) else str(v).replace("_", r"\_")) for v in df[c]]
        for c in cols
    ])))
    body = " \\\\\n".join(" & ".join(r) for r in rows) + " \\\\"
    return (
        "\\begin{tabular}{" + align + "}\n"
        + " & ".join(c.replace("_", r"\_") for c in cols) + " \\\\\n"
        + "\\hline\n"
        + body + "\n"
        + "\\end{tabular}"
    )


def _df_to_typst(df: pd.DataFrame, index: bool = True) -> str:
    """Self-contained Typst table renderer."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = list(df.columns.astype(str))
    n_cols = len(cols)
    header_cells = [f"  [* {c} *]" for c in cols]
    data_cells = []
    for _, row in df.iterrows():
        for v in row:
            val_str = "" if pd.isna(v) else str(v)
            data_cells.append(f"  [{val_str}]")
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
    stars : bool, default False
        If True, append significance stars (* p<0.10, ** p<0.05, *** p<0.01) to coef.

    Returns
    -------
    String table.
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
        return _df_to_markdown(df)
    if fmt == "latex":
        return _df_to_latex(df)
    if fmt == "typst":
        return _df_to_typst(df)
    return df.to_string()


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
    point = np.asarray(point, dtype=float)
    if point.ndim == 3:
        # If user passes (H+1, n_resp, n_shock), pick the first shock by default.
        point = point[:, :, 0]
        if lower is not None:
            lower = lower[:, :, 0]
        if upper is not None:
            upper = upper[:, :, 0]
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
    return _df_to_markdown(pd.DataFrame(rows).set_index("h"))


def summary_to_markdown(d: dict, *, title: str = "Result") -> str:
    """Quick markdown summary for a flat result dict.

    Numeric scalars become a key/value table. NumPy arrays and pandas
    objects are summarised by shape.
    """
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
    return f"### {title}\n\n" + _df_to_markdown(pd.DataFrame(rows), index=False)


# ---------------------------------------------------------------------------
# Lightweight dataclass wrappers
# ---------------------------------------------------------------------------
@dataclass
class IRFResult:
    """Wraps a triple (point, lower, upper) of IRF arrays (H+1, n_resp, n_shock)."""
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    var_names: tuple = ()
    method: str = ""
    alpha: float = 0.10

    def to_markdown(self, **kwargs) -> str:
        return irf_to_markdown(
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

    def to_markdown(self) -> str:
        return _df_to_markdown(self.df)

    def to_latex(self) -> str:
        return _df_to_latex(self.df)

    def to_typst(self) -> str:
        return _df_to_typst(self.df)


__all__ = [
    "coef_table", "irf_to_markdown", "summary_to_markdown",
    "IRFResult", "LPResult",
]
