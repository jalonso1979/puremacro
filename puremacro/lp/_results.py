"""Result classes for puremacro.lp."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


# Suffix-free column set of a single-coefficient result and the suffixed
# layout of regime / sign results (``beta_H``/``beta_L``, ``beta_pos``/
# ``beta_neg``, ``beta_high``/``beta_low``).
_BASES = ("beta", "se", "lo", "hi")


def _coef_labels(columns: Any) -> list[str]:
    """Coefficient labels of a result frame.

    ``[""]`` for a plain single-coefficient frame (``beta``/``se``/...);
    otherwise the suffixes ``s`` of every ``beta_s`` column in column
    order (e.g. ``["H", "L"]``, ``["pos", "neg"]``).  Empty when the frame
    carries no coefficient columns at all.
    """
    cols = [str(c) for c in columns]
    if "beta" in cols:
        return [""]
    return [c[len("beta_"):] for c in cols
            if c.startswith("beta_") and len(c) > len("beta_")]


def _col(base: str, label: str) -> str:
    return base if label == "" else f"{base}_{label}"


def _sig_flag(beta: float, se: float, lo: float, hi: float) -> str:
    """Significance marker: stars from the normal z-test when an SE is
    available, otherwise a single ``*`` when the reported band excludes 0."""
    if np.isfinite(se) and se > 0 and np.isfinite(beta):
        z = abs(beta / se)
        if z > 2.5758:
            return "***"
        if z > 1.9600:
            return "**"
        if z > 1.6449:
            return "*"
        return ""
    if np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
        return "*"
    return ""


class LPResult(pd.DataFrame):
    """Result of local projection estimators.

    Subclasses :class:`pandas.DataFrame` so that all existing DataFrame operations
    (column access, `.loc`, `.iloc`, slicing) continue to work seamlessly, while
    providing convenient properties (`.point`, `.se_arr`, `.ci_lower`, `.ci_upper`),
    plotting (`.plot()`), and formatted summary tables (`.summary()`).

    Two column layouts are supported:

    * single-coefficient results (``lp_hac``, ``lp_iv``, ``panel_lp``, ...):
      columns ``h, beta, se, lo, hi`` (plus estimator-specific extras);
      ``.point``/``.se``/``.ci_lower``/``.ci_upper`` return 1-D arrays;
    * regime or sign results (``lp_state_dep``, ``lp_state_dep_iv``,
      ``lp_asymmetric``, ``lp_garch_state``): columns ``beta_<label>``,
      ``se_<label>``, ``lo_<label>``, ``hi_<label>`` for each label
      (``H``/``L``, ``pos``/``neg``); the same properties then return a
      :class:`pandas.DataFrame` indexed by ``h`` with one column per label
      (see :attr:`labels`).

    ``lp_quantile`` reports bootstrap bands without an analytical SE: its
    ``.se`` is all-NaN and ``.t_stat`` is NaN by construction.
    """

    _metadata = [
        "y_name", "x_name", "method", "ci_level", "is_cumulative", "_spec",
        # la_lp / la_lp_iv lag specification (also mirrored in ``attrs``)
        "n_lags", "extra_lags", "control_lags",
        # smooth_lp estimation metadata (must survive pandas operations)
        "optimal_lambda", "df_lambda", "theta", "vcov", "B", "P", "lambda_grid",
        "selection_criterion", "ci_type", "n_knots", "degree", "penalty_order", "gls",
    ]

    is_cumulative: bool = False

    @property
    def _constructor(self):
        return LPResult

    def __finalize__(self, other: Any, method: str | None = None, **kwargs: Any) -> LPResult:
        super().__finalize__(other, method=method, **kwargs)
        spec = getattr(self, "_spec", None)
        if isinstance(spec, dict) and spec and "specification" not in self.attrs:
            self.attrs["specification"] = dict(spec)
        return self

    def _set_specification(self, spec: dict[str, Any]) -> None:
        """Attach reconstructible specification metadata to ``_spec`` and ``attrs``."""
        clean = dict(spec)
        object.__setattr__(self, "_spec", clean)
        self.attrs["specification"] = dict(clean)

    # ------------------------------------------------------------------
    # Column layout helpers
    # ------------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        """Coefficient labels: ``[]`` for a single-coefficient result,
        otherwise e.g. ``['H', 'L']`` or ``['pos', 'neg']``."""
        labs = _coef_labels(self.columns)
        return [] if labs == [""] else labs

    def _stack(self, base: str, missing: str = "raise") -> Any:
        """Return column ``base`` (1-D array) or the ``base_<label>``
        columns as a DataFrame indexed by ``h``.

        ``missing`` controls what happens when the column is absent:
        ``"raise"`` -> :class:`KeyError`; ``"none"`` -> ``None``;
        ``"nan"`` -> an all-NaN array/frame of the right shape.
        """
        labs = _coef_labels(self.columns)
        if not labs:
            if missing == "none":
                return None
            if missing == "nan":
                return np.full(len(self), np.nan)
            raise KeyError(
                "LPResult has no coefficient columns (expected 'beta' or "
                f"'beta_<label>'); columns are {list(self.columns)}")
        if labs == [""]:
            if base in self.columns:
                return self[base].to_numpy()
            if missing == "none":
                return None
            if missing == "nan":
                return np.full(len(self), np.nan)
            raise KeyError(f"LPResult has no {base!r} column")
        present = {lab: _col(base, lab) for lab in labs
                   if _col(base, lab) in self.columns}
        if not present:
            if missing == "none":
                return None
            if missing == "nan":
                return pd.DataFrame(
                    {lab: np.full(len(self), np.nan) for lab in labs},
                    index=pd.Index(self.horizons, name="h"))
            raise KeyError(f"LPResult has no {base}_<label> columns")
        return pd.DataFrame(
            {lab: self[c].to_numpy() for lab, c in present.items()},
            index=pd.Index(self.horizons, name="h"))

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------
    @property
    def point(self) -> Any:
        """Point estimates β_h: 1-D array, or a DataFrame with one column
        per regime/sign label for multi-coefficient results."""
        return self._stack("beta")

    @property
    def se(self) -> Any:
        """Standard errors (alias for se_arr). NaN when the estimator
        reports bands only (``lp_quantile``)."""
        return self._stack("se", missing="nan")

    @property
    def se_arr(self) -> Any:
        """Standard errors across horizons."""
        return self.se

    @property
    def ci_lower(self) -> Any:
        """Lower confidence band (``None`` when the result has no bands)."""
        return self._stack("lo", missing="none")

    @property
    def ci_upper(self) -> Any:
        """Upper confidence band (``None`` when the result has no bands)."""
        return self._stack("hi", missing="none")

    @property
    def t_stat(self) -> Any:
        """t-statistics β_h / se_h (NaN where no positive SE is available)."""
        labs = _coef_labels(self.columns)
        if labs == [""] and "t" in self.columns:
            return self["t"].to_numpy()
        point = self.point
        se = self.se
        with np.errstate(divide="ignore", invalid="ignore"):
            if isinstance(point, pd.DataFrame):
                out = point.copy()
                for lab in point.columns:
                    s = se[lab].to_numpy() if lab in se.columns else np.full(len(point), np.nan)
                    out[lab] = np.where(s > 0, point[lab].to_numpy() / s, np.nan)
                return out
            return np.where(se > 0, point / se, np.nan)

    @property
    def horizons(self) -> np.ndarray:
        """Array of horizon values."""
        if "h" in self.columns:
            return self["h"].to_numpy()
        return np.asarray(self.index)

    @property
    def frame(self) -> pd.DataFrame:
        """Return as a standard pandas DataFrame."""
        return pd.DataFrame(self)

    @property
    def metadata(self) -> dict[str, Any]:
        """Estimation metadata and specification provenance dictionary."""
        base: dict[str, Any] = {
            "y_name": getattr(self, "y_name", None),
            "x_name": getattr(self, "x_name", None),
            "method": getattr(self, "method", "LP"),
        }
        ci_level = getattr(self, "ci_level", None)
        if ci_level is not None:
            base["ci_level"] = float(ci_level)
        spec = getattr(self, "_spec", None) or self.attrs.get("specification")
        if isinstance(spec, dict) and spec:
            merged = dict(spec)
            merged.update({k: v for k, v in base.items() if v is not None})
            merged["specification"] = dict(spec)
            return merged
        return base

    def to_frame(self) -> pd.DataFrame:
        """Convert to a standard pandas.DataFrame."""
        return pd.DataFrame(self)

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def plot(
        self,
        *,
        title: str = "",
        ylabel: str = "Response",
        scale: float = 1.0,
        ax=None,
    ):
        """Plot the impulse response(s) with error bands.

        Single-coefficient results draw one line with a shaded band;
        regime/sign results draw one line and band per label with a
        legend; ``lp_quantile`` results draw one line per quantile.
        Lazily delegates to :func:`puremacro.plot.plot_irf_single`.
        """
        from ..plot import plot_irf_single

        return plot_irf_single(self, title=title, ylabel=ylabel, scale=scale, ax=ax)

    def _weak_iv_footer(self) -> str | None:
        """Build a weak-IV diagnostic warning line when first-stage F falls below threshold."""
        if "mop_f" in self.columns and "mop_cv_10" in self.columns:
            mop_vals = pd.to_numeric(self["mop_f"], errors="coerce")
            cv_vals = pd.to_numeric(self["mop_cv_10"], errors="coerce")
            mask = mop_vals.notna() & cv_vals.notna() & (mop_vals < cv_vals)
            if bool(mask.any()):
                h_vals = self.loc[mask, "h"].astype(int).tolist() if "h" in self.columns else list(self.index[mask])
                min_f = float(mop_vals[mask].min())
                cv_ref = float(cv_vals[mask].iloc[0])
                h_str = ", ".join(str(h) for h in h_vals)
                if "ar_lo" in self.columns:
                    return (
                        f"Weak-IV warning: MOP effective F < 10% critical value ({cv_ref:.2f}) "
                        f"at h=[{h_str}] (min F={min_f:.2f}); use ar_lo/ar_hi/ar_set_type "
                        "rather than normal Wald bands."
                    )
                return (
                    f"Weak-IV warning: MOP effective F < 10% critical value ({cv_ref:.2f}) "
                    f"at h=[{h_str}] (min F={min_f:.2f}); re-run with anderson_rubin=True "
                    "for weak-IV-robust confidence sets."
                )
        elif "first_stage_f" in self.columns:
            f_vals = pd.to_numeric(self["first_stage_f"], errors="coerce")
            mask = f_vals.notna() & (f_vals < 10.0)
            if bool(mask.any()):
                h_vals = self.loc[mask, "h"].astype(int).tolist() if "h" in self.columns else list(self.index[mask])
                min_f = float(f_vals[mask].min())
                h_str = ", ".join(str(h) for h in h_vals)
                return (
                    f"Weak-IV warning: first-stage F < 10 at h=[{h_str}] "
                    f"(min F={min_f:.2f}); normal 2SLS Wald bands may under-cover."
                )
        return None

    def summary(self) -> str:
        """Formatted text summary of the estimates actually present.

        One row per horizon (and per regime / sign label or quantile
        ``tau`` for multi-coefficient results) with ``beta``, ``se``,
        the confidence band and a significance flag: ``***`` p<0.01,
        ``**`` p<0.05, ``*`` p<0.10 from the two-sided normal z-test; for
        band-only estimators (``lp_quantile``) a single ``*`` marks bands
        that exclude zero.
        """
        method = getattr(self, "method", "LP")
        labs = _coef_labels(self.columns)
        multi = labs not in ([""], [])
        ci_level = getattr(self, "ci_level", None)
        spec = getattr(self, "_spec", None) or self.attrs.get("specification")
        head = f"Local Projection Result (method: {method})"
        info = []
        y_name = getattr(self, "y_name", None)
        x_name = getattr(self, "x_name", None)
        if y_name is not None and x_name is not None:
            info.append(f"y: {y_name}   x: {x_name}")
        if ci_level is not None:
            info.append(f"bands: {100 * float(ci_level):.0f}%")
        if multi:
            info.append("coefficients: " + ", ".join(labs))
        if isinstance(spec, dict) and spec:
            if "n_lags" in spec and spec["n_lags"] is not None:
                info.append(f"lags: {int(spec['n_lags'])}")
            ctls = spec.get("controls") or []
            if ctls:
                if len(ctls) <= 3:
                    info.append("controls: " + ", ".join(str(c) for c in ctls))
                else:
                    shown = ", ".join(str(c) for c in ctls[:3])
                    info.append(f"controls: {len(ctls)} ({shown}, ...)")
            cov_desc = spec.get("cov_summary")
            if cov_desc:
                info.append(f"SE: {cov_desc}")
            n_map = spec.get("n_obs_by_horizon")
            if isinstance(n_map, dict) and n_map:
                if "h" in self.columns:
                    active_hs = [int(h) for h in self["h"].dropna().tolist()]
                    counts = [int(n_map[h]) for h in active_hs if h in n_map and int(n_map[h]) > 0]
                else:
                    counts = [int(v) for v in n_map.values() if int(v) > 0]
                if counts:
                    lo_n, hi_n = min(counts), max(counts)
                    info.append(f"N: {lo_n}" if lo_n == hi_n else f"N: {lo_n}..{hi_n}")
        lines = [head]
        if info:
            lines.append("   ".join(info))
        if self.is_cumulative:
            lines.append("estimand: cumulative multiplier (sum of y over sum of x, horizons 0..h)")

        if not labs:
            lines.append(f"(no coefficient columns; columns: {list(self.columns)})")
            return "\n".join(lines)

        id_cols = [c for c in ("tau", "regime") if c in self.columns]
        f_col = "first_stage_f"
        has_f = any(_col(f_col, lab) in self.columns for lab in labs)
        # The label column is as wide as the widest label, not a hardcoded 6:
        # the regime/sign estimators use short labels (``H``/``pos``), but
        # ``spatial_lp`` reports ``direct``/``indirect``/``indirect_all``, which
        # overflowed the field and shifted that row's numbers out from under
        # their headers.
        lab_w = max(6, *(len(str(l)) for l in labs)) if labs else 6
        header = f"{'h':>4}"
        if multi:
            header += f"  {'label':>{lab_w}}"
        for c in id_cols:
            header += f"  {c:>8}"
        header += f"  {'beta':>10}  {'se':>10}  {'lo':>10}  {'hi':>10}"
        if has_f:
            header += f"  {'F_1st':>9}"
        header += "  sig"
        lines.append(header)
        lines.append("-" * len(header))

        def fmt(v: Any, width: int = 10) -> str:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return f"{'':>{width}}"
            if not np.isfinite(fv):
                return f"{'nan':>{width}}"
            return f"{fv:>{width}.4f}"

        for _, row in self.iterrows():
            h_raw = row.get("h", np.nan)
            h_txt = f"{int(h_raw):>4}" if pd.notna(h_raw) else f"{'':>4}"
            for lab in labs:
                b = float(row.get(_col("beta", lab), np.nan))
                s = float(row.get(_col("se", lab), np.nan))
                lo = float(row.get(_col("lo", lab), np.nan))
                hi = float(row.get(_col("hi", lab), np.nan))
                line = h_txt
                if multi:
                    line += f"  {lab:>{lab_w}}"
                for c in id_cols:
                    v = row.get(c, "")
                    numeric = isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool)
                    line += f"  {float(v):>8.2f}" if numeric else f"  {str(v):>8}"
                line += f"  {fmt(b)}  {fmt(s)}  {fmt(lo)}  {fmt(hi)}"
                if has_f:
                    line += f"  {fmt(row.get(_col(f_col, lab), np.nan), 9)}"
                line += f"  {_sig_flag(b, s, lo, hi)}"
                lines.append(line.rstrip())
        lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10 (two-sided normal); "
                     "band-only estimators: * = band excludes 0")
        weak_note = self._weak_iv_footer()
        if weak_note is not None:
            lines.append(weak_note)
        return "\n".join(lines)

    def to_markdown(self, index: bool = False) -> str:
        """Render LP table as GitHub-flavored Markdown."""
        from ..reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), index=index)

    def to_latex(self, index: bool = False) -> str:
        """Render LP table as LaTeX tabular environment."""
        from ..reports import _df_to_latex

        return _df_to_latex(self.to_frame(), index=index)

    def to_typst(self, index: bool = False) -> str:
        """Render LP table as Typst table function."""
        from ..reports import _df_to_typst

        return _df_to_typst(self.to_frame(), index=index)
