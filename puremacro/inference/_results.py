"""Frozen-dataclass result objects for puremacro.inference.

Every result implements the presentation contract shared by the package's
result objects: ``summary()`` (plain text), ``to_frame()`` (a tidy
``pandas.DataFrame``), ``to_markdown()`` / ``to_latex()`` / ``to_typst()``
(rendered through the :mod:`puremacro.reports` table helpers) and
``plot()`` (a matplotlib ``Figure``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _render(df: pd.DataFrame, fmt: str) -> str:
    """Render a table through the shared reports helpers."""
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst

    if fmt == "markdown":
        return _df_to_markdown(df, index=False)
    if fmt == "latex":
        return _df_to_latex(df, index=False)
    if fmt == "typst":
        return _df_to_typst(df, index=False)
    raise ValueError(f"unknown fmt {fmt!r}")  # pragma: no cover


def _fmt(v: Any, digits: int) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), digits)
    return v


def _kv_frame(rows: list[tuple[str, Any]], digits: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"statistic": [r[0] for r in rows],
         "value": [_fmt(r[1], digits) for r in rows]}
    )


@dataclass(frozen=True)
class ARTestResult:
    """Result of :func:`puremacro.inference.weak_iv.anderson_rubin_test`.

    Attributes
    ----------
    stat : float
        Anderson-Rubin F-statistic.
    p_value : float
        F-distribution p-value of the test.
    df_num : int
        Numerator degrees of freedom (number of instruments).
    df_den : int
        Denominator degrees of freedom (T − k_unrestricted).
    residual_ss : float
        Residual sum of squares of the unrestricted regression.

    References
    ----------
    Anderson, T.W. and Rubin, H. (1949). Estimation of the parameters of
        a single equation in a complete system of stochastic equations.
        Annals of Mathematical Statistics 20(1), 46-63.
    """

    stat: float
    p_value: float
    df_num: int
    df_den: int
    residual_ss: float

    def summary(self) -> str:
        return (
            f"Anderson-Rubin test\n"
            f"  F-statistic       : {self.stat:.4f}\n"
            f"  p-value           : {self.p_value:.4f}\n"
            f"  df (num, den)     : ({self.df_num}, {self.df_den})\n"
            f"  residual SS       : {self.residual_ss:.4f}\n"
        )

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        """Two-column ``statistic`` / ``value`` table."""
        return _kv_frame([
            ("F", self.stat),
            ("p_value", self.p_value),
            ("df_num", self.df_num),
            ("df_den", self.df_den),
            ("residual_ss", self.residual_ss),
        ], digits)

    def to_markdown(self, digits: int = 4) -> str:
        """Render as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 4) -> str:
        """Render as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 4) -> str:
        """Render as a Typst ``#table``."""
        return _render(self.to_frame(digits), "typst")

    def plot(self, *, alpha: float = 0.05, title: str = "", ax=None):
        """Plot the null F(df_num, df_den) density with the observed
        statistic and the ``1 - alpha`` critical value marked. Returns the
        Figure."""
        from scipy.stats import f as _f_dist

        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        if self.df_num > 0 and self.df_den > 0 and np.isfinite(self.stat):
            crit = float(_f_dist.ppf(1 - alpha, self.df_num, self.df_den))
            hi = max(crit, self.stat) * 1.25 + 1e-9
            grid = np.linspace(1e-6, hi, 400)
            ax.plot(grid, _f_dist.pdf(grid, self.df_num, self.df_den),
                    color="0.1", lw=1.0, label=f"F({self.df_num}, {self.df_den})")
            ax.axvline(crit, color="0.5", ls="--", lw=0.9,
                       label=f"{100 * (1 - alpha):.0f}% critical value")
            ax.axvline(self.stat, color="0.1", ls="-", lw=1.4,
                       label=f"AR statistic (p={self.p_value:.3f})")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("F")
        ax.set_ylabel("density")
        ax.set_title(title or "Anderson-Rubin test")
        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class SupTBandResult:
    """Result of :func:`puremacro.inference.supt.supt_band`.

    Sup-t simultaneous confidence band of Montiel Olea & Plagborg-Møller
    (2019): ``lower/upper = center ± crit_value * scale``, where
    ``crit_value`` is calibrated so the whole path is covered jointly with
    probability 1 - alpha.

    Attributes
    ----------
    lower : ndarray, shape (H,)
        Lower simultaneous band, ``center - crit_value * scale``.
    upper : ndarray, shape (H,)
        Upper simultaneous band, ``center + crit_value * scale``.
    center : ndarray, shape (H,)
        Band center: ``theta`` (plugin/bootstrap) or the posterior mean
        of the draws (bayes; also the bootstrap fallback when ``theta``
        is omitted).
    scale : ndarray, shape (H,)
        Per-coordinate standard errors: ``sqrt(diag(Sigma))`` (plugin) or
        the sample std of the draws (bootstrap/bayes).
    crit_value : float
        Sup-t critical value ``c`` — the (1 - alpha) quantile of the max
        absolute studentized deviation. Always >= the pointwise normal
        critical value and <= the Bonferroni one (up to MC error).
    method : str
        'plugin', 'bootstrap', or 'bayes' (Algorithms 1-3 of the paper).
    alpha : float
        Simultaneous two-sided level (0.10 -> 90% band).
    n_draws : int
        Monte Carlo draws used (plugin) or number of bootstrap/posterior
        draws supplied.

    References
    ----------
    Montiel Olea, J.L. and Plagborg-Møller, M. (2019). Simultaneous
        confidence bands: Theory, implementation, and an application to
        SVARs. Journal of Applied Econometrics 34(1), 1-17.
    """

    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    crit_value: float
    method: str
    alpha: float
    n_draws: int

    def summary(self) -> str:
        width = self.upper - self.lower
        return (
            f"Sup-t simultaneous band ({100 * (1 - self.alpha):.0f}%)\n"
            f"  method            : {self.method}\n"
            f"  coordinates (H)   : {self.lower.shape[0]}\n"
            f"  crit value (c)    : {self.crit_value:.4f}\n"
            f"  draws used        : {self.n_draws}\n"
            f"  mean band width   : {width.mean():.4f}\n"
        )

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        """Per-coordinate table with columns ``h``, ``center``, ``scale``,
        ``lower``, ``upper``."""
        H = self.lower.shape[0]
        return pd.DataFrame({
            "h": np.arange(H),
            "center": np.round(np.asarray(self.center, dtype=float), digits),
            "scale": np.round(np.asarray(self.scale, dtype=float), digits),
            "lower": np.round(np.asarray(self.lower, dtype=float), digits),
            "upper": np.round(np.asarray(self.upper, dtype=float), digits),
        })

    def to_markdown(self, digits: int = 4) -> str:
        """Render the band table as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 4) -> str:
        """Render the band table as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 4) -> str:
        """Render the band table as a Typst ``#table``."""
        return _render(self.to_frame(digits), "typst")

    def plot(self, *, title: str = "", ylabel: str = "response", ax=None):
        """Plot the center path with the simultaneous band. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        h = np.arange(self.lower.shape[0])
        ax.fill_between(h, self.lower, self.upper, color="0.85",
                        label=f"{100 * (1 - self.alpha):.0f}% sup-t band")
        ax.plot(h, self.center, color="0.1", lw=1.2, label="center")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Sup-t band ({self.method}, c={self.crit_value:.3f})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class LewbelIVResult:
    """Result of :func:`puremacro.inference.lewbel_iv.lewbel_iv`.

    Lewbel (2012) 2SLS using instruments constructed from
    heteroskedasticity in exogenous drivers.

    Attributes
    ----------
    beta : ndarray, shape (k_endog + k_exog,)
        2SLS coefficients. Endogenous-regressor coefficients come first,
        in the order given to ``lewbel_iv``; exogenous follow.
    se : ndarray, shape (k_endog + k_exog,)
        Standard errors (homoskedastic 2SLS; HAC variant deferred).
    t : ndarray, shape (k_endog + k_exog,)
        t-statistics, ``beta / se``.
    n_obs : int
        Sample size after dropping rows with NaN inputs.
    n_iv_constructed : int
        Number of Lewbel instruments constructed: (number of non-constant
        columns of ``heterosk_source``) × k_endog.
    first_stage_F : float
        First-stage F statistic for the **first** endogenous regressor only
        (joint significance of the constructed IVs). Multi-regressor extension
        is deferred to a future release; with `k_endog > 1`, inspect the
        coefficients on each regressor separately for instrument relevance.
    lewbel_diagnostic : dict
        Breusch-Pagan-style identification test. Keys: ``stat``,
        ``p_value``. Small p indicates the constructed instrument is
        strong; ``p > 0.10`` is treated as weak identification.

    References
    ----------
    Lewbel, A. (2012). Using heteroscedasticity to identify and estimate
        mismeasured and endogenous regressor models. Journal of Business
        and Economic Statistics 30(1), 67-80.
    """

    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    n_obs: int
    n_iv_constructed: int
    first_stage_F: float
    lewbel_diagnostic: dict

    def summary(self) -> str:
        diag = self.lewbel_diagnostic
        strength = "STRONG" if diag["p_value"] < 0.10 else "WEAK"
        return (
            f"Lewbel-IV result\n"
            f"  obs (n_obs)              : {self.n_obs}\n"
            f"  Lewbel IVs constructed   : {self.n_iv_constructed}\n"
            f"  first-stage F            : {self.first_stage_F:.2f}\n"
            f"  Lewbel diag p-value      : {diag['p_value']:.4f} [{strength}]\n"
        )

    def to_frame(self, digits: int = 4,
                 names: list[str] | None = None) -> pd.DataFrame:
        """Coefficient table with columns ``variable``, ``beta``, ``se``, ``t``."""
        beta = np.asarray(self.beta, dtype=float).ravel()
        if names is None:
            names = [f"x{i}" for i in range(beta.size)]
        if len(names) != beta.size:
            raise ValueError(
                f"names has {len(names)} entries but beta has {beta.size}"
            )
        return pd.DataFrame({
            "variable": list(names),
            "beta": np.round(beta, digits),
            "se": np.round(np.asarray(self.se, dtype=float).ravel(), digits),
            "t": np.round(np.asarray(self.t, dtype=float).ravel(), digits),
        })

    def to_markdown(self, digits: int = 4, names: list[str] | None = None) -> str:
        """Render the coefficient table as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits, names), "markdown")

    def to_latex(self, digits: int = 4, names: list[str] | None = None) -> str:
        """Render the coefficient table as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits, names), "latex")

    def to_typst(self, digits: int = 4, names: list[str] | None = None) -> str:
        """Render the coefficient table as a Typst ``#table``."""
        return _render(self.to_frame(digits, names), "typst")

    def plot(self, *, names: list[str] | None = None, level: float = 0.95,
             title: str = "", ax=None):
        """Coefficient plot with ``level`` normal confidence intervals.
        Returns the Figure."""
        from scipy.stats import norm

        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        beta = np.asarray(self.beta, dtype=float).ravel()
        se = np.asarray(self.se, dtype=float).ravel()
        if names is None:
            names = [f"x{i}" for i in range(beta.size)]
        z = float(norm.ppf(0.5 + level / 2))
        pos = np.arange(beta.size)
        ax.errorbar(pos, beta, yerr=z * se, fmt="o", color="0.1",
                    ecolor="0.4", capsize=3)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(list(names))
        ax.set_ylabel("coefficient")
        ax.set_title(title or f"Lewbel IV (first-stage F={self.first_stage_F:.1f})")
        fig.tight_layout()
        return fig
