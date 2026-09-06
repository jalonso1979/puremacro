"""Frozen-dataclass result objects for puremacro.garch.

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
    """Render a parameter table through the shared reports helpers."""
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


@dataclass(frozen=True)
class GARCH11Result:
    """Result of :func:`puremacro.garch.fit.garch11_fit`.

    Attributes
    ----------
    omega : float
        Constant in the GARCH(1,1) recursion.
    alpha : float
        ARCH coefficient (loading on lagged squared shock).
    beta : float
        GARCH coefficient (loading on lagged conditional variance).
    sigma : pd.Series
        Conditional volatility σ_t.
    loglik : float
        Log-likelihood at the MLE.
    converged : bool
        Optimiser convergence flag.
    persistence : float
        ``α + β``.

    References
    ----------
    Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    """

    omega: float
    alpha: float
    beta: float
    sigma: pd.Series
    loglik: float
    converged: bool
    persistence: float

    def summary(self) -> str:
        return (
            f"GARCH(1,1) fit\n"
            f"  ω                 : {self.omega:.6f}\n"
            f"  α                 : {self.alpha:.4f}\n"
            f"  β                 : {self.beta:.4f}\n"
            f"  persistence (α+β) : {self.persistence:.4f}\n"
            f"  log-lik           : {self.loglik:+.4f}\n"
            f"  converged         : {self.converged}\n"
        )

    def to_frame(self, digits: int = 6) -> pd.DataFrame:
        """Parameter table with columns ``parameter`` and ``value``."""
        rows = [
            ("omega", self.omega),
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("persistence", self.persistence),
            ("loglik", self.loglik),
            ("n_obs", int(len(self.sigma))),
            ("converged", self.converged),
        ]
        return pd.DataFrame(
            {"parameter": [r[0] for r in rows],
             "value": [_fmt(r[1], digits) for r in rows]}
        )

    def to_markdown(self, digits: int = 6) -> str:
        """Render the parameter table as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 6) -> str:
        """Render the parameter table as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 6) -> str:
        """Render the parameter table as a Typst ``#table``."""
        return _render(self.to_frame(digits), "typst")

    def plot(self, *, title: str = "", ylabel: str = "conditional volatility",
             ax=None):
        """Plot the conditional-volatility path σ_t. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        ax.plot(self.sigma.index, self.sigma.values, color="0.1", lw=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"GARCH(1,1): α={self.alpha:.3f}, β={self.beta:.3f}")
        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class DCCResult:
    """Result of :func:`puremacro.garch.dcc.dcc_fit`.

    Attributes
    ----------
    a : float
        DCC innovation parameter.
    b : float
        DCC persistence parameter.
    Qbar : ndarray, shape (n, n)
        Unconditional standardised-residual covariance.
    sigma : pd.DataFrame, shape (T, n)
        Per-asset conditional volatility.
    R : ndarray, shape (T, n, n)
        Conditional correlation path.
    H : ndarray, shape (T, n, n)
        Conditional covariance path.
    garch_params : list of dict
        Per-asset GARCH(1,1) parameters and diagnostics.
    loglik : float
        DCC-stage log-likelihood.
    converged : bool
        Optimiser convergence flag.

    References
    ----------
    Engle, R. (2002). Dynamic conditional correlation: a simple class of
        multivariate generalized autoregressive conditional
        heteroskedasticity models. JBES 20(3), 339-350.
    """

    a: float
    b: float
    Qbar: np.ndarray
    sigma: pd.DataFrame
    R: np.ndarray
    H: np.ndarray
    garch_params: list
    loglik: float
    converged: bool

    def summary(self) -> str:
        T, n = self.sigma.shape
        return (
            f"DCC(1,1) fit\n"
            f"  series (n)        : {n}\n"
            f"  observations (T)  : {T}\n"
            f"  a                 : {self.a:.4f}\n"
            f"  b                 : {self.b:.4f}\n"
            f"  a + b             : {self.a + self.b:.4f}\n"
            f"  log-lik (DCC)     : {self.loglik:+.4f}\n"
            f"  converged         : {self.converged}\n"
        )

    def to_frame(self, digits: int = 6) -> pd.DataFrame:
        """Parameter table: DCC ``a``/``b`` rows followed by one row per
        asset with its GARCH(1,1) ``omega``, ``alpha``, ``beta``.

        Columns: ``series``, ``parameter``, ``value``.
        """
        T, n = self.sigma.shape
        rows: list[tuple[str, str, Any]] = [
            ("DCC", "a", self.a),
            ("DCC", "b", self.b),
            ("DCC", "a+b", self.a + self.b),
            ("DCC", "loglik", self.loglik),
            ("DCC", "n_obs", int(T)),
            ("DCC", "converged", self.converged),
        ]
        for name, g in zip(self.sigma.columns, self.garch_params):
            for key in ("omega", "alpha", "beta", "persistence"):
                if key in g:
                    rows.append((str(name), key, g[key]))
        return pd.DataFrame(
            {"series": [r[0] for r in rows],
             "parameter": [r[1] for r in rows],
             "value": [_fmt(r[2], digits) for r in rows]}
        )

    def to_markdown(self, digits: int = 6) -> str:
        """Render the parameter table as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 6) -> str:
        """Render the parameter table as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 6) -> str:
        """Render the parameter table as a Typst ``#table``."""
        return _render(self.to_frame(digits), "typst")

    def correlations(self) -> pd.DataFrame:
        """Conditional correlations ρ_{ij,t} for every pair i < j, as a
        ``(T, n(n-1)/2)`` DataFrame indexed like ``sigma``."""
        cols = list(self.sigma.columns)
        n = len(cols)
        data = {}
        for i in range(n):
            for j in range(i + 1, n):
                data[f"{cols[i]}-{cols[j]}"] = self.R[:, i, j]
        return pd.DataFrame(data, index=self.sigma.index)

    def plot(self, *, title: str = "", ylabel: str = "conditional correlation",
             ax=None):
        """Plot the conditional-correlation paths ρ_{ij,t}. Returns the Figure."""
        from ..plot import _new_ax, _palette, _styles

        fig, ax = _new_ax(ax)
        corr = self.correlations()
        colors = _palette(corr.shape[1])
        styles = _styles(corr.shape[1])
        for k, col in enumerate(corr.columns):
            ax.plot(corr.index, corr[col].values, color=colors[k],
                    linestyle=styles[k], lw=1.0, label=col)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"DCC(1,1): a={self.a:.3f}, b={self.b:.3f}")
        if corr.shape[1] > 1:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig
