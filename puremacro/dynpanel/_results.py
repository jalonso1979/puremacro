"""Frozen-dataclass result objects for puremacro.dynpanel.

References
----------
Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data: Monte Carlo evidence and an application to employment equations.
    Review of Economic Studies 58(2), 277-297.
Blundell, R. and Bond, S. (1998). Initial conditions and moment
    restrictions in dynamic panel data models. Journal of Econometrics
    87(1), 115-143.
Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators. Journal of Econometrics
    126, 25-51.
Roodman, D. (2009). How to do xtabond2: an introduction to difference and
    system GMM in Stata. Stata Journal 9(1), 86-136.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GMMResult:
    """Result of :func:`puremacro.dynpanel.ab_gmm` and
    :func:`puremacro.dynpanel.bb_gmm`.

    Attributes
    ----------
    coefs : ndarray, shape (k,)
        Estimated coefficient vector (lagged y first, then endogenous,
        predetermined, and exogenous regressors in that order).
    se : ndarray, shape (k,)
        Standard errors. Two-step + Windmeijer (2005) corrected by
        default; one-step robust if ``two_step=False``.
    cov : ndarray, shape (k, k)
        Coefficient covariance matrix.
    names : tuple[str, ...]
        Coefficient labels (length k).
    hansen_j : float
        Hansen J overidentification test statistic.
    hansen_j_p : float
        Chi-square p-value for the Hansen J test.
    hansen_j_df : int
        Degrees of freedom: ``n_instruments - n_regressors``.
    ar1_p : float
        Arellano-Bond AR(1) test p-value (two-sided normal). Under
        correct specification this is typically small (differencing
        induces lag-1 serial correlation by construction).
    ar2_p : float
        Arellano-Bond AR(2) test p-value (two-sided normal). Under
        correct specification this should NOT reject (large p-value).
    n_instruments : int
        Number of moment conditions (columns of ``Z``).
    n_obs : int
        Total observations used in the moment conditions (sum across
        panels of usable post-differencing rows).
    n_panels : int
        Number of panels (units) contributing observations.
    step : int
        1 (one-step GMM) or 2 (two-step GMM).
    windmeijer : bool
        Whether the Windmeijer (2005) finite-sample SE correction was
        applied. Only relevant for ``step=2``.
    estimator : str
        Either ``"ab"`` (Arellano-Bond difference GMM) or ``"bb"``
        (Blundell-Bond system GMM).
    converged : bool
        Whether the estimator produced finite estimates.

    References
    ----------
    See module-level docstring.
    """

    coefs: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    names: tuple
    hansen_j: float
    hansen_j_p: float
    hansen_j_df: int
    ar1_p: float
    ar2_p: float
    n_instruments: int
    n_obs: int
    n_panels: int
    step: int
    windmeijer: bool
    estimator: str
    converged: bool

    def summary(self) -> str:
        """Multi-line human-readable summary of the fit."""
        label = {
            "ab": "Arellano-Bond difference GMM",
            "bb": "Blundell-Bond system GMM",
        }.get(self.estimator, self.estimator)
        lines = [
            f"{label} (step {self.step}"
            + (", Windmeijer)" if (self.step == 2 and self.windmeijer) else ")"),
            f"  panels            : {self.n_panels}",
            f"  observations      : {self.n_obs}",
            f"  instruments       : {self.n_instruments}",
            f"  Hansen J          : {self.hansen_j:.3f}"
            f"  (df {self.hansen_j_df}, p {self.hansen_j_p:.3f})",
            f"  AR(1) p-value     : {self.ar1_p:.3f}",
            f"  AR(2) p-value     : {self.ar2_p:.3f}",
            "",
            "  coefficient        estimate        se         z",
        ]
        for nm, b, s in zip(self.names, self.coefs, self.se):
            z = b / s if s > 0 else float("nan")
            lines.append(f"  {nm:<18s} {b:+.4f}        {s:.4f}    {z:+.2f}")
        return "\n".join(lines) + "\n"

    def to_frame(self):
        """Return coefficient table as a pandas DataFrame."""
        import pandas as pd
        from scipy.stats import norm

        rows = []
        for nm, b, s in zip(self.names, self.coefs, self.se):
            z = b / s if s > 0 else float("nan")
            p_val = 2.0 * float(norm.sf(abs(z))) if not np.isnan(z) else float("nan")
            rows.append({
                "variable": nm,
                "coef": float(b),
                "se": float(s),
                "z": float(z),
                "p": p_val,
            })
        return pd.DataFrame(rows).set_index("variable")

    def to_markdown(self, **kwargs) -> str:
        """Render GMM estimates as Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render GMM estimates as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render GMM estimates as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


__all__ = ["GMMResult"]
