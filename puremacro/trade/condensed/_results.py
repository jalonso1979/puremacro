"""Result container of the condensed solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .calibration import CondensedCalibration
from .certify import RawFlowCertificate
from .measures import _ev_all, consumer_price_relative_to_wages, tot_fisher
from .model import CondensedState
from .tariffs import TariffWedges


@dataclass(frozen=True)
class CondensedEquilibriumResult:
    """Solved equilibrium of the condensed model with its certificate and diagnostics.

    Attributes
    ----------
    state : CondensedState
        The accepted allocation (prices, quantities, revenues, residuals).
    certificate : RawFlowCertificate or None
        The independent ten-block raw-flow certificate (None when the solve was
        run with ``certify=False``; the result is then neither ``passed`` nor
        ``certified``).
    converged : bool
        True; a failed solve raises instead of returning.
    iterations, jacobians : int
        Newton iterations and Jacobian evaluations of the final stage.
    max_scaled_residual, walras_residual : float
        ``max |r|`` over the ``2N`` equations and the dropped income residual.
    path : tuple of dict
        Continuation stage history (``lam``, ``ok``, iterations, residuals, seconds).
    arclength_path, multistart : tuple of dict
        Pseudo-arclength step history and near-start multistart diagnostics, when run.
    existence : dict
        Collatz-Wielandt bounds on ``rho(B_tau)`` from the existence gate.
    seconds : float
        Wall-clock time of the whole solve.
    calib : CondensedCalibration
    wedges : TariffWedges
    options : dict
        Closure, numeraire, final-demand rule, inventory treatment, method, tolerances.
    metadata : dict
    """

    state: CondensedState
    certificate: RawFlowCertificate | None
    converged: bool
    iterations: int
    jacobians: int
    max_scaled_residual: float
    walras_residual: float
    path: tuple[dict[str, Any], ...]
    seconds: float
    calib: CondensedCalibration
    wedges: TariffWedges
    options: dict[str, Any] = field(default_factory=dict)
    arclength_path: tuple[dict[str, Any], ...] = ()
    multistart: tuple[dict[str, Any], ...] = ()
    existence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the solve converged and the raw-flow certificate was evaluated and passed.

        A result produced with ``certify=False`` has no certificate and is never
        ``passed``; :attr:`converged` alone reports the Newton stopping rule.
        """
        return bool(self.converged and self.certificate is not None and self.certificate.passed)

    @property
    def certified(self) -> bool:
        """True when a certificate was evaluated and every block is within its tolerance."""
        return bool(self.certificate is not None and self.certificate.passed)

    @property
    def country_codes(self) -> tuple[str, ...]:
        """Country registry of the calibration, in storage order."""
        return self.calib.country_codes

    @property
    def z(self) -> np.ndarray:
        """The unknown vector ``(log w, Y / Y0)``."""
        return self.state.z

    @property
    def pc_over_w_pct(self) -> dict[str, float]:
        """M1 by country: consumer prices relative to wages, percent."""
        return {c: 100.0 * consumer_price_relative_to_wages(self.calib, self.state, country=k)
                for k, c in enumerate(self.country_codes)}

    @property
    def ev_pct_gdp(self) -> dict[str, float]:
        """M2 by country: equivalent variation as percent of base GDP."""
        ev = _ev_all(self.calib, self.state)
        return {c: 100.0 * float(ev[k] / self.calib.Y0[k]) for k, c in enumerate(self.country_codes)}

    @property
    def tot_pct(self) -> dict[str, float]:
        """M4 by country: Fisher fob ex-duty terms of trade, percent."""
        return {c: 100.0 * tot_fisher(self.calib, self.state, country=k) for k, c in enumerate(self.country_codes)}

    def to_dataframe(self) -> pd.DataFrame:
        """One row per country: wage, income ratio, transfers, revenues and the numeraire-free headline measures."""
        st = self.state
        pc = self.pc_over_w_pct
        ev = self.ev_pct_gdp
        tot = self.tot_pct
        codes = list(self.country_codes)
        return pd.DataFrame({
            "w": st.w, "Y_hat": st.Y_hat, "XN": st.XN, "T": st.T, "TR": st.TR, "PT": st.PT,
            "pc_over_w_pct": [pc[c] for c in codes], "ev_pct_gdp": [ev[c] for c in codes],
            "tot_pct": [tot[c] for c in codes],
        }, index=pd.Index(codes, name="country"))

    def summary(self) -> str:
        """One-line description: size, options, residuals, certificate status and solver effort."""
        cert = "no certificate" if self.certificate is None else f"certificate max block {self.certificate.max_block:.2e}"
        return (f"CondensedEquilibriumResult: {self.calib.n_countries} economies x {self.calib.n_sectors} industries; "
                f"closure={self.options.get('closure')}, numeraire={self.options.get('numeraire')}; "
                f"max|r|={self.max_scaled_residual:.2e}, Walras={self.walras_residual:.2e}; {cert}; "
                f"{self.iterations} Newton iterations, {len(self.path)} stage(s), {self.seconds:.2f} s")

    def to_markdown(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The country table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


__all__ = ["CondensedEquilibriumResult"]
