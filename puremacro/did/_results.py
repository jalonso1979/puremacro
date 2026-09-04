"""Frozen-dataclass result objects for puremacro.did estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CallawaySantannaResult:
    """Result of :func:`puremacro.did.callaway_santanna`.

    Attributes
    ----------
    att_gt : pd.DataFrame
        Group-time ATTs with columns ``[g, t, event_time, att, se, lo, hi]``.
    att_event_study : pd.DataFrame
        Cohort-mean event-study aggregation with columns
        ``[event_time, att, se, lo, hi, n_cohorts]``.
    att_overall : float
        Simple-mean of post-treatment ATTs across (g, t).

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics 225(2), 200-230.
    """

    att_gt: pd.DataFrame
    att_event_study: pd.DataFrame
    att_overall: float

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        n_gt = len(self.att_gt)
        n_es = len(self.att_event_study)
        return (
            f"Callaway-Sant'Anna result\n"
            f"  group-time ATTs   : {n_gt}\n"
            f"  event-study rows  : {n_es}\n"
            f"  overall ATT       : {self.att_overall:+.4f}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return event-study ATT estimates as a DataFrame."""
        return self.att_event_study.copy()

    def to_markdown(self, **kwargs) -> str:
        """Export event-study ATTs to Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export event-study ATTs to LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export event-study ATTs to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class SunAbrahamResult:
    """Result of :func:`puremacro.did.sun_abraham`.

    Attributes
    ----------
    att_gt : pd.DataFrame
        Group-time ATTs (same as Callaway-Sant'Anna).
    att_event_study : pd.DataFrame
        Cohort-share-weighted event-study aggregation.
    att_overall : float
        Share-weighted mean of post-treatment ATTs.

    References
    ----------
    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
        in event studies with heterogeneous treatment effects. Journal of
        Econometrics 225(2), 175-199.
    """

    att_gt: pd.DataFrame
    att_event_study: pd.DataFrame
    att_overall: float

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        n_gt = len(self.att_gt)
        n_es = len(self.att_event_study)
        return (
            f"Sun-Abraham result\n"
            f"  group-time ATTs   : {n_gt}\n"
            f"  event-study rows  : {n_es}\n"
            f"  overall ATT       : {self.att_overall:+.4f}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return event-study ATT estimates as a DataFrame."""
        return self.att_event_study.copy()

    def to_markdown(self, **kwargs) -> str:
        """Export event-study ATTs to Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export event-study ATTs to LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export event-study ATTs to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class BorusyakJaravelSpiessResult:
    """Result of :func:`puremacro.did.borusyak_jaravel_spiess`.

    Attributes
    ----------
    tau_it : pd.DataFrame
        Per-treated-cell estimates with columns
        ``[unit, time, event_time, tau]``.
    att_event_study : pd.DataFrame
        Event-study aggregation with columns
        ``[event_time, att, se, lo, hi, n_obs]``.
    att_overall : float
        Cell-weighted mean of post-treatment ATTs.

    References
    ----------
    Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-
        study designs: robust and efficient estimation. Review of
        Economic Studies (forthcoming).
    """

    tau_it: pd.DataFrame
    att_event_study: pd.DataFrame
    att_overall: float

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        n_treated = len(self.tau_it)
        n_es = len(self.att_event_study)
        return (
            f"Borusyak-Jaravel-Spiess result\n"
            f"  treated cells     : {n_treated}\n"
            f"  event-study rows  : {n_es}\n"
            f"  overall ATT       : {self.att_overall:+.4f}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return event-study ATT estimates as a DataFrame."""
        return self.att_event_study.copy()

    def to_markdown(self, **kwargs) -> str:
        """Export event-study ATTs to Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export event-study ATTs to LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export event-study ATTs to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class SyntheticDiDResult:
    """Result of :func:`puremacro.did.synthetic_did`.

    Attributes
    ----------
    tau : float
        Synthetic-DiD point estimate.
    omega : pd.Series
        Donor-unit weights (sum to 1).
    lambda_w : pd.Series
        Pre-period time weights (sum to 1). Renamed from ``lambda`` since
        ``lambda`` is a Python reserved keyword.
    se : float
        Bootstrap standard error.
    lo : float
        Lower bootstrap percentile.
    hi : float
        Upper bootstrap percentile.
    treatment_time : float
        Common treatment time identified by the estimator.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118.
    """

    tau: float
    omega: pd.Series
    lambda_w: pd.Series
    se: float
    lo: float
    hi: float
    treatment_time: float

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        return (
            f"Synthetic-DiD result\n"
            f"  treatment time    : {self.treatment_time}\n"
            f"  donor units       : {len(self.omega)}\n"
            f"  pre-period weights: {len(self.lambda_w)}\n"
            f"  τ̂                 : {self.tau:+.4f} (se {self.se:.4f})\n"
            f"  CI                : [{self.lo:+.4f}, {self.hi:+.4f}]\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return summary of point estimate, standard error, and CI."""
        return pd.DataFrame([{
            "tau": self.tau,
            "se": self.se,
            "lo": self.lo,
            "hi": self.hi,
            "treatment_time": self.treatment_time,
        }])

    def to_markdown(self, **kwargs) -> str:
        """Export summary to Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export summary to LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export summary to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=False, **kwargs)


@dataclass(frozen=True)
class CdHResult:
    """Result of :func:`puremacro.did.cdh_did`.

    Attributes
    ----------
    att_M : float
        Instantaneous DID_M — average treatment effect on switchers
        at the moment they switch.
    att_M_l : np.ndarray
        Long-run DID_M^l estimates for ``l = 1, ..., L`` periods after
        the switch. Shape ``(L,)``; entries are ``NaN`` for horizons
        with no switchers.
    se_M : float
        Bootstrap standard error of ``att_M``.
    se_M_l : np.ndarray
        Bootstrap standard errors of ``att_M_l``. Shape ``(L,)``.
    placebo_p : float | None
        Uniform p-value of the switchers placebo test
        (``None`` if ``placebo=False``).
    n_switchers : int
        Number of (unit, time) switching events on which DID_M is
        identified.
    n_boot : int
        Number of bootstrap replications used for SEs.
    horizons : tuple[int, ...]
        The ``l`` values for which long-run effects are reported.
    names : tuple[str, ...]
        Column / feature names; for CdH this is the singleton tuple
        ``("att_M",)`` plus event-time labels for ``att_M_l``.

    References
    ----------
    de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed
        effects estimators with heterogeneous treatment effects. AER
        110(9), 2964-2996.
    """

    att_M: float
    att_M_l: np.ndarray
    se_M: float
    se_M_l: np.ndarray
    placebo_p: Optional[float]
    n_switchers: int
    n_boot: int
    horizons: tuple
    names: tuple

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        ll_lines = []
        for h, val, se in zip(self.horizons, self.att_M_l, self.se_M_l):
            if np.isnan(val):
                ll_lines.append(f"    l = {h:>2d} :   (no switchers)")
            else:
                ll_lines.append(
                    f"    l = {h:>2d} : {val:+.4f} (se {se:.4f})"
                )
        ll_block = "\n".join(ll_lines) if ll_lines else "    (none)"
        placebo_line = (
            f"  placebo p         : {self.placebo_p:.4f}"
            if self.placebo_p is not None
            else "  placebo p         : (not run)"
        )
        return (
            f"de Chaisemartin-D'Haultfoeuille (CdH 2020) result\n"
            f"  DID_M (instantaneous): {self.att_M:+.4f} "
            f"(se {self.se_M:.4f})\n"
            f"  DID_M^l (long-run)   :\n{ll_block}\n"
            f"  switchers         : {self.n_switchers}\n"
            f"  bootstrap reps    : {self.n_boot}\n"
            f"{placebo_line}\n"
        )


@dataclass(frozen=True)
class SDIDMultiResult:
    """Result of :func:`puremacro.did.sdid_multi_cohort`.

    Attributes
    ----------
    att : float
        Aggregated ATT (cohort-size-weighted) when
        ``aggregation="att"``; the cohort-size-weighted average of the
        cohort-specific SDID estimates when ``aggregation="att_g_t"``.
    se : float
        Bootstrap standard error of ``att``.
    cohort_weights : np.ndarray
        Per-cohort population weights, shape ``(n_cohorts,)``;
        sum to 1.
    cohort_atts : np.ndarray
        Per-cohort SDID point estimates, shape ``(n_cohorts,)``.
    cohort_times : np.ndarray
        First-treat times of the identified cohorts,
        shape ``(n_cohorts,)``.
    att_g_t : pd.DataFrame | None
        Full ``cohort × event_time`` grid with columns
        ``[cohort, event_time, att]`` (``None`` unless
        ``aggregation="att_g_t"``).
    aggregation : str
        Either ``"att"`` or ``"att_g_t"`` — which aggregation produced
        the headline ``att``.
    n_boot : int
        Bootstrap replications.
    names : tuple[str, ...]
        Cohort labels (string-cast ``cohort_times``).

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118.
    Roth, J., Sant'Anna, P., Bilinski, A. and Poe, J. (2023). What's
        trending in difference-in-differences? A synthesis of the
        recent econometrics literature. Journal of Econometrics
        235(2), 2218-2244.
    """

    att: float
    se: float
    cohort_weights: np.ndarray
    cohort_atts: np.ndarray
    cohort_times: np.ndarray
    att_g_t: Optional[pd.DataFrame]
    aggregation: str
    n_boot: int
    names: tuple

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        cohort_lines = []
        for c, w, a in zip(self.cohort_times, self.cohort_weights,
                            self.cohort_atts):
            cohort_lines.append(
                f"    cohort g={c}: weight={w:.3f}  ATT={a:+.4f}"
            )
        block = "\n".join(cohort_lines) if cohort_lines else "    (none)"
        return (
            f"Multi-cohort Synthetic-DiD result\n"
            f"  aggregation       : {self.aggregation}\n"
            f"  cohorts           : {len(self.cohort_atts)}\n"
            f"  aggregated ATT    : {self.att:+.4f} (se {self.se:.4f})\n"
            f"  bootstrap reps    : {self.n_boot}\n"
            f"  per-cohort:\n{block}\n"
        )
