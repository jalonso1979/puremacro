"""Frozen-dataclass result objects for puremacro.did estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _overall_line(att: float, se: float, lo: float, hi: float,
                  alpha: Optional[float]) -> str:
    """``+x.xxxx  (se s, 90% CI [lo, hi])`` for the summaries below."""
    line = f"{att:+.4f}"
    if se is not None and np.isfinite(se):
        level = "" if alpha is None or not np.isfinite(alpha) else f"{100 * (1 - alpha):.0f}% "
        line += f"  (se {se:.4f}, {level}CI [{lo:+.4f}, {hi:+.4f}])"
    return line


def _aggregation_line(aggregation: Optional[str]) -> str:
    """The ``aggregation :`` summary line, or nothing when it was not set."""
    if aggregation is None:
        return ""
    return f"  aggregation       : {aggregation}\n"


def _ci_label(alpha: Optional[float]) -> str:
    """``"90% CI"`` for ``alpha = 0.10``; plain ``"CI"`` when unknown."""
    if alpha is None or not np.isfinite(alpha):
        return "CI"
    return f"{100 * (1 - alpha):.0f}% CI"


#: Section 5 algorithm of Arkhangelsky et al. (2021) behind each SDID se.
_SDID_SE_ALGORITHMS = {"placebo": "Algorithm 4", "bootstrap": "Algorithm 2",
                       "jackknife": "Algorithm 3"}


@dataclass(frozen=True)
class CallawaySantannaResult:
    """Result of :func:`puremacro.did.callaway_santanna`.

    Equation numbers are those of Callaway and Sant'Anna (2021), arXiv
    1803.09015v4. ``n_g`` is the number of units in cohort ``g``; every
    aggregate weights cohorts by it (it estimates P(G = g | G <= T)).

    Attributes
    ----------
    att_gt : pd.DataFrame
        Group-time ATTs with columns ``[g, t, event_time, att, se, lo, hi]``.
    att_event_study : pd.DataFrame
        Event study theta_es(e), eq. 3.4: at each event time the ATT(g, g+e)
        of the cohorts that identify it, weighted by cohort size. Columns
        ``[event_time, att, se, lo, hi, n_cohorts]``. (Equal cohort weights
        only under ``aggregation="unweighted"``, the rule of puremacro 4.3.0
        and earlier.)
    att_overall : float
        The overall summary chosen by ``aggregation``; by default
        theta^O_sel (eq. 3.11), the cohort-size-weighted mean of the cohort
        averages theta_sel(g) — the paper's recommended summary.
    att_overall_se, att_overall_lo, att_overall_hi : float
        Bootstrap standard error and percentile band of ``att_overall`` (the
        cohort sizes are re-estimated on every draw). NaN when ``n_boot=0``.
    aggregation : str, optional
        ``"group"`` (eq. 3.11), ``"simple"`` (eq. 3.10), ``"dynamic"`` or
        ``"calendar"`` (eq. 3.12), or ``"unweighted"`` (legacy, not a CS
        estimand). :func:`~puremacro.did.callaway_santanna` always sets it;
        it is ``None`` for a result built by hand, and ``summary()`` then
        leaves the line out.
    alpha : float, optional
        ``1 − coverage`` of every band.
    att_group : pd.DataFrame, optional
        theta_sel(g) (eq. 3.7) per cohort: columns
        ``[g, n_g, weight, att, se, lo, hi]``; ``weight`` is the cohort's
        weight in theta^O_sel.
    overall_aggregations : pd.DataFrame, optional
        Every overall summary, one row each: columns
        ``[aggregation, estimand, att, se, lo, hi]``.
    event_study_vcov : pd.DataFrame, optional
        Bootstrap covariance matrix of the ``att_event_study`` coefficients,
        indexed by event time on both axes; its diagonal is ``se**2``. Pass it
        as ``sigma=`` to :func:`puremacro.did.honest_did`.

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics 225(2), 200-230.
    """

    att_gt: pd.DataFrame
    att_event_study: pd.DataFrame
    att_overall: float
    att_overall_se: float = float("nan")
    att_overall_lo: float = float("nan")
    att_overall_hi: float = float("nan")
    aggregation: Optional[str] = None
    alpha: Optional[float] = None
    att_group: Optional[pd.DataFrame] = None
    overall_aggregations: Optional[pd.DataFrame] = None
    event_study_vcov: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        n_gt = len(self.att_gt)
        n_es = len(self.att_event_study)
        overall = _overall_line(self.att_overall, self.att_overall_se,
                                self.att_overall_lo, self.att_overall_hi,
                                self.alpha)
        return (
            f"Callaway-Sant'Anna result\n"
            f"  group-time ATTs   : {n_gt}\n"
            f"  event-study rows  : {n_es}\n"
            f"  overall ATT       : {overall}\n"
            + _aggregation_line(self.aggregation)
        )

    def to_frame(self) -> pd.DataFrame:
        """Return event-study ATT estimates as a DataFrame."""
        return self.att_event_study.copy()

    def to_markdown(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Markdown table.

        The event-study frame carries a plain positional ``RangeIndex``,
        so ``index=False`` (the default) keeps it out of the table.
        """
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=index, **kwargs)

    def to_latex(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a LaTeX ``tabular`` (no index column)."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=index, **kwargs)

    def to_typst(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Typst ``#table`` (no index column)."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=index, **kwargs)

    def plot(self, *, ax=None, title: str = "Callaway-Sant'Anna Event Study", **kwargs):
        """Plot event-study ATT estimates with confidence intervals."""
        from puremacro.plot import plot_event_study
        return plot_event_study(self.att_event_study, ax=ax, title=title, **kwargs)



@dataclass(frozen=True)
class SunAbrahamResult:
    """Result of :func:`puremacro.did.sun_abraham`.

    Attributes
    ----------
    att_gt : pd.DataFrame
        Group-time ATTs (the cohort-specific CATT(e, l); same as
        Callaway-Sant'Anna).
    att_event_study : pd.DataFrame
        Interaction-weighted event study nu_l (SA eq. 27): at each relative
        period the cohort effects weighted by the sample share of each cohort
        among the cohorts observed there. ``se`` is the bootstrap standard
        deviation of the aggregate itself (joint unit-level draws with the
        shares re-estimated, so the covariance between cohorts that share
        controls is kept); ``lo``/``hi`` are ``att -/+ z se``.
    att_overall : float
        Overall summary chosen by ``aggregation``; by default ``"simple"``,
        every post-treatment ATT(g, t) cell weighted by cohort size (CS eq.
        3.10). ``"dynamic"`` is SA's own nu_g for g = all post periods.
    att_overall_se, att_overall_lo, att_overall_hi : float
        Joint-bootstrap standard error and normal band of ``att_overall``.
    aggregation : str, optional
        ``"simple"``, ``"group"``, ``"dynamic"`` or ``"calendar"``.
        :func:`~puremacro.did.sun_abraham` always sets it; it is ``None``
        for a result built by hand, and ``summary()`` then leaves the line
        out.
    alpha : float, optional
        ``1 − coverage`` of every band.
    overall_aggregations : pd.DataFrame, optional
        Every overall summary: columns ``[aggregation, estimand, att, se, lo, hi]``.
    event_study_vcov : pd.DataFrame, optional
        Bootstrap covariance matrix of the event-study coefficients; its
        diagonal is ``se**2``. Pass it as ``sigma=`` to
        :func:`puremacro.did.honest_did`.

    References
    ----------
    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
        in event studies with heterogeneous treatment effects. Journal of
        Econometrics 225(2), 175-199.
    """

    att_gt: pd.DataFrame
    att_event_study: pd.DataFrame
    att_overall: float
    att_overall_se: float = float("nan")
    att_overall_lo: float = float("nan")
    att_overall_hi: float = float("nan")
    aggregation: Optional[str] = None
    alpha: Optional[float] = None
    overall_aggregations: Optional[pd.DataFrame] = None
    event_study_vcov: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        n_gt = len(self.att_gt)
        n_es = len(self.att_event_study)
        overall = _overall_line(self.att_overall, self.att_overall_se,
                                self.att_overall_lo, self.att_overall_hi,
                                self.alpha)
        return (
            f"Sun-Abraham result\n"
            f"  group-time ATTs   : {n_gt}\n"
            f"  event-study rows  : {n_es}\n"
            f"  overall ATT       : {overall}\n"
            + _aggregation_line(self.aggregation)
        )

    def to_frame(self) -> pd.DataFrame:
        """Return event-study ATT estimates as a DataFrame."""
        return self.att_event_study.copy()

    def to_markdown(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Markdown table.

        The event-study frame carries a plain positional ``RangeIndex``,
        so ``index=False`` (the default) keeps it out of the table.
        """
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=index, **kwargs)

    def to_latex(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a LaTeX ``tabular`` (no index column)."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=index, **kwargs)

    def to_typst(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Typst ``#table`` (no index column)."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=index, **kwargs)

    def plot(self, *, ax=None, title: str = "Sun-Abraham Event Study", **kwargs):
        """Plot event-study ATT estimates with confidence intervals."""
        from puremacro.plot import plot_event_study
        return plot_event_study(self.att_event_study, ax=ax, title=title, **kwargs)



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

    def to_markdown(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Markdown table.

        The event-study frame carries a plain positional ``RangeIndex``,
        so ``index=False`` (the default) keeps it out of the table.
        """
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=index, **kwargs)

    def to_latex(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a LaTeX ``tabular`` (no index column)."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=index, **kwargs)

    def to_typst(self, *, index: bool = False, **kwargs) -> str:
        """Export event-study ATTs to a Typst ``#table`` (no index column)."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=index, **kwargs)

    def plot(self, *, ax=None, title: str = "Borusyak-Jaravel-Spiess Event Study", **kwargs):
        """Plot event-study ATT estimates with confidence intervals."""
        from puremacro.plot import plot_event_study
        return plot_event_study(self.att_event_study, ax=ax, title=title, **kwargs)



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
        Standard error: the square root of the ``se_method`` variance
        estimate of Arkhangelsky et al. (2021, Section 5), in the outcome's
        units. NaN when inference was skipped (``n_boot=0``) or is not
        defined (the jackknife with one treated unit).
    lo, hi : float
        Gaussian confidence interval ``tau ∓ z_{1−α/2}·se`` (eq. 5.1), not
        bootstrap percentiles. NaN whenever ``se`` is.
    treatment_time : float
        Common treatment time identified by the estimator.
    y_treated : pd.Series | None
        Mean outcome path of the treated units, indexed by period
        (``None`` when the result was built without trajectories).
    y_synthetic : pd.Series | None
        ω-weighted donor outcome path over the same periods; the SDID
        estimate is the post-period gap between ``y_treated`` and
        ``y_synthetic`` net of the λ-weighted pre-period gap.
    se_method : str | None
        Variance estimator behind ``se``: ``"placebo"`` (Algorithm 4),
        ``"bootstrap"`` (Algorithm 2) or ``"jackknife"`` (Algorithm 3). For
        ``synthetic_did(se_method="auto")`` it is the method ``"auto"``
        resolved to. ``None`` when :func:`~puremacro.did.synthetic_did` ran
        with ``n_boot=0`` (no variance was computed, so there is no
        estimator to name) and for a result built by hand.
    alpha : float | None
        ``1 − coverage`` of ``[lo, hi]`` (0.10 for a 90 % interval).
    n_reps : int | None
        Replications actually used: ``B`` (``n_boot``) for the placebo and
        the bootstrap, the number of units ``N`` for the jackknife, and 0
        when no inference was run (``n_boot=0``) or the jackknife is not
        defined.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118 (arXiv:1812.09970v4: eq. 5.1; Algorithms 2-4).
    """

    tau: float
    omega: pd.Series
    lambda_w: pd.Series
    se: float
    lo: float
    hi: float
    treatment_time: float
    y_treated: Optional[pd.Series] = None
    y_synthetic: Optional[pd.Series] = None
    se_method: Optional[str] = None
    alpha: Optional[float] = None
    n_reps: Optional[int] = None

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        lines = [
            "Synthetic-DiD result",
            f"  treatment time    : {self.treatment_time}",
            f"  donor units       : {len(self.omega)}",
            f"  pre-period weights: {len(self.lambda_w)}",
            f"  τ̂                 : {self.tau:+.4f} (se {self.se:.4f})",
            f"  {_ci_label(self.alpha):<18}: [{self.lo:+.4f}, {self.hi:+.4f}]",
        ]
        if self.se_method is not None:
            method = self.se_method
            algo = _SDID_SE_ALGORITHMS.get(method)
            if algo is not None:
                method += f" ({algo})"
            if self.n_reps is not None:
                method += f", {self.n_reps} replications"
            lines.append(f"  se method         : {method}")
        return "\n".join(lines) + "\n"

    def to_frame(self) -> pd.DataFrame:
        """Return summary of point estimate, standard error, and CI.

        Columns ``[tau, se, lo, hi, treatment_time]``, followed by
        ``se_method``, ``alpha`` and ``n_reps`` for each of them that the
        result records (is not ``None``). :func:`~puremacro.did.synthetic_did`
        always records ``alpha`` and ``n_reps``, and ``se_method`` only when
        ``n_boot > 0``: with ``n_boot=0`` there is no ``se_method`` column.
        """
        row = {
            "tau": self.tau,
            "se": self.se,
            "lo": self.lo,
            "hi": self.hi,
            "treatment_time": self.treatment_time,
        }
        for name in ("se_method", "alpha", "n_reps"):
            value = getattr(self, name)
            if value is not None:
                row[name] = value
        return pd.DataFrame([row])

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

    def plot(self, *, ax=None, title: str = "Synthetic DiD"):
        """Plot the treated-mean and ω-weighted synthetic outcome paths.

        The pre-period time weights ``lambda_w`` are drawn as a bar strip
        along the bottom of the axis, and the treatment time is marked
        with a vertical line. Returns the matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        if self.y_treated is None or self.y_synthetic is None:
            raise ValueError(
                "this SyntheticDiDResult carries no outcome trajectories "
                "(y_treated / y_synthetic are None); re-run synthetic_did "
                "to obtain a plottable result"
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
        else:
            fig = ax.figure
        x_tr = np.asarray(self.y_treated.index, dtype=float)
        x_sy = np.asarray(self.y_synthetic.index, dtype=float)
        ax.plot(x_tr, self.y_treated.values, color="0.0", lw=1.5,
                label="Treated (mean)")
        ax.plot(x_sy, self.y_synthetic.values, color="0.4", ls="--", lw=1.5,
                label="Synthetic (ω-weighted donors)")
        ax.axvline(self.treatment_time, color="0.5", ls=":", lw=1.0,
                   label="Treatment")
        # λ weights as a strip at the bottom of the axis.
        y0, y1 = ax.get_ylim()
        strip = 0.12 * (y1 - y0)
        lam = self.lambda_w
        if len(lam) and float(lam.max()) > 0:
            heights = strip * (lam.values / float(lam.max()))
            ax.bar(np.asarray(lam.index, dtype=float), heights, bottom=y0,
                   width=0.8, color="0.7", alpha=0.6, label="λ (time weights)")
            ax.set_ylim(y0, y1)
        ax.set_title(f"{title}: τ̂ = {self.tau:+.3f} (se {self.se:.3f})")
        ax.set_xlabel("Period")
        ax.set_ylabel("Outcome")
        ax.legend(loc="best", frameon=False, fontsize=8)
        return fig


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

    def to_frame(self) -> pd.DataFrame:
        """Estimates as a table: ``DID_M`` then ``DID_M^l`` per horizon.

        Columns ``[estimand, horizon, att, se]``; the instantaneous
        ``DID_M`` row carries ``horizon = 0``.
        """
        rows = [{"estimand": "DID_M", "horizon": 0,
                 "att": float(self.att_M), "se": float(self.se_M)}]
        for h, val, se in zip(self.horizons, self.att_M_l, self.se_M_l):
            rows.append({"estimand": "DID_M^l", "horizon": int(h),
                         "att": float(val), "se": float(se)})
        return pd.DataFrame(rows)

    def to_markdown(self, **kwargs) -> str:
        """Export the DID_M / DID_M^l table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export the DID_M / DID_M^l table to a LaTeX ``tabular``."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export the DID_M / DID_M^l table to a Typst ``#table``."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=False, **kwargs)

    def plot(self, *, ax=None, title: str = "de Chaisemartin-D'Haultfoeuille DID_M"):
        """Plot ``DID_M`` (horizon 0) and ``DID_M^l`` with ±1.96·se bars.

        Returns the matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
        else:
            fig = ax.figure
        tab = self.to_frame()
        ok = np.isfinite(tab["att"].to_numpy(dtype=float))
        tab = tab[ok]
        x = tab["horizon"].to_numpy(dtype=float)
        y = tab["att"].to_numpy(dtype=float)
        se = tab["se"].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(se), 1.96 * se, 0.0)
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="0.0", ecolor="0.4",
                    elinewidth=1.2, capsize=3, label="DID_M / DID_M^l (95% CI)")
        ax.axhline(0.0, color="0.3", lw=0.8, ls=":")
        ax.set_xlabel("Horizon l (0 = instantaneous DID_M)")
        ax.set_ylabel("Treatment effect")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
        return fig


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

    def to_frame(self) -> pd.DataFrame:
        """Per-cohort SDID estimates plus the aggregate as the last row.

        Columns ``[cohort, weight, att, se]``. Only the aggregate carries
        a bootstrap standard error; per-cohort rows have ``se = NaN``.
        """
        rows = [
            {"cohort": str(c), "weight": float(w), "att": float(a),
             "se": float("nan")}
            for c, w, a in zip(self.cohort_times, self.cohort_weights,
                               self.cohort_atts)
        ]
        rows.append({"cohort": "aggregate", "weight": 1.0,
                     "att": float(self.att), "se": float(self.se)})
        return pd.DataFrame(rows)

    def to_markdown(self, **kwargs) -> str:
        """Export the per-cohort / aggregate table to Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export the per-cohort / aggregate table to a LaTeX ``tabular``."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export the per-cohort / aggregate table to a Typst ``#table``."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=False, **kwargs)

    def plot(self, *, ax=None, title: str = "Multi-cohort Synthetic DiD"):
        """Plot per-cohort SDID ATTs against cohort time, with the
        cohort-weighted aggregate (±1.96·se band) as a horizontal line.

        Returns the matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
        else:
            fig = ax.figure
        x = np.asarray(self.cohort_times, dtype=float)
        y = np.asarray(self.cohort_atts, dtype=float)
        w = np.asarray(self.cohort_weights, dtype=float)
        sizes = 40.0 + 160.0 * (w / w.max() if w.size and w.max() > 0 else w)
        ax.scatter(x, y, s=sizes, color="0.0", zorder=3,
                   label="Cohort ATT (marker ∝ weight)")
        ax.axhline(self.att, color="0.3", lw=1.2, label="Aggregate ATT")
        if np.isfinite(self.se):
            ax.axhspan(self.att - 1.96 * self.se, self.att + 1.96 * self.se,
                       color="0.8", alpha=0.5, label="95% band (bootstrap)")
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("Cohort (first-treatment period)")
        ax.set_ylabel("ATT")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False, fontsize=8)
        return fig
