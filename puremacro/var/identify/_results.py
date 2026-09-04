"""Frozen-dataclass result objects for var.identify estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class _IRFPlotMixin:
    """Mixin providing .plot() to IRF result objects."""

    def plot(
        self,
        *,
        target_idx: int = 0,
        shock_idx: int = 0,
        title: str = "",
        ylabel: str = "Response",
        scale: float = 1.0,
        ax=None,
    ):
        """Plot impulse response with error bands.

        Lazily delegates to puremacro.plot.plot_irf_single.
        """
        from ...plot import plot_irf_single

        return plot_irf_single(
            self,
            target_idx=target_idx,
            shock_idx=shock_idx,
            title=title,
            ylabel=ylabel,
            scale=scale,
            ax=ax,
        )

    def to_frame(
        self,
        *,
        target_idx: int | None = None,
        shock_idx: int | None = None,
    ):
        """Return a tidy pandas DataFrame of IRF estimates and bands."""
        import pandas as pd

        point = getattr(self, "irf_point", getattr(self, "irf_median", None))
        lower = getattr(self, "irf_lower", None)
        upper = getattr(self, "irf_upper", None)
        if point is None:
            raise ValueError("Result object does not contain impulse response matrices.")

        H_plus, n_resp, n_shock = point.shape
        rows = []
        for h in range(H_plus):
            if target_idx is not None and shock_idx is not None:
                row = {
                    "h": h,
                    "point": float(point[h, target_idx, shock_idx]),
                }
                if lower is not None:
                    row["lower"] = float(lower[h, target_idx, shock_idx])
                if upper is not None:
                    row["upper"] = float(upper[h, target_idx, shock_idx])
                rows.append(row)
            else:
                for r in range(n_resp):
                    for s in range(n_shock):
                        row = {
                            "h": h,
                            "response": r,
                            "shock": s,
                            "point": float(point[h, r, s]),
                        }
                        if lower is not None:
                            row["lower"] = float(lower[h, r, s])
                        if upper is not None:
                            row["upper"] = float(upper[h, r, s])
                        rows.append(row)
        return pd.DataFrame(rows)

    def to_markdown(self, *, target_idx: int = 0, shock_idx: int = 0) -> str:
        """Render IRF path for target_idx and shock_idx as Markdown."""
        from ...reports import _df_to_markdown

        return _df_to_markdown(
            self.to_frame(target_idx=target_idx, shock_idx=shock_idx), index=False
        )

    def to_latex(self, *, target_idx: int = 0, shock_idx: int = 0) -> str:
        """Render IRF path for target_idx and shock_idx as LaTeX."""
        from ...reports import _df_to_latex

        return _df_to_latex(
            self.to_frame(target_idx=target_idx, shock_idx=shock_idx), index=False
        )

    def to_typst(self, *, target_idx: int = 0, shock_idx: int = 0) -> str:
        """Render IRF path for target_idx and shock_idx as Typst."""
        from ...reports import _df_to_typst

        return _df_to_typst(
            self.to_frame(target_idx=target_idx, shock_idx=shock_idx), index=False
        )


@dataclass(frozen=True)
class ProxySVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.proxy.proxy_svar`.

    Attributes
    ----------
    irf_point : ndarray, shape (H+1, n, n)
        Point-estimate impulse responses.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower bootstrap band.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper bootstrap band.
    B : ndarray, shape (n, n)
        Identified structural impact matrix; column 0 is the proxy-identified shock.
    first_stage_F : float
        Olea-Pflueger (2013) effective F-statistic on the proxy.
    n_boot : int
        Number of bootstrap draws used.
    ci : float
        Confidence-interval level (e.g., 0.9 for 90% bands).
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    B: np.ndarray
    first_stage_F: float
    n_boot: int
    ci: float

    def summary(self) -> str:
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
        f_flag = "STRONG" if self.first_stage_F > 23.0 else "WEAK (use weak-IV-robust inference)"
        return (
            f"ProxySVAR result\n"
            f"  shocks identified : 1 (column 0 of B)\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  bootstrap draws   : {self.n_boot}\n"
            f"  CI level          : {self.ci:.2f}\n"
            f"  first-stage F (OP): {self.first_stage_F:.2f}  [{f_flag}]\n"
        )

    def __getitem__(self, item):
        target_irf = self.irf_point[:, :, 0]
        return target_irf[item]


@dataclass(frozen=True)
class CholeskySVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.cholesky.cholesky_svar`.

    Attributes
    ----------
    irf_point : ndarray, shape (H+1, n, n)
        Point-estimate impulse responses (response of variable i to shock j
        at horizon h is ``irf_point[h, i, j]``).
    irf_lower : ndarray, shape (H+1, n, n)
        Lower bootstrap percentile band.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper bootstrap percentile band.
    n_boot : int
        Number of bootstrap draws requested.
    n_fail : int
        Number of bootstrap draws dropped (non-PD reduced-form Σ).
    ci : float
        Confidence-interval level (e.g., 0.9 for 90% bands).
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    n_boot: int
    n_fail: int
    ci: float

    def summary(self) -> str:
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
        rate = self.n_fail / max(self.n_boot, 1)
        return (
            f"Cholesky SVAR result\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  CI level          : {self.ci:.2f}\n"
            f"  bootstrap draws   : {self.n_boot} (dropped {self.n_fail}, {rate:.1%})\n"
        )


@dataclass(frozen=True)
class BQSVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.bq.bq_svar`.

    Long-run-restriction SVAR (Blanchard-Quah 1989). The IRFs are
    cumulated along the horizon axis, so ``irf_point[h]`` represents
    the *level* response of each variable at horizon ``h``.

    Attributes
    ----------
    irf_point : ndarray, shape (H+1, n, n)
        Cumulated point-estimate impulse responses.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower bootstrap percentile band (cumulated).
    irf_upper : ndarray, shape (H+1, n, n)
        Upper bootstrap percentile band (cumulated).
    n_boot : int
        Number of bootstrap draws requested.
    n_fail : int
        Number of bootstrap draws dropped (non-PD long-run Ω).
    ci : float
        Confidence-interval level.

    References
    ----------
    Blanchard, O.J. and Quah, D. (1989). The dynamic effects of
        aggregate demand and supply disturbances. AER 79(4), 655-673.
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    n_boot: int
    n_fail: int
    ci: float

    def summary(self) -> str:
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
        rate = self.n_fail / max(self.n_boot, 1)
        return (
            f"Blanchard-Quah SVAR result\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  CI level          : {self.ci:.2f}\n"
            f"  bootstrap draws   : {self.n_boot} (dropped {self.n_fail}, {rate:.1%})\n"
        )


@dataclass(frozen=True)
class SignRestrictionResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.sign.sign_restriction_svar`.

    Sign-restriction SVAR per Rubio-Ramirez-Waggoner-Zha (2010). The
    central tendency reported is the median over admissible Q draws,
    not a point estimate.

    Attributes
    ----------
    irf_median : ndarray, shape (H+1, n, n)
        Median impulse response across admissible draws.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower posterior quantile across admissible draws.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper posterior quantile across admissible draws.
    n_draws : int
        Total Haar Q draws attempted.
    n_accepted : int
        Number of draws satisfying all sign restrictions.
    ci : float
        Confidence-interval level.

    References
    ----------
    Rubio-Ramírez, J.F., Waggoner, D.F. and Zha, T. (2010). Structural
        vector autoregressions: theory of identification and algorithms
        for inference. RES 77(2), 665-696.
    """

    irf_median: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    n_draws: int
    n_accepted: int
    ci: float

    def summary(self) -> str:
        H = self.irf_median.shape[0] - 1
        n = self.irf_median.shape[1]
        rate = self.n_accepted / max(self.n_draws, 1)
        return (
            f"Sign-restriction SVAR result\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  CI level          : {self.ci:.2f}\n"
            f"  Q draws           : {self.n_draws} ({self.n_accepted} accepted, {rate:.1%})\n"
        )


@dataclass(frozen=True)
class NarrativeSignSVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.narrative_sign.narrative_sign_svar`.

    Sign-restriction SVAR sharpened with narrative restrictions per
    Antolín-Díaz & Rubio-Ramírez (2018). Bands are pointwise *weighted*
    percentiles across the narrative-accepted Haar draws, where each
    draw's weight is the AD-RR importance weight ``1/omega`` (inverse
    probability that the narrative restrictions hold when the shocks on
    the restricted dates are redrawn i.i.d. standard normal).

    Attributes
    ----------
    irf_median : ndarray, shape (H+1, n, n)
        Weighted median impulse response across narrative-accepted draws.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower weighted percentile band.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper weighted percentile band.
    n_draws : int
        Total Haar Q draws attempted.
    n_traditional_accepted : int
        Draws satisfying the traditional sign restrictions.
    n_narrative_accepted : int
        Draws additionally satisfying every narrative restriction
        (these carry the weights and the bands).
    weights : ndarray, shape (n_narrative_accepted,)
        Raw AD-RR importance weights ``1/omega_hat`` per surviving draw
        (unnormalized; constant ``2**m`` when all m restrictions are
        Type I shock-sign restrictions on distinct (date, shock) pairs).
    ess : float
        Kish effective sample size of the importance weights,
        ``(sum w)^2 / sum(w^2)``; ``ess << n_narrative_accepted`` warns
        that a few draws dominate the weighted bands.
    ci : float
        Pointwise band coverage level.
    restriction_labels : tuple of str
        Human-readable label per narrative restriction (input order).
    restriction_fail_counts : tuple of int
        Per-restriction count of traditionally-accepted draws on which
        the restriction failed — the binding-ness diagnostic.

    References
    ----------
    Antolín-Díaz, J. and Rubio-Ramírez, J.F. (2018). Narrative sign
        restrictions for SVARs. AER 108(10), 2802-2829.
    """

    irf_median: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    n_draws: int
    n_traditional_accepted: int
    n_narrative_accepted: int
    weights: np.ndarray
    ess: float
    ci: float
    restriction_labels: tuple
    restriction_fail_counts: tuple

    def summary(self) -> str:
        H = self.irf_median.shape[0] - 1
        n = self.irf_median.shape[1]
        trad_rate = self.n_traditional_accepted / max(self.n_draws, 1)
        narr_rate = self.n_narrative_accepted / max(self.n_traditional_accepted, 1)
        lines = [
            "Narrative-sign SVAR result (AD-RR 2018)",
            f"  variables (n)     : {n}",
            f"  horizon (H)       : {H}",
            f"  CI level          : {self.ci:.2f}",
            f"  Q draws           : {self.n_draws}",
            f"  traditional accept: {self.n_traditional_accepted} ({trad_rate:.1%})",
            f"  narrative accept  : {self.n_narrative_accepted} ({narr_rate:.1%} of traditional)",
            f"  weight ESS        : {self.ess:.1f} / {self.n_narrative_accepted}",
        ]
        for lab, fails in zip(self.restriction_labels, self.restriction_fail_counts):
            lines.append(f"    - {lab}: failed {fails}/{self.n_traditional_accepted}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class GKRobustBandsResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.sign_robust.gk_robust_bands`
    and :func:`puremacro.var.identify.sign_robust.gk_robust_bands_from_gibbs`.

    Giacomini-Kitagawa (2021) robust bands separate VAR uncertainty from
    set-identification uncertainty for sign-restricted SVARs.

    Attributes
    ----------
    irf_lower : ndarray, shape (H+1, n)
        Lower endpoint of the identified set, per-variable response
        to the target shock.
    irf_upper : ndarray, shape (H+1, n)
        Upper endpoint of the identified set.
    irf_median : ndarray, shape (H+1, n)
        Within-set median (across admissible Q at each VAR draw).
    n_accepted_per_draw : ndarray, shape (n_var_draws,)
        Number of admissible Q draws at each VAR estimate.

    References
    ----------
    Giacomini, R. and Kitagawa, T. (2021). Robust Bayesian inference
        for set-identified models. Econometrica 89(4), 1519-1556.
    """

    irf_lower: np.ndarray
    irf_upper: np.ndarray
    irf_median: np.ndarray
    n_accepted_per_draw: np.ndarray

    def summary(self) -> str:
        H = self.irf_median.shape[0] - 1
        n = self.irf_median.shape[1]
        n_draws = len(self.n_accepted_per_draw)
        mean_acc = float(self.n_accepted_per_draw.mean())
        return (
            f"GK robust bands result\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  VAR draws         : {n_draws}\n"
            f"  mean Q acceptance : {mean_acc:.1f} per VAR draw\n"
        )


@dataclass(frozen=True)
class NonGaussianSVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.non_gaussian.non_gaussian_svar`.

    LMS (2017) non-Gaussian SVAR via FastICA on reduced-form residuals.

    Attributes
    ----------
    B0 : ndarray, shape (n, n)
        Identified impact matrix, columns ordered by descending |excess
        kurtosis| so the most non-Gaussian shock is column 0.
    Q : ndarray, shape (n, n)
        Orthogonal rotation, ``B0 = chol(Σ) @ Q``.
    kurtosis : ndarray, shape (n,)
        Excess kurtosis of recovered shocks (already sorted).
    irf : ndarray, shape (H+1, n, n)
        Structural impulse responses.
    ordering_by_kurt : ndarray, shape (n,)
        Permutation from original ICA columns to final structural-shock columns
        (composite of the kurtosis sort + any tiebreak refinement).
    lr_test : dict or None
        New in 0.51.0. Result of :func:`gaussian_lr_test` against the
        Gaussian baseline. Keys: ``stat``, ``df``, ``p_value``.
    consistency_check : dict or None
        New in 0.51.0. Result of :func:`variance_decomposition_consistency`.
        Keys: ``max_abs_diff``, ``rms_diff``, ``passed``.

    References
    ----------
    Lanne, M., Meitz, M. and Saikkonen, P. (2017). Identification and
        estimation of non-Gaussian structural vector autoregressions.
        Journal of Econometrics 196(2), 288-304.
    """

    B0: np.ndarray
    Q: np.ndarray
    kurtosis: np.ndarray
    irf: np.ndarray
    ordering_by_kurt: np.ndarray
    lr_test: Optional[dict] = None
    consistency_check: Optional[dict] = None

    def summary(self) -> str:
        n = self.B0.shape[0]
        H = self.irf.shape[0] - 1
        kurt_str = ", ".join(f"{k:+.2f}" for k in self.kurtosis)
        lines = [
            f"Non-Gaussian SVAR (LMS 2017) result",
            f"  variables (n)     : {n}",
            f"  horizon (H)       : {H}",
            f"  shock kurtosis    : [{kurt_str}]",
        ]
        if self.lr_test is not None:
            lines.append(f"  LR vs Gaussian    : stat={self.lr_test['stat']:.2f}, p={self.lr_test['p_value']:.3f}")
        if self.consistency_check is not None:
            lines.append(f"  B0·B0' vs Σ_u     : max_abs_diff={self.consistency_check['max_abs_diff']:.2e}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SignZeroResult:
    """Result of :func:`puremacro.var.identify.sign_zero.sign_zero`.

    Arias-Rubio Ramírez-Waggoner (2018) sign + zero restrictions SVAR.
    Always returned (no None case): inspect ``success`` to know whether
    a draw satisfying every constraint was found.

    Attributes
    ----------
    success : bool
        ``True`` iff at least one Haar draw satisfied all constraints.
    B0 : ndarray of shape (n, n) or None
        Identified impact matrix; ``None`` when ``success=False``.
    Q : ndarray of shape (n, n) or None
        Orthogonal rotation, ``B0 = chol(Σ) @ Q``; ``None`` on failure.
    n_draws_used : int
        Number of Haar draws attempted before success or exhaustion.

    References
    ----------
    Arias, J., Rubio-Ramírez, J. and Waggoner, D. (2018). Inference
        based on structural vector autoregressions identified with sign
        and zero restrictions: theory and applications. Econometrica
        86(2), 685-720.
    """

    success: bool
    B0: Optional[np.ndarray]
    Q: Optional[np.ndarray]
    n_draws_used: int

    def summary(self) -> str:
        if self.success and self.B0 is not None:
            n = self.B0.shape[0]
            return (
                f"Sign-zero SVAR result\n"
                f"  status            : SUCCESS\n"
                f"  variables (n)     : {n}\n"
                f"  draws used        : {self.n_draws_used}\n"
            )
        return (
            f"Sign-zero SVAR result\n"
            f"  status            : FAILED (no admissible draw)\n"
            f"  draws used        : {self.n_draws_used}\n"
        )


@dataclass(frozen=True)
class PanelSVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.panel.mean_group_svar`.

    Canova-Ciccarelli (2013) mean-group panel SVAR. Each country
    estimates its own SVAR(p); IRFs are averaged across countries
    (mean-group estimator) with cross-country percentile bands.

    Attributes
    ----------
    irf_mean : ndarray, shape (H+1, n, n)
        Mean-group IRF: simple cross-country average.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower percentile band from cross-country distribution.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper percentile band.
    country_irfs : ndarray, shape (N, H+1, n, n)
        Stacked country-level IRFs.
    country_ids : tuple of str
        Country identifiers in stacking order.
    identification : str
        Identification scheme used ('cholesky' or 'bq').
    p : int
        VAR lag order.
    horizon : int
        IRF horizon H.
    ci : float
        Coverage level for percentile bands.

    References
    ----------
    Canova, F. and Ciccarelli, M. (2013). Panel vector autoregressive
        models: a survey. Advances in Econometrics 32, 205-246.
    """
    irf_mean: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    country_irfs: np.ndarray
    country_ids: tuple
    identification: str
    p: int
    horizon: int
    ci: float

    def summary(self) -> str:
        n = self.irf_mean.shape[1]
        H = self.irf_mean.shape[0] - 1
        N = len(self.country_ids)
        return (
            f"Panel SVAR (mean-group, {self.identification})\n"
            f"  countries (N)     : {N}\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  lag order (p)     : {self.p}\n"
            f"  CI level          : {self.ci:.2f}\n"
        )


@dataclass(frozen=True)
class MaxShareResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.maxshare.identify_maxshare`.

    Faust-Uhlig (2003) max-share identification: pick the structural
    shock whose contribution to the target variable's forecast-error
    variance at horizon ``max_fev_at`` is maximised.

    Attributes
    ----------
    B : ndarray, shape (n, n)
        Full structural impact matrix. Column 0 is the max-share shock;
        columns 1..n-1 are orthogonal completions (no structural label).
    q : ndarray, shape (n,)
        Optimal unit vector, ``B[:, 0] = chol(Sigma_u) @ q``.
    fev_share_at_target : float
        Achieved FEV share at horizon ``max_fev_at`` for the target
        variable. In ``[0, 1]``.
    irfs : ndarray, shape (H+1, n, n)
        Structural impulse responses.
    fevd : ndarray, shape (H+1, n, n)
        Forecast-error variance decomposition.
    max_fev_at : int
        Horizon used for identification.
    irf_lower : ndarray or None, shape (H+1, n, n)
        Lower bootstrap band; ``None`` if ``n_bootstrap == 0``.
    irf_upper : ndarray or None, shape (H+1, n, n)
        Upper bootstrap band; ``None`` if ``n_bootstrap == 0``.
    ci : float
        Coverage for bootstrap bands (e.g. 0.68 for 68% bands).

    References
    ----------
    Faust, J. (1998). The robustness of identified VAR conclusions
        about money. Carnegie-Rochester Conf. Series 49, 207-244.
    Uhlig, H. (2003). What moves real GNP? unpublished.
    """
    B: np.ndarray
    q: np.ndarray
    fev_share_at_target: float
    irfs: np.ndarray
    fevd: np.ndarray
    max_fev_at: int
    irf_lower: Optional[np.ndarray]
    irf_upper: Optional[np.ndarray]
    ci: float

    def summary(self) -> str:
        n = self.B.shape[0]
        H = self.irfs.shape[0] - 1
        band_str = ("no bootstrap" if self.irf_lower is None
                    else f"CI {self.ci:.2f}")
        return (
            f"Max-share SVAR (Faust-Uhlig)\n"
            f"  variables (n)        : {n}\n"
            f"  horizon (H)          : {H}\n"
            f"  FEV target horizon   : {self.max_fev_at}\n"
            f"  FEV share at target  : {self.fev_share_at_target:.4f}\n"
            f"  bands                : {band_str}\n"
        )


@dataclass(frozen=True)
class MagMavSVARResult(_IRFPlotMixin):
    """Result of :func:`puremacro.var.identify.magmav.magmav_svar`.

    Magnusson-Mavroeidis (2014) SVAR identified by continuous time-varying
    structural-shock variance; break dates discovered endogenously by
    sup-Wald + BIC.

    Attributes
    ----------
    irf_point : ndarray, shape (H+1, n, n)
        Point-estimate impulse responses.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower bootstrap band.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper bootstrap band.
    B : ndarray, shape (n, n)
        Identified structural impact matrix. Columns ordered by descending
        cross-regime variance ratio max_g D_g[j,j] / min_g D_g[j,j].
    variance_change_dates : tuple of int
        Break dates (residual-row indices) selected by sup-Wald + BIC.
    k_breaks : int
        Number of breaks selected (0 if BIC chose homoskedastic baseline).
    n_boot : int
        Number of bootstrap draws requested.
    ci : float
        Bootstrap CI level.
    eu : tuple of int
        Convergence proxy: (1, 1) iff the B-matrix optimiser declared
        scipy.optimize success; (0, 0) iff k_breaks selected 0 (no
        heteroskedasticity-based identification) OR the optimiser did
        not converge. This is NOT the full Magnusson-Mavroeidis
        existence/uniqueness test from the paper.
    n_fail : int
        Bootstrap draws that failed to converge.

    References
    ----------
    Magnusson, L.M. and Mavroeidis, S. (2014). Identification using
        stability restrictions. Econometrica 82(5), 1799-1851.
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    B: np.ndarray
    variance_change_dates: tuple
    k_breaks: int
    n_boot: int
    ci: float
    eu: tuple[int, int]
    n_fail: int

    def summary(self) -> str:
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
        flag = "OK" if self.eu == (1, 1) else f"FAIL (eu={self.eu})"
        rate = self.n_fail / max(self.n_boot, 1)
        return (
            f"Magnusson-Mavroeidis SVAR result\n"
            f"  variables (n)        : {n}\n"
            f"  horizon (H)          : {H}\n"
            f"  break dates          : {self.variance_change_dates}\n"
            f"  k_breaks             : {self.k_breaks}\n"
            f"  identification       : {flag}\n"
            f"  bootstrap draws      : {self.n_boot} (dropped {self.n_fail}, {rate:.1%}), CI {self.ci:.2f}\n"
        )
