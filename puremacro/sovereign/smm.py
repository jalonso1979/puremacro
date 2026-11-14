"""Simulated Method of Moments (SMM) & Minimum-Distance Calibration for Fiscal Sovereign Default.

Bridges :func:`puremacro.sovereign.solve_fiscal_sovereign_default` with
:func:`puremacro.structural.bridge.fit_structural` and
:func:`puremacro.structural.bridge.ags_sensitivity` (Andrews, Gentzkow & Shapiro 2017).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.sovereign._fiscal_paths import draw_uniforms, ergodic_moments, simulate_paths
from puremacro.sovereign._results import FiscalSovereignResult
from puremacro.sovereign.fiscal import solve_fiscal_sovereign_default
from puremacro.sovereign.rules import FiscalRule
from puremacro.structural.bridge import (
    AGSSensitivityResult,
    MomentTargets,
    StructuralFitResult,
    ags_sensitivity,
    fit_structural,
)

__all__ = [
    "SovereignSMMResult",
    "calibrate_fiscal_sovereign_smm",
]


@dataclass(frozen=True)
class SovereignSMMResult:
    """Result container for SMM / minimum-distance calibration of the fiscal sovereign default model.

    Parameters
    ----------
    fit : StructuralFitResult
        Minimum-distance estimation result from :func:`puremacro.structural.bridge.fit_structural`.
    sensitivity : AGSSensitivityResult
        Andrews, Gentzkow & Shapiro (2017) sensitivity matrix ``Lambda`` and standardized ``Lambda_tilde``.
    model : FiscalSovereignResult
        Solved :class:`FiscalSovereignResult` evaluated at the calibrated parameter vector ``fit.theta``.
    parameter_dict : Mapping[str, float]
        Dictionary mapping calibrated parameter names to their point estimates.
    metadata : Mapping[str, Any]
        Calibration settings and diagnostic metadata.
    """

    fit: StructuralFitResult
    sensitivity: AGSSensitivityResult
    model: FiscalSovereignResult
    parameter_dict: Mapping[str, float]
    metadata: Mapping[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Return parameter estimates, standard errors, and top identifying moments."""
        df = self.fit.summary().copy()
        top_map = self.sensitivity.metadata.get("top_identifying_moment", {})
        df["top_identifying_moment"] = [top_map.get(p, "") for p in self.fit.parameter_names]
        return df

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> str:
        """Return a formatted ASCII summary of the SMM calibration and AGS sensitivity."""
        param_str = self.to_dataframe().to_string(index=False, float_format=lambda x: f"{x:.4f}")
        mom_str = self.fit.moment_fit().to_string(index=False, float_format=lambda x: f"{x:.4f}")
        lines = [
            "==========================================================================",
            "Fiscal Sovereign Default SMM / Minimum-Distance Calibration",
            "==========================================================================",
            f"Success: {self.fit.success} | Inference Valid: {self.fit.inference_valid} | "
            f"Objective Q(theta): {self.fit.objective:.6f}",
            f"Condition Number: {self.fit.condition_number:.2e} | Function Evals: {self.fit.n_evals}",
            "--------------------------------------------------------------------------",
            "Calibrated Structural Parameters:",
            param_str,
            "--------------------------------------------------------------------------",
            "Target vs. Model Moments:",
            mom_str,
            "--------------------------------------------------------------------------",
            self.sensitivity.summary(),
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Export calibration parameter table to GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export calibration parameter table to LaTeX booktabs format."""
        return df_to_latex(self.to_dataframe(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export calibration parameter table to Typst format."""
        return df_to_typst(self.to_dataframe(), index=False, **kwargs)

    def plot(self, *, ax: Any = None):
        """Plot a 2-panel comparison of Target vs. Model Moments and AGS Sensitivity Heatmap."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout="constrained")
        else:
            fig = ax.figure
            axes = [ax, ax]

        ax0 = axes[0]
        mfit = self.fit.moment_fit()
        x = np.arange(len(mfit))
        width = 0.36
        ax0.bar(
            x - width / 2,
            mfit["target"],
            width,
            yerr=1.96 * mfit["target_se"],
            label="Empirical Target (±1.96 SE)",
            color="#1f77b4",
            alpha=0.85,
            capsize=3,
        )
        ax0.bar(
            x + width / 2,
            mfit["model"],
            width,
            label="Calibrated Model",
            color="#ff7f0e",
            alpha=0.85,
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels(mfit["label"], rotation=25, ha="right")
        ax0.set_title("Empirical vs. Calibrated Model Moments")
        ax0.legend(frameon=False)
        ax0.grid(axis="y", alpha=0.25)

        if len(axes) > 1 and axes[1] is not axes[0]:
            self.sensitivity.plot(ax=axes[1])
        return fig


_SUPPORTED_MOMENTS: tuple[str, ...] = (
    "debt_pct",
    "spread_mean_pct",
    "annual_default_pct",
    "spread_std_pp",
    "spending_pct",
    "corr_pb_y",
    "boundary_choice_share_pct",
)

_REPAYMENT_MOMENTS: frozenset[str] = frozenset(
    {
        "debt_pct",
        "spread_mean_pct",
        "spread_std_pp",
        "corr_pb_y",
        "boundary_choice_share_pct",
    }
)

_CONTINUOUS_SOLVER_PARAMS: frozenset[str] = frozenset(
    {
        "tau",
        "psi",
        "gamma_c",
        "gamma_g",
        "beta",
        "r_star",
        "theta",
        "rho_y",
        "sigma_y",
        "b_min",
        "b_max",
        "sigma_eps",
        "d0",
        "d1",
        "m_tauchen",
        "damping",
        "recovery_rate",
    }
)


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of moment names, not str or bytes")
    lbls = tuple(labels)
    if not lbls or any(not isinstance(lbl, str) or not lbl.strip() for lbl in lbls):
        raise ValueError("labels must contain at least one nonempty string")
    unknown = [lbl for lbl in lbls if lbl not in _SUPPORTED_MOMENTS]
    if unknown:
        raise ValueError(
            f"Unsupported moment label(s): {unknown}; supported labels are {_SUPPORTED_MOMENTS}"
        )
    return lbls


def _is_int_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_simulation_args(
    seeds: Sequence[int],
    periods: int,
    burn: int,
) -> tuple[list[int], int, int]:
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
    seed_seq = list(seeds)
    if not seed_seq:
        raise ValueError("seeds must be nonempty when use_simulation=True")
    clean_seeds: list[int] = []
    for s in seed_seq:
        if not _is_int_like(s) or int(s) < 0:
            raise ValueError(f"Each seed must be a nonnegative integer; got {s!r}")
        clean_seeds.append(int(s))
    if not _is_int_like(periods) or int(periods) < 1:
        raise ValueError(f"periods must be an integer >= 1; got {periods!r}")
    if not _is_int_like(burn) or not (0 <= int(burn) < int(periods)):
        raise ValueError(f"burn must be an integer in [0, {int(periods) - 1}]; got {burn!r}")
    return clean_seeds, int(periods), int(burn)


def _validate_fiscal_solution(
    sol: FiscalSovereignResult,
    *,
    need_y_def: bool,
    require_metadata_zero: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray | None,
]:
    """Validate solved fiscal arrays, policy consistency, and exclusion metadata."""
    if not isinstance(sol, FiscalSovereignResult):
        raise TypeError(f"sol must be a FiscalSovereignResult; got {type(sol).__name__}")
    if sol.metadata.get("debt_choice_probabilities") is not None:
        raise ValueError(
            "Fiscal sovereign SMM requires discrete short-term policy_index; "
            "smoothed debt_choice_probabilities are not supported by _fiscal_paths"
        )

    p_def = np.asarray(sol.default_prob, dtype=float)
    if p_def.ndim != 2 or p_def.shape[0] < 2 or p_def.shape[1] < 2:
        raise ValueError(f"sol.default_prob must be a 2D array with shape >= (2, 2); got {p_def.shape}")
    if not np.all(np.isfinite(p_def)) or np.any((p_def < 0.0) | (p_def > 1.0)):
        raise ValueError("sol.default_prob must be finite and in [0, 1]")
    n_b, n_y = p_def.shape

    b_grid = np.asarray(sol.b_grid, dtype=float)
    if b_grid.shape != (n_b,) or not np.all(np.isfinite(b_grid)) or not np.all(np.diff(b_grid) > 0.0):
        raise ValueError(f"sol.b_grid must be finite and strictly increasing of length {n_b}")

    y_grid = np.asarray(sol.y_grid, dtype=float)
    if (
        y_grid.shape != (n_y,)
        or not np.all(np.isfinite(y_grid))
        or np.any(y_grid <= 0.0)
        or not np.all(np.diff(y_grid) > 0.0)
    ):
        raise ValueError(f"sol.y_grid must be finite, positive, and strictly increasing of length {n_y}")

    p_trans = np.asarray(sol.p_trans, dtype=float)
    if p_trans.shape != (n_y, n_y) or not np.all(np.isfinite(p_trans)) or np.any(p_trans < 0.0):
        raise ValueError(f"sol.p_trans must be a finite nonnegative ({n_y}, {n_y}) transition matrix")
    row_sums = p_trans.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("sol.p_trans rows must sum to 1")
    p_trans = p_trans / row_sums[:, None]

    pol_idx_arr = np.asarray(sol.policy_index)
    if pol_idx_arr.shape != (n_b, n_y) or pol_idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"sol.policy_index must be an integer array of shape ({n_b}, {n_y})")
    b_next_idx = pol_idx_arr.astype(int)
    if np.any((b_next_idx < 0) | (b_next_idx >= n_b)):
        raise ValueError(f"sol.policy_index entries must lie in [0, {n_b - 1}]")

    pol_b = np.asarray(sol.policy_b, dtype=float)
    if pol_b.size > 0:
        if pol_b.shape != (n_b, n_y) or not np.allclose(b_grid[b_next_idx], pol_b, rtol=1e-10, atol=1e-12):
            raise ValueError("sol.policy_b is inconsistent with sol.b_grid[sol.policy_index]")

    for arr_name, expected_shape in (
        ("debt_to_gdp", (n_b, n_y)),
        ("spread", (n_b, n_y)),
        ("g_policy", (n_b, n_y)),
        ("primary_balance", (n_b, n_y)),
        ("g_default", (n_y,)),
    ):
        arr = np.asarray(getattr(sol, arr_name), dtype=float)
        if arr.shape != expected_shape:
            raise ValueError(f"sol.{arr_name} must have shape {expected_shape}; got {arr.shape}")

    if "theta" not in sol.metadata:
        raise ValueError("sol.metadata['theta'] is required for exclusion/reentry dynamics")
    raw_theta = sol.metadata["theta"]
    if isinstance(raw_theta, (bool, np.bool_)) or not isinstance(raw_theta, (int, float, np.number)):
        raise ValueError(f"sol.metadata['theta'] must be a real scalar in [0, 1]; got {raw_theta!r}")
    theta = float(raw_theta)
    if not np.isfinite(theta) or not (0.0 <= theta <= 1.0):
        raise ValueError(f"sol.metadata['theta'] must be finite and in [0, 1]; got {theta}")

    if "zero_index" in sol.metadata:
        raw_zero = sol.metadata["zero_index"]
        if not _is_int_like(raw_zero) or not (0 <= int(raw_zero) < n_b):
            raise ValueError(f"sol.metadata['zero_index'] must be an integer in [0, {n_b - 1}]; got {raw_zero!r}")
        zero_idx = int(raw_zero)
    elif require_metadata_zero:
        raise ValueError("sol.metadata['zero_index'] is required for path simulation")
    else:
        zeros = np.flatnonzero(b_grid == 0.0)
        if zeros.size != 1:
            raise ValueError("sol.b_grid must contain a unique exact zero-debt node when zero_index is omitted")
        zero_idx = int(zeros[0])

    y_def: np.ndarray | None = None
    if "y_default_vec" in sol.metadata:
        y_def = np.asarray(sol.metadata["y_default_vec"], dtype=float)
        if y_def.shape != (n_y,) or not np.all(np.isfinite(y_def)) or np.any(y_def <= 0.0):
            raise ValueError(f"sol.metadata['y_default_vec'] must be finite and positive of length {n_y}")
    elif need_y_def:
        raise ValueError("sol.metadata['y_default_vec'] is required for default-period spending and simulation")

    return p_def, b_grid, y_grid, p_trans, b_next_idx, theta, zero_idx, y_def


def _stationary_access_exclusion_dist(
    p_def: np.ndarray,
    b_next_idx: np.ndarray,
    p_trans: np.ndarray,
    theta: float,
    zero_idx: int,
    *,
    tol: float = 1e-13,
    max_iter: int = 50_000,
    damping: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stationary pre-decision weights ``(w_acc, w_exc)`` matching ``_fiscal_paths``.

    Timing matches :func:`puremacro.sovereign._fiscal_paths.simulate_paths`:
    - ``w_acc[b, y]`` is the mass entering the current period's default/repayment
      decision with market access at debt ``b`` and endowment ``y``.
    - ``w_exc[y]`` is the mass that remained excluded after the start-of-period
      reentry draw (probability ``theta``) and therefore produces default output
      and zero primary balance/debt in the current period.
    """
    n_b, n_y = p_def.shape
    rep_prob = 1.0 - p_def
    y_cols = np.arange(n_y)[None, :]

    # Start from the same initial state as simulate_paths so reducible/theta=0
    # chains stay in the reachable component from (zero_idx, n_y // 2).
    w_acc = np.zeros((n_b, n_y), dtype=float)
    w_acc[zero_idx, n_y // 2] = 1.0
    w_exc = np.zeros(n_y, dtype=float)

    residual = np.inf
    for _ in range(max_iter):
        w_rep_curr = w_acc * rep_prob
        w_out = w_exc + np.sum(w_acc * p_def, axis=0)

        t_acc = np.zeros((n_b, n_y), dtype=float)
        np.add.at(t_acc, (b_next_idx, y_cols), w_rep_curr)
        t_acc = t_acc @ p_trans

        w_out_next = w_out @ p_trans
        t_acc[zero_idx, :] += theta * w_out_next
        t_exc = (1.0 - theta) * w_out_next

        residual = max(
            float(np.max(np.abs(t_acc - w_acc))),
            float(np.max(np.abs(t_exc - w_exc))),
        )
        acc_curr_sum = float(np.sum(w_acc))
        acc_next_sum = float(np.sum(t_acc))
        acc_mass_settled = (acc_next_sum <= 1e-15) or (
            acc_curr_sum > 0.0 and abs(acc_next_sum - acc_curr_sum) <= tol * acc_curr_sum
        )
        if residual < tol and acc_mass_settled:
            total_t = acc_next_sum + float(np.sum(t_exc))
            return t_acc / total_t, t_exc / total_t

        w_acc_next = (1.0 - damping) * w_acc + damping * t_acc
        w_exc_next = (1.0 - damping) * w_exc + damping * t_exc
        total = float(np.sum(w_acc_next) + np.sum(w_exc_next))
        w_acc = w_acc_next / total
        w_exc = w_exc_next / total

    raise RuntimeError(
        f"Stationary (access/excluded, b, y) distribution did not converge within {max_iter} "
        f"iterations (residual={residual:.3e})."
    )


def _require_finite_masked(arr: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    vals = arr[mask]
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"Nonfinite values encountered in {name} on positive-weight stationary states")
    return vals


def _compute_stationary_or_simulated_moments(
    sol: FiscalSovereignResult,
    labels: Sequence[str],
    *,
    use_simulation: bool,
    seeds: Sequence[int],
    periods: int,
    burn: int,
) -> np.ndarray:
    """Compute model moments matching ``labels`` from a solved ``FiscalSovereignResult``."""
    lbls = _validate_labels(labels)
    need_y_def = bool(use_simulation or "spending_pct" in lbls)
    p_def, _, y_grid, p_trans, b_next_idx, theta, zero_idx, y_def = _validate_fiscal_solution(
        sol,
        need_y_def=need_y_def,
        require_metadata_zero=bool(use_simulation),
    )

    if use_simulation:
        clean_seeds, clean_periods, clean_burn = _validate_simulation_args(seeds, periods, burn)
        y0_idx = len(y_grid) // 2
        u_draws = draw_uniforms(clean_seeds, clean_periods)
        paths = simulate_paths(sol, zero_idx, y0_idx, u_draws)
        per_path = ergodic_moments(paths, burn=clean_burn)
        out = []
        for lbl in lbls:
            vals = [float(d[lbl]) for d in per_path]
            bad_seeds = [clean_seeds[k] for k, v in enumerate(vals) if not np.isfinite(v)]
            if bad_seeds:
                raise ValueError(
                    f"Simulated moment {lbl!r} is nonfinite for seed(s) {bad_seeds} "
                    f"({len(bad_seeds)}/{len(per_path)} path(s); e.g. absorbing exclusion or "
                    "insufficient repayment observations/variation after burn-in)."
                )
            out.append(float(np.mean(vals)))
        return np.asarray(out, dtype=float)

    n_b, n_y = p_def.shape
    w_acc, w_exc = _stationary_access_exclusion_dist(p_def, b_next_idx, p_trans, theta, zero_idx)

    rep_prob = 1.0 - p_def
    w_rep = w_acc * rep_prob
    w_rep_sum = float(np.sum(w_rep))
    w_acc_sum = float(np.sum(w_acc))

    lbl_set = set(lbls)
    need_repay = bool(lbl_set & _REPAYMENT_MOMENTS)
    if need_repay and w_rep_sum <= 1e-12:
        raise ValueError(
            "Stationary repayment probability mass is zero (economy absorbed into default/exclusion); "
            f"cannot compute repayment moment(s) {sorted(lbl_set & _REPAYMENT_MOMENTS)}."
        )

    w_rep_norm = w_rep / w_rep_sum if w_rep_sum > 1e-12 else np.zeros_like(w_rep)
    rep_mask = w_rep_norm > 0.0
    w_rep_weights = w_rep_norm[rep_mask]

    moment_map: dict[str, float] = {}

    if "debt_pct" in lbl_set:
        debt_vals = _require_finite_masked(np.asarray(sol.debt_to_gdp, dtype=float), rep_mask, "sol.debt_to_gdp")
        moment_map["debt_pct"] = 100.0 * float(np.sum(w_rep_weights * debt_vals))

    if "spread_mean_pct" in lbl_set or "spread_std_pp" in lbl_set:
        y_idx = np.broadcast_to(np.arange(n_y)[None, :], (n_b, n_y))
        chosen_spread = np.asarray(sol.spread, dtype=float)[b_next_idx, y_idx]
        spread_vals = _require_finite_masked(chosen_spread, rep_mask, "sol.spread")
        spread_mean_pct = 100.0 * float(np.sum(w_rep_weights * spread_vals))
        spread_var = float(np.sum(w_rep_weights * ((100.0 * spread_vals - spread_mean_pct) ** 2)))
        moment_map["spread_mean_pct"] = spread_mean_pct
        moment_map["spread_std_pp"] = float(np.sqrt(max(spread_var, 0.0)))

    if "annual_default_pct" in lbl_set:
        if w_acc_sum <= 1e-12:
            raise ValueError(
                "Stationary market-access probability mass is zero; 'annual_default_pct' is undefined."
            )
        moment_map["annual_default_pct"] = 400.0 * float(np.sum(w_acc * p_def)) / w_acc_sum

    if "spending_pct" in lbl_set:
        assert y_def is not None
        y_mat = np.broadcast_to(y_grid[None, :], (n_b, n_y))
        g_pol = np.asarray(sol.g_policy, dtype=float)
        g_def = np.asarray(sol.g_default, dtype=float)
        w_non_rep_y = w_exc + np.sum(w_acc * p_def, axis=0)
        non_rep_mask = w_non_rep_y > 0.0
        rep_spending = (
            float(np.sum(w_rep[rep_mask] * (_require_finite_masked(g_pol, rep_mask, "sol.g_policy") / y_mat[rep_mask])))
            if np.any(rep_mask)
            else 0.0
        )
        def_spending = (
            float(
                np.sum(
                    w_non_rep_y[non_rep_mask]
                    * (_require_finite_masked(g_def, non_rep_mask, "sol.g_default") / y_def[non_rep_mask])
                )
            )
            if np.any(non_rep_mask)
            else 0.0
        )
        moment_map["spending_pct"] = 100.0 * (rep_spending + def_spending)

    if "corr_pb_y" in lbl_set:
        y_mat = np.broadcast_to(y_grid[None, :], (n_b, n_y))
        pb_ratio = np.asarray(sol.primary_balance, dtype=float) / y_mat
        pb_vals = _require_finite_masked(pb_ratio, rep_mask, "sol.primary_balance")
        ly_vals = np.log(y_mat[rep_mask])
        mean_pb = float(np.sum(w_rep_weights * pb_vals))
        mean_ly = float(np.sum(w_rep_weights * ly_vals))
        var_pb = float(np.sum(w_rep_weights * ((pb_vals - mean_pb) ** 2)))
        var_ly = float(np.sum(w_rep_weights * ((ly_vals - mean_ly) ** 2)))
        std_pb = float(np.sqrt(max(var_pb, 0.0)))
        std_ly = float(np.sqrt(max(var_ly, 0.0)))
        if std_pb <= 1e-14 or std_ly <= 1e-14:
            raise ValueError(
                "Stationary moment 'corr_pb_y' is undefined because primary_balance/y "
                "or log(y) has zero variance on repayment states."
            )
        cov_pb_y = float(np.sum(w_rep_weights * (pb_vals - mean_pb) * (ly_vals - mean_ly)))
        moment_map["corr_pb_y"] = float(np.clip(cov_pb_y / (std_pb * std_ly), -1.0, 1.0))

    if "boundary_choice_share_pct" in lbl_set:
        moment_map["boundary_choice_share_pct"] = 100.0 * float(
            np.sum(w_rep_weights * (b_next_idx[rep_mask] == (n_b - 1)))
        )

    return np.asarray([moment_map[lbl] for lbl in lbls], dtype=float)


def calibrate_fiscal_sovereign_smm(
    targets: MomentTargets,
    *,
    param_names: Sequence[str] = ("beta", "d0"),
    theta0: Sequence[float] = (0.95, -0.188),
    bounds: Sequence[tuple[float | None, float | None]] | None = ((0.80, 0.99), (-0.45, -0.01)),
    tau: float = 0.20,
    psi: float = 1.0,
    n_b: int = 51,
    n_y: int = 9,
    sigma_eps: float = 0.002,
    rule: FiscalRule | None = None,
    use_simulation: bool = False,
    seeds: Sequence[int] = (1979, 1980, 1981, 1982),
    periods: int = 400,
    burn: int = 100,
    max_nfev: int = 25,
    tol: float = 1e-5,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> SovereignSMMResult:
    """Calibrate structural parameters of the fiscal sovereign default model via SMM / minimum distance.

    Wraps :func:`puremacro.sovereign.solve_fiscal_sovereign_default` into
    :func:`puremacro.structural.bridge.fit_structural` and automatically computes the
    Andrews, Gentzkow & Shapiro (2017) sensitivity matrix :func:`ags_sensitivity`.

    Parameters
    ----------
    targets : MomentTargets
        Empirical target moments and their covariance matrix. Supported moment labels:
        ``"debt_pct"``, ``"spread_mean_pct"``, ``"spread_std_pp"``, ``"annual_default_pct"``,
        ``"spending_pct"``, ``"corr_pb_y"``, and ``"boundary_choice_share_pct"``.
    param_names : Sequence[str], default ("beta", "d0")
        Names of continuous parameters in :func:`solve_fiscal_sovereign_default` to estimate.
    theta0 : Sequence[float], default (0.95, -0.188)
        Initial parameter guess matching ``param_names``.
    bounds : Sequence[tuple[float | None, float | None]] | None
        Box bounds for each parameter.
    tau, psi : float
        Baseline tax rate and fiscal utility weight (if not included in ``param_names``).
    n_b, n_y : int
        Debt and endowment grid sizes for fast structural evaluation.
    sigma_eps : float, default 0.002
        Extreme-value taste shock scale for smooth numerical Jacobians during optimization.
    rule : FiscalRule | None, default None
        Optional fiscal rule active during calibration.
    use_simulation : bool, default False
        If False (default), evaluates moments directly from the stationary distribution
        over ``(access/excluded, b, y)`` matching the exclusion/reentry Markov chain of
        ``_fiscal_paths.simulate_paths``. If True, simulates common-random-number paths.
    seeds : Sequence[int]
        Common random number seeds when ``use_simulation=True``.
    periods, burn : int
        Simulation horizon and burn-in periods when ``use_simulation=True``.
    max_nfev : int, default 25
        Maximum number of optimizer function evaluations.
    tol : float, default 1e-5
        Optimizer convergence tolerance.
    solver_kwargs : Mapping[str, Any] | None
        Additional keyword arguments forwarded to :func:`solve_fiscal_sovereign_default`.

    Returns
    -------
    SovereignSMMResult
    """
    if not isinstance(targets, MomentTargets):
        raise ValueError("targets must be MomentTargets")
    _validate_labels(targets.labels)

    if isinstance(param_names, (str, bytes)):
        raise ValueError("param_names must be a sequence of parameter names, not str or bytes")
    pnames = tuple(param_names)
    if not pnames or any(not isinstance(p, str) or not p.strip() for p in pnames):
        raise ValueError("param_names must contain at least one nonempty string")
    if len(set(pnames)) != len(pnames):
        raise ValueError(f"param_names must be unique; got {pnames}")
    unknown_params = sorted(set(pnames) - _CONTINUOUS_SOLVER_PARAMS)
    if unknown_params:
        raise ValueError(f"Unsupported continuous solver parameter(s): {unknown_params}")

    if np.iscomplexobj(theta0):
        raise ValueError("theta0 must be real")
    theta0_arr = np.asarray(theta0, dtype=float)
    if theta0_arr.ndim != 1 or len(theta0_arr) != len(pnames) or not np.all(np.isfinite(theta0_arr)):
        raise ValueError(f"theta0 must be a finite 1D array of length {len(pnames)}")

    if not isinstance(use_simulation, (bool, np.bool_)):
        raise ValueError("use_simulation must be boolean")
    if not _is_int_like(n_b) or int(n_b) < 2:
        raise ValueError(f"n_b must be an integer >= 2; got {n_b!r}")
    if not _is_int_like(n_y) or int(n_y) < 2:
        raise ValueError(f"n_y must be an integer >= 2; got {n_y!r}")
    if rule is not None and not isinstance(rule, FiscalRule):
        raise TypeError(f"rule must be a FiscalRule or None; got {type(rule).__name__}")
    if solver_kwargs is not None and not isinstance(solver_kwargs, Mapping):
        raise TypeError("solver_kwargs must be a mapping or None")
    if use_simulation:
        _validate_simulation_args(seeds, periods, burn)

    default_search = (
        "dense"
        if (float(sigma_eps) > 0.0 or (rule is not None and rule.deficit_limit is not None))
        else "monotone"
    )
    base_kwargs: dict[str, Any] = {
        "tau": float(tau),
        "psi": float(psi),
        "n_b": int(n_b),
        "n_y": int(n_y),
        "sigma_eps": float(sigma_eps),
        "policy_search": default_search,
        "tol": 1e-8,
        "max_iter": 2000,
        "rule": rule,
    }
    if solver_kwargs:
        base_kwargs.update(solver_kwargs)

    def _solve_at(theta_vec: np.ndarray) -> FiscalSovereignResult:
        kw = dict(base_kwargs)
        for name, val in zip(pnames, theta_vec):
            kw[name] = float(val)
        sol = solve_fiscal_sovereign_default(**kw)
        if not sol.converged:
            raise RuntimeError(
                f"solve_fiscal_sovereign_default did not converge "
                f"(residual={sol.residual:.3e}, iterations={sol.iterations})"
            )
        return sol

    def _moments_at(theta_vec: np.ndarray) -> np.ndarray:
        sol = _solve_at(theta_vec)
        return _compute_stationary_or_simulated_moments(
            sol,
            targets.labels,
            use_simulation=bool(use_simulation),
            seeds=seeds,
            periods=periods,
            burn=burn,
        )

    fit = fit_structural(
        _moments_at,
        targets,
        theta0_arr,
        parameter_names=pnames,
        bounds=bounds,
        max_nfev=max_nfev,
        tol=tol,
    )
    sens = ags_sensitivity(fit)
    solved_model = _solve_at(fit.theta)
    param_dict = {name: float(val) for name, val in zip(pnames, fit.theta)}
    return SovereignSMMResult(
        fit=fit,
        sensitivity=sens,
        model=solved_model,
        parameter_dict=param_dict,
        metadata={
            "use_simulation": bool(use_simulation),
            "n_b": int(n_b),
            "n_y": int(n_y),
            "sigma_eps": float(sigma_eps),
            "theta": float(solved_model.metadata["theta"]),
        },
    )
