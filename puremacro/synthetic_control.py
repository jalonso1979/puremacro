"""Abadie-Diamond-Hainmueller synthetic control method.

The treated unit's counterfactual is built as a convex combination of
donor units that matches the treated unit's pre-treatment outcome
trajectory (and optionally a vector of pre-treatment covariates).

Donor weights w solve

    min_w  || X_treat - X_donors @ w ||_V^2
    s.t.   w >= 0,    sum(w) = 1

where ``V`` is a diagonal weighting of predictors. The default V is the
identity (Abadie 2010 unweighted version). Inference is via in-space
placebo distribution: re-run the SCM treating each donor as the
"placebo treated" unit and rank the post-treatment treatment effects.

References
----------
Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic control
    methods for comparative case studies: estimating the effect of
    California's tobacco control program. JASA 105(490), 493-505.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SyntheticControlResult:
    """Result of :func:`synthetic_control`.

    Attributes
    ----------
    weights : pd.Series
        Donor weights indexed by donor unit name. Non-negative and sum to 1.
    synthetic : pd.Series
        Counterfactual outcome path for the treated unit (indexed over
        both pre- and post-treatment periods).
    actual : pd.Series
        Observed outcome path for the treated unit (same index).
    rmse_pre : float
        Root-mean-squared pre-treatment match error.
    treatment_effect : pd.Series
        Treated minus synthetic over the post-treatment period.
    placebo_gaps : pd.DataFrame | None
        Per-donor placebo treatment-effect paths (one column per donor),
        or ``None`` when ``placebo=False``.
    """

    weights: pd.Series
    synthetic: pd.Series
    actual: pd.Series
    rmse_pre: float
    treatment_effect: pd.Series
    placebo_gaps: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        eff = float(self.treatment_effect.mean())
        n_post = len(self.treatment_effect)
        nz = int((self.weights > 1e-4).sum())
        return (
            f"Synthetic control result\n"
            f"  donors used (>1e-4 weight): {nz} / {len(self.weights)}\n"
            f"  pre-treatment RMSE        : {self.rmse_pre:.4f}\n"
            f"  mean post-period gap      : {eff:+.4f}  (n_post={n_post})\n"
            f"  placebo distribution      : "
            f"{'present (%d donors)' % len(self.placebo_gaps.columns) if self.placebo_gaps is not None else 'not computed'}\n"
        )

    def plot(self, *, title: str = "", ax=None):
        """Plot actual vs synthetic outcome trajectories.

        Lazily creates a figure comparing actual outcome vs synthetic counterfactual.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
        else:
            fig = ax.figure

        ax.plot(self.actual.index, self.actual.values, color="0.0", lw=1.5, label="Actual")
        ax.plot(self.synthetic.index, self.synthetic.values, color="0.4", ls="--", lw=1.5, label="Synthetic")
        if len(self.treatment_effect) > 0:
            treat_start = self.treatment_effect.index[0]
            ax.axvline(treat_start, color="0.5", ls=":", lw=1.0, label="Intervention")
        ax.set_title(title or "Synthetic Control: Actual vs. Synthetic")
        ax.legend(loc="best", frameon=False)
        return fig


def _scm_weights(X_treat: np.ndarray, X_donors: np.ndarray,
                 V: np.ndarray | None = None) -> np.ndarray:
    """Solve the constrained QP for donor weights via SLSQP."""
    n_donors = X_donors.shape[1]
    if V is None:
        V = np.eye(X_treat.shape[0])

    def loss(w):
        diff = X_treat - X_donors @ w
        return float(diff @ V @ diff)

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.full(n_donors, 1.0 / n_donors)
    res = minimize(loss, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 200})
    if not res.success:
        # Fall back to projected-gradient on the simplex
        w = w0
        for _ in range(2000):
            grad = -2.0 * X_donors.T @ V @ (X_treat - X_donors @ w)
            w = w - 0.001 * grad
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s > 0:
                w = w / s
        return w
    return res.x


def synthetic_control(
    Y: pd.DataFrame,
    treated_unit: str,
    pre_period: list,
    post_period: list,
    predictors: pd.DataFrame | None = None,
    placebo: bool = True,
) -> SyntheticControlResult:
    """Run synthetic control for a single treated unit.

    Parameters
    ----------
    Y : pd.DataFrame indexed by time, columns are unit names. Must
        include both pre and post periods, and the treated unit.
    treated_unit : column name of the treated unit.
    pre_period, post_period : lists of index labels.
    predictors : optional (n_predictors, n_units) DataFrame of
        pre-treatment covariates to match on (concatenated to the
        outcome match; one row per predictor, columns matching Y).
    placebo : if True, run the same SCM treating each donor as a
        placebo treated unit.

    Returns
    -------
    SyntheticControlResult with ``weights``, ``synthetic``, ``actual``,
    ``rmse_pre``, ``treatment_effect``, and ``placebo_gaps`` (the last
    is ``None`` when ``placebo=False``).
    """
    donors = [c for c in Y.columns if c != treated_unit]
    Y_pre = Y.loc[pre_period]
    Y_post = Y.loc[post_period]

    X_treat_pre = Y_pre[treated_unit].values.astype(float)
    X_donors_pre = Y_pre[donors].values.astype(float)
    if predictors is not None:
        p_treat = predictors[treated_unit].values.astype(float)
        p_donors = predictors[donors].values.astype(float)
        X_treat = np.concatenate([X_treat_pre, p_treat])
        X_donors = np.vstack([X_donors_pre, p_donors])
    else:
        X_treat = X_treat_pre
        X_donors = X_donors_pre

    w = _scm_weights(X_treat, X_donors)
    synthetic_pre = X_donors_pre @ w
    synthetic_post = Y_post[donors].values @ w
    synthetic_full = np.concatenate([synthetic_pre, synthetic_post])
    actual_full = np.concatenate([X_treat_pre, Y_post[treated_unit].values])
    rmse_pre = float(np.sqrt(np.mean((X_treat_pre - synthetic_pre) ** 2)))

    weights = pd.Series(w, index=donors)
    synthetic = pd.Series(synthetic_full,
                          index=list(pre_period) + list(post_period))
    actual = pd.Series(actual_full,
                       index=list(pre_period) + list(post_period))
    treatment_effect = pd.Series(
        Y_post[treated_unit].values - synthetic_post,
        index=post_period,
    )

    placebo_gaps: Optional[pd.DataFrame] = None
    if placebo:
        gaps = {}
        for placebo_unit in donors:
            pl_donors = [c for c in Y.columns if c != placebo_unit]
            Y_pre_p = Y_pre[placebo_unit].values.astype(float)
            X_p = Y_pre[pl_donors].values.astype(float)
            if predictors is not None:
                p_p = predictors[placebo_unit].values.astype(float)
                p_d = predictors[pl_donors].values.astype(float)
                Xt = np.concatenate([Y_pre_p, p_p])
                Xd = np.vstack([X_p, p_d])
            else:
                Xt = Y_pre_p
                Xd = X_p
            try:
                w_p = _scm_weights(Xt, Xd)
                synth_post_p = Y_post[pl_donors].values @ w_p
                gaps[placebo_unit] = (
                    Y_post[placebo_unit].values - synth_post_p
                ).tolist()
            except Exception:
                continue
        placebo_gaps = pd.DataFrame(gaps, index=post_period)

    return SyntheticControlResult(
        weights=weights,
        synthetic=synthetic,
        actual=actual,
        rmse_pre=rmse_pre,
        treatment_effect=treatment_effect,
        placebo_gaps=placebo_gaps,
    )


__all__ = ["synthetic_control", "SyntheticControlResult"]
