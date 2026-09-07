"""Marginal likelihood and model comparison.

``estimate_dsge`` has said, in its own docstring, "There is no
marginal-likelihood estimator, so no model comparison." This supplies two, and
is candid about the difference between them.

**Laplace.** ``log p(y) ~= log p(y|th*) + log p(th*) + (d/2) log 2pi +
0.5 log|Sigma*|`` with ``Sigma*`` the inverse Hessian of the negative log
posterior at the mode. Cheap, deterministic, and **exact** when the posterior
is Gaussian — which is what the test suite pins it against. It is only as good
as the mode: a "mode" that is not one makes it meaningless, so run
:func:`puremacro.dsge.mode.mode_check` first.

**Modified harmonic mean** (Geweke 1999). Reweights the posterior draws by a
truncated normal ``f`` matched to their own mean and covariance, and estimates
``p(y)^-1`` as the average of ``f(th_m) / [p(y|th_m) p(th_m)]``. Evaluated at
every truncation level ``p``, and **the spread across levels is returned, not
hidden**: an estimator that moves by more than a log point depending on where
you cut the tails has not converged, and says so.

A note on comparability that is not optional. Marginal likelihoods computed
before puremacro 2.5.0 are not comparable to these or to each other:
``estimate_dsge`` then started the Kalman recursion at a diffuse ``P0 = 1e6*I``
rather than the unconditional covariance, which added an arbitrary,
scale-dependent constant to every log-likelihood. On Smets-Wouters (2007) that
constant is about 114 log points.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

__all__ = [
    "HarmonicMeanResult",
    "laplace_mdd",
    "harmonic_mean_mdd",
    "model_comparison",
]

_DEFAULT_TRUNCATIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Above this swing across truncation levels the modified harmonic mean is
#: reported as not converged. One log point is already a Bayes factor of e.
_SPREAD_TOL = 1.0


def laplace_mdd(log_post_mode: float, hessian_inv: np.ndarray) -> float:
    """Laplace approximation to ``log p(y)``.

    Parameters
    ----------
    log_post_mode : float
        ``log p(y | th*) + log p(th*)`` at the posterior mode — the *positive*
        log posterior, not the objective ``estimate_dsge`` minimises.
    hessian_inv : ndarray
        ``Sigma*``, the inverse Hessian of the negative log posterior at the
        mode. Must be positive definite: a non-PD matrix means the point is not
        a maximum, and an answer computed there would be meaningless.

    Returns
    -------
    float
    """
    S = np.asarray(hessian_inv, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"hessian_inv must be square; got shape {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("hessian_inv contains non-finite entries")
    if not np.isfinite(log_post_mode):
        raise ValueError(f"log_post_mode must be finite; got {log_post_mode!r}")
    d = S.shape[0]
    S = 0.5 * (S + S.T)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError(
            "laplace_mdd: the inverse Hessian at the mode is not positive "
            f"definite (slogdet sign = {sign}). That means the point supplied "
            "is not a posterior maximum, so the Laplace approximation around "
            "it has no meaning. Re-run the mode search, or check it with "
            "puremacro.dsge.mode.mode_check."
        )
    return float(log_post_mode + 0.5 * d * math.log(2.0 * math.pi) + 0.5 * logdet)


@dataclass(frozen=True)
class HarmonicMeanResult:
    """Geweke (1999) modified harmonic mean, at every truncation level.

    Attributes
    ----------
    estimate : float
        ``log p(y)`` at the median truncation level.
    by_truncation : dict[float, float]
        The estimate at each level ``p``.
    spread : float
        ``max - min`` across levels. This is the diagnostic: the estimator is
        supposed to be invariant to ``p``, so a large spread means it has not
        converged.
    converged : bool
        ``spread <= 1.0`` log point.
    n_draws : int
        Draws used.
    """

    estimate: float
    by_truncation: dict
    spread: float
    converged: bool
    n_draws: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"log_mdd": list(self.by_truncation.values())},
            index=pd.Index(list(self.by_truncation), name="truncation"),
        )

    def summary(self) -> str:
        lines = [
            "MODIFIED HARMONIC MEAN (Geweke 1999)",
            "=" * 56,
            f"log p(y)  : {self.estimate:.4f}   (median truncation)",
            f"spread    : {self.spread:.4f} log points across truncations",
            f"draws     : {self.n_draws}",
            "",
            self.to_frame().round(4).to_string(),
            "",
        ]
        lines.append(
            "Converged: the estimate is stable across truncation levels."
            if self.converged else
            f"NOT converged: the estimate moves by {self.spread:.2f} log points "
            f"depending on where the tails are cut, which is a Bayes factor of "
            f"{math.exp(min(self.spread, 700.0)):.3g}. Do not report it. More "
            "draws, or a Laplace approximation at a checked mode, instead."
        )
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(list(self.by_truncation), list(self.by_truncation.values()),
                marker="o", **kwargs)
        ax.set_xlabel("truncation level p")
        ax.set_ylabel("log p(y)")
        ax.set_title("Modified harmonic mean by truncation"
                     + ("" if self.converged else "  (not converged)"))
        return ax

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


def harmonic_mean_mdd(
    draws: np.ndarray,
    log_post: np.ndarray,
    *,
    truncations: Sequence[float] = _DEFAULT_TRUNCATIONS,
) -> HarmonicMeanResult:
    """Geweke's modified harmonic-mean estimate of ``log p(y)``.

    Parameters
    ----------
    draws : ndarray
        ``(n_draws, n_params)`` or ``(n_chains, n_draws, n_params)``.
    log_post : ndarray
        The **positive** log posterior at each draw, same leading shape.
    truncations : Sequence[float]
        Truncation probabilities. The spread across them is the diagnostic.
    """
    theta = np.asarray(draws, dtype=float)
    lp = np.asarray(log_post, dtype=float)
    if theta.ndim == 3:
        theta = theta.reshape(-1, theta.shape[-1])
        lp = lp.reshape(-1)
    if theta.ndim != 2:
        raise ValueError(f"draws must be 2-D or 3-D; got shape {np.shape(draws)}")
    if len(theta) != len(lp):
        raise ValueError(
            f"draws and log_post disagree on the number of draws: "
            f"{len(theta)} vs {len(lp)}"
        )
    keep = np.isfinite(lp)
    theta, lp = theta[keep], lp[keep]
    m, d = theta.shape
    if m <= d + 1:
        raise ValueError(
            f"harmonic_mean_mdd: {m} usable draws for {d} parameters is not "
            "enough to estimate the posterior covariance the weighting "
            "function needs."
        )

    mean = theta.mean(axis=0)
    cov = np.cov(theta, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(
            "harmonic_mean_mdd: the posterior covariance of the draws is "
            "singular, so Geweke's weighting function is undefined. The chain "
            "is degenerate in at least one direction — check the acceptance "
            "rate before reading anything else."
        )
    cov_inv = np.linalg.inv(cov)
    dev = theta - mean
    quad = np.einsum("ij,jk,ik->i", dev, cov_inv, dev)

    by_p: dict = {}
    for p in truncations:
        cutoff = float(stats.chi2.ppf(p, d))
        inside = quad <= cutoff
        if not inside.any():
            continue
        log_f = (-math.log(p) - 0.5 * d * math.log(2.0 * math.pi)
                 - 0.5 * logdet - 0.5 * quad[inside])
        # log( (1/M) sum f/posterior ) over the WHOLE sample: draws outside the
        # truncation contribute f = 0, they are not dropped from the average.
        log_mean_ratio = logsumexp(log_f - lp[inside]) - math.log(m)
        by_p[float(p)] = float(-log_mean_ratio)

    if not by_p:
        raise ValueError(
            "harmonic_mean_mdd: no draw fell inside any truncation ellipsoid; "
            "the draws cannot be summarised by their own mean and covariance."
        )
    values = np.array(list(by_p.values()))
    spread = float(values.max() - values.min())
    median = float(np.median(values))
    converged = bool(spread <= _SPREAD_TOL)
    if not converged:
        warnings.warn(
            f"harmonic_mean_mdd: the estimate moves by {spread:.2f} log points "
            f"across truncation levels (from {values.min():.2f} to "
            f"{values.max():.2f}). It is supposed to be invariant to the "
            "truncation, so this one has not converged and should not be "
            "reported.",
            UserWarning,
            stacklevel=2,
        )
    return HarmonicMeanResult(
        estimate=median, by_truncation=by_p, spread=spread,
        converged=converged, n_draws=int(m),
    )


def model_comparison(
    results: Mapping[str, Any] | Sequence[Any],
    *,
    model_priors: Mapping[str, float] | None = None,
    method: str = "laplace",
) -> pd.DataFrame:
    """Posterior model probabilities from marginal likelihoods.

    Every model is scored by the **same** ``method``. A model that cannot
    supply it raises, rather than falling back to the other estimator: a
    Laplace value and a harmonic-mean value are not on the same footing, and
    silently mixing them is how a Bayes factor becomes fiction.

    Parameters
    ----------
    results : Mapping[str, DSGEPosteriorResult] or Sequence
        Estimation results. A sequence is keyed by each result's
        ``model_name``.
    model_priors : Mapping[str, float], optional
        Prior model probabilities; defaults to uniform. Normalised.
    method : {'laplace', 'harmonic'}
        The marginal-likelihood estimator, applied to every model.
    """
    if method not in ("laplace", "harmonic"):
        raise ValueError(
            f"unknown method {method!r}; expected 'laplace' or 'harmonic'"
        )
    if not isinstance(results, Mapping):
        items = list(results)
        names = [getattr(r, "model_name", None) or f"model_{i}"
                 for i, r in enumerate(items)]
        if len(set(names)) != len(names):
            raise ValueError(
                f"model names are not unique: {names}. Pass a mapping "
                "{name: result} instead."
            )
        results = dict(zip(names, items))
    if len(results) < 2:
        raise ValueError(
            f"model_comparison needs at least two models; got {len(results)}"
        )

    log_mdd = {}
    for name, res in results.items():
        try:
            log_mdd[name] = float(res.log_mdd(method=method))
        except Exception as exc:
            raise ValueError(
                f"model_comparison: model {name!r} cannot supply a "
                f"{method!r} marginal likelihood ({type(exc).__name__}: {exc}). "
                "Every model must be scored by the same estimator — falling "
                "back to another one for this model would make the resulting "
                "Bayes factors meaningless."
            ) from exc

    names = list(log_mdd)
    lm = np.array([log_mdd[n] for n in names])
    if model_priors is None:
        log_prior = np.full(len(names), -math.log(len(names)))
    else:
        missing = [n for n in names if n not in model_priors]
        if missing:
            raise ValueError(f"model_priors is missing {missing}")
        pr = np.array([float(model_priors[n]) for n in names])
        if np.any(pr <= 0):
            raise ValueError("model_priors must be strictly positive")
        log_prior = np.log(pr / pr.sum())

    log_post = lm + log_prior
    post = np.exp(log_post - logsumexp(log_post))
    best = int(np.argmax(log_post))
    return pd.DataFrame(
        {
            "log_mdd": lm,
            "log_prior": log_prior,
            "posterior_prob": post,
            "log_bayes_factor_vs_best": log_post - log_post[best],
        },
        index=pd.Index(names, name="model"),
    ).sort_values("posterior_prob", ascending=False)
