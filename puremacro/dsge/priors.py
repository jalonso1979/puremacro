"""Model-agnostic prior framework for Bayesian DSGE estimation.

A `priors` dict has the shape:
    {param_name: {"dist": str, "mean": float, "std": float, "lb": float, "ub": float}}

Supported distributions: ``"beta"``, ``"gamma"``, ``"normal"``, ``"invgamma"``,
``"uniform"``, ``"weibull"``.

Optional per-family fields, added in 2.6.0 so a Dynare ``estimated_params``
block round-trips in full:

===========  ==================  ===================================================
family       extra keys          meaning
===========  ==================  ===================================================
beta         ``shift``,          generalised beta on ``[shift, shift + scale]``
             ``scale``           (``PRIOR_P3``, ``PRIOR_P4 - PRIOR_P3``)
gamma        ``shift``           lower bound (``PRIOR_P3``)
weibull      ``shape``,          the internal pair, as an alternative to mean/std;
             ``scale``,          ``shift`` is the lower bound
             ``shift``
invgamma     ``s``, ``nu``,      internal pair; ``kind`` selects Dynare's
             ``kind``            ``inv_gamma1_pdf`` (default, a prior on a standard
                                 deviation) or ``inv_gamma2_pdf`` (on a variance)
===========  ==================  ===================================================

In every case ``mean``/``std`` remain the mean and standard deviation of the
**actual** variable, shift and scale included — this is Dynare's
``beta_specification`` / ``gamma_specification`` convention, not a reading of
the standardised variable. Reversing it silently re-specifies a whole block.

Parameterisation
----------------
Every spec is read the way Dynare's ``estimated_params`` block reads
``PRIOR_P1, PRIOR_P2``: **``"mean"`` and ``"std"`` are the mean and the
standard deviation of the prior itself**, not internal shape parameters.
That holds for ``invgamma`` too — the entry
``{"dist": "invgamma", "mean": 0.1, "std": 2.0}`` is Dynare's
``INV_GAMMA_PDF, 0.1, 2`` and denotes the type-1 inverse gamma whose mean
is 0.1 and whose standard deviation is 2.  The internal Dynare pair
``(s, nu)`` is recovered by :func:`_invgamma_s_nu` (Dynare's
``inverse_gamma_specification``); an ``invgamma`` spec may also carry an
explicit ``{"s": ..., "nu": ...}``, which then takes precedence.

The inverse-gamma *family* is Dynare's ``lpdfig1``: if ``x`` has this
prior then ``x**2 ~ InvGamma(shape=nu/2, scale=s/2)``.  It is the prior a
DSGE puts on a shock **standard deviation**.  Before 2.4.1 this module
evaluated ``scipy.stats.invgamma(a=nu/2, scale=s**2*nu/2)`` on ``x``
directly and read ``("mean", "std")`` as ``(s, nu)``; both were wrong and
both are fixed here (see CHANGELOG).

This module is the engine-side complement to model-specific prior dicts
like ``puremacro.dsge.sw07_priors.PRIORS``.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gammaln


def _beta_ab(mean: float, std: float) -> tuple[float, float]:
    """(a, b) of the Beta whose mean/std are ``mean``/``std``.

    Raises ``ValueError`` when no such Beta exists — the feasibility
    condition is ``0 < mean < 1`` and ``std**2 < mean * (1 - mean)``.
    Without this check scipy silently returns NaN for a negative shape,
    and the NaN then propagates into the log-posterior as a bogus -inf.
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        raise ValueError(
            f"beta prior: mean={mean!r} and std={std!r} must both be finite"
        )
    if not (0.0 < mean < 1.0):
        raise ValueError(
            f"beta prior: mean must lie strictly inside (0, 1), got {mean!r}"
        )
    if std <= 0.0:
        raise ValueError(f"beta prior: std must be > 0, got {std!r}")
    var_max = mean * (1.0 - mean)
    if std ** 2 >= var_max:
        raise ValueError(
            f"beta prior with mean={mean!r} admits no std >= "
            f"sqrt(mean*(1-mean)) = {math.sqrt(var_max):.6g}; got std={std!r}. "
            "No Beta distribution has that mean and standard deviation "
            "(the implied shape parameter a would be <= 0)."
        )
    a = mean * (var_max / std ** 2 - 1.0)
    b = a * (1.0 - mean) / mean
    return a, b


def _logpdf_beta(
    x: float, mean: float, std: float,
    shift: float = 0.0, scale: float = 1.0,
) -> float:
    """log-pdf of the (generalised) Beta parameterised by mean/std.

    ``shift``/``scale`` are Dynare's ``PRIOR_P3``/``PRIOR_P4 - PRIOR_P3``: the
    variable lives on ``[shift, shift + scale]`` and ``mean``/``std`` are the
    mean and standard deviation **of that variable**, not of the standardised
    one (this is Dynare's ``beta_specification``, which forms
    ``(mu - lb) / (ub - lb)`` and ``sigma / (ub - lb)`` internally).  The
    ``1 / scale`` Jacobian is applied here, so the density integrates to 1 on
    the shifted support.

    The defaults ``shift=0, scale=1`` reproduce the plain Beta exactly.
    """
    if scale <= 0.0:
        raise ValueError(f"beta prior: scale must be > 0, got {scale!r}")
    a, b = _beta_ab((mean - shift) / scale, std / scale)
    z = (x - shift) / scale
    if not (0.0 < z < 1.0):
        return -math.inf
    return float(stats.beta.logpdf(z, a, b)) - math.log(scale)


def _logpdf_gamma(
    x: float, mean: float, std: float, shift: float = 0.0,
) -> float:
    """log-pdf of Gamma(k, theta) parameterised by mean/std.

    ``shift`` is Dynare's ``PRIOR_P3`` (the lower bound).  Following Dynare's
    ``gamma_specification``, ``mean`` is the mean of ``x`` itself, so the
    shifted variable ``x - shift`` has mean ``mean - shift`` and standard
    deviation ``std``.  ``shift=0`` reproduces the plain Gamma exactly.
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        raise ValueError(
            f"gamma prior: mean={mean!r} and std={std!r} must both be finite"
        )
    if mean <= 0.0 or std <= 0.0:
        raise ValueError(
            f"gamma prior requires mean > 0 and std > 0, got "
            f"mean={mean!r}, std={std!r}"
        )
    if mean - shift <= 0.0:
        raise ValueError(
            f"gamma prior: shift={shift!r} must be strictly below the prior "
            f"mean {mean!r}; the shifted variable x - shift would otherwise "
            "have a non-positive mean and no Gamma has that."
        )
    if x <= shift:
        return -math.inf
    k = ((mean - shift) / std) ** 2
    theta = std ** 2 / (mean - shift)
    return float(stats.gamma.logpdf(x - shift, a=k, scale=theta))


def _logpdf_normal(x: float, mean: float, std: float) -> float:
    """log-pdf of Normal(mean, std)."""
    if not (math.isfinite(mean) and math.isfinite(std)) or std <= 0.0:
        raise ValueError(
            f"normal prior requires a finite mean and a finite std > 0, got "
            f"mean={mean!r}, std={std!r}"
        )
    return float(stats.norm.logpdf(x, loc=mean, scale=std))


@lru_cache(maxsize=512)
def _invgamma_s_nu(mean: float, std: float) -> tuple[float, float]:
    """Dynare's ``inverse_gamma_specification`` for the type-1 inverse gamma.

    Returns the ``(s, nu)`` for which ``x**2 ~ InvGamma(nu/2, s/2)`` has
    ``E[x] = mean`` and ``sd(x) = std``.  The two moment conditions are

        E[x]   = sqrt(s/2) * Gamma((nu-1)/2) / Gamma(nu/2) = mean
        E[x^2] = s / (nu - 2)                              = mean^2 + std^2

    so ``s = (nu - 2) * (mean^2 + std^2)`` and ``nu`` solves a single
    monotone scalar equation on ``(2, inf)``, done here with ``brentq``.
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        raise ValueError(
            f"invgamma prior: mean={mean!r} and std={std!r} must both be finite"
        )
    if mean <= 0.0 or std <= 0.0:
        raise ValueError(
            f"invgamma prior requires mean > 0 and std > 0, got "
            f"mean={mean!r}, std={std!r}"
        )
    second_moment = mean * mean + std * std

    def _gap(nu: float) -> float:
        s = (nu - 2.0) * second_moment
        return (
            math.sqrt(s / 2.0)
            * math.exp(gammaln((nu - 1.0) / 2.0) - gammaln(nu / 2.0))
            - mean
        )

    lo, hi = 2.0 + 1e-12, 4.0
    for _ in range(200):
        if _gap(hi) > 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - unreachable for finite mean/std
        raise ValueError(
            f"invgamma prior: could not bracket nu for mean={mean!r}, std={std!r}"
        )
    nu = brentq(_gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    return (nu - 2.0) * second_moment, nu


def _logpdf_invgamma_s_nu(x: float, s: float, nu: float) -> float:
    """Dynare ``lpdfig1(x, s, nu)`` — the type-1 inverse gamma log-density.

    ``x`` has this density iff ``x**2 ~ InvGamma(shape=nu/2, scale=s/2)``.
    """
    if not (s > 0.0 and nu > 0.0):
        raise ValueError(
            f"invgamma prior requires s > 0 and nu > 0, got s={s!r}, nu={nu!r}"
        )
    if x <= 0.0:
        return -math.inf
    return float(
        math.log(2.0)
        - gammaln(nu / 2.0)
        + (nu / 2.0) * math.log(s / 2.0)
        - (nu + 1.0) * math.log(x)
        - s / (2.0 * x * x)
    )


def _logpdf_invgamma(x: float, mean: float, std: float) -> float:
    """log-pdf of the type-1 inverse gamma with mean ``mean`` and std ``std``.

    This is Dynare's ``INV_GAMMA_PDF`` in full: ``(mean, std)`` are the
    ``PRIOR_P1, PRIOR_P2`` an ``estimated_params`` line carries, they are
    mapped to Dynare's internal ``(s, nu)`` by :func:`_invgamma_s_nu`, and
    the density evaluated is Dynare's ``lpdfig1`` — i.e. ``x**2`` is
    inverse-gamma, so ``x`` is a standard deviation, not a variance.
    """
    s, nu = _invgamma_s_nu(float(mean), float(std))
    return _logpdf_invgamma_s_nu(x, s, nu)


def _invgamma_mean_std(s: float, nu: float) -> tuple[float, float]:
    """Inverse of :func:`_invgamma_s_nu`: mean and std implied by ``(s, nu)``."""
    if not (s > 0.0 and nu > 2.0):
        raise ValueError(
            "invgamma prior given as (s, nu) needs s > 0 and nu > 2 for the "
            f"standard deviation to exist; got s={s!r}, nu={nu!r}"
        )
    mean = math.sqrt(s / 2.0) * math.exp(
        gammaln((nu - 1.0) / 2.0) - gammaln(nu / 2.0)
    )
    var = s / (nu - 2.0) - mean * mean
    return mean, math.sqrt(max(var, 0.0))


def _logpdf_invgamma2_s_nu(x: float, s: float, nu: float) -> float:
    """Dynare ``lpdfig2(x, s, nu)`` — the type-2 inverse gamma log-density.

    Type 2 puts the inverse gamma on ``x`` itself: ``x ~ InvGamma(nu/2, s/2)``.
    It is the prior a DSGE puts on a **variance**, where type 1 (``lpdfig1``,
    :func:`_logpdf_invgamma_s_nu`) is the prior on a standard deviation.  The
    two are different densities and the ``estimated_params`` shapes
    ``inv_gamma1_pdf`` / ``inv_gamma2_pdf`` name them apart.
    """
    if not (s > 0.0 and nu > 0.0):
        raise ValueError(
            f"invgamma prior requires s > 0 and nu > 0, got s={s!r}, nu={nu!r}"
        )
    if x <= 0.0:
        return -math.inf
    return float(
        -gammaln(nu / 2.0)
        + (nu / 2.0) * math.log(s / 2.0)
        - 0.5 * (nu + 2.0) * math.log(x)
        - s / (2.0 * x)
    )


def _invgamma2_s_nu(mean: float, std: float) -> tuple[float, float]:
    """``(s, nu)`` of the type-2 inverse gamma with this mean and std.

    Closed form, unlike type 1: for ``x ~ InvGamma(nu/2, s/2)``,
    ``E[x] = s / (nu - 2)`` and ``Var[x] = 2 s^2 / ((nu - 2)^2 (nu - 4))``, so
    ``mean^2 / var = (nu - 4) / 2``.
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        raise ValueError(
            f"invgamma prior: mean={mean!r} and std={std!r} must both be finite"
        )
    if mean <= 0.0 or std <= 0.0:
        raise ValueError(
            f"invgamma prior requires mean > 0 and std > 0, got "
            f"mean={mean!r}, std={std!r}"
        )
    nu = 2.0 * (mean / std) ** 2 + 4.0
    return mean * (nu - 2.0), nu


def _invgamma2_mean_std(s: float, nu: float) -> tuple[float, float]:
    """Inverse of :func:`_invgamma2_s_nu`."""
    if not (s > 0.0 and nu > 0.0):
        raise ValueError(
            f"type-2 invgamma requires s > 0 and nu > 0, got s={s!r}, nu={nu!r}"
        )
    # The density exists for every (s, nu) > 0; its moments do not. Refusing to
    # build a perfectly good prior because its variance is infinite would be
    # wrong, so the missing moments are reported as inf and _validate_priors
    # warns about the consequence (a non-finite MH proposal scale).
    if nu <= 2.0:
        return math.inf, math.inf
    mean = s / (nu - 2.0)
    if nu <= 4.0:
        return mean, math.inf
    var = 2.0 * s * s / ((nu - 2.0) ** 2 * (nu - 4.0))
    return mean, math.sqrt(var)


@lru_cache(maxsize=512)
def _weibull_shape_scale(mean: float, std: float) -> tuple[float, float]:
    """``(shape k, scale lambda)`` of the Weibull with this mean and std.

    ``E[x] = lambda * Gamma(1 + 1/k)`` and
    ``Var[x] = lambda^2 (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)``, so the squared
    coefficient of variation ``(std/mean)^2`` pins ``k`` alone through a
    strictly decreasing function of ``k``; ``lambda`` then follows from the
    mean.  Same shape of solve as :func:`_invgamma_s_nu`.
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        raise ValueError(
            f"weibull prior: mean={mean!r} and std={std!r} must both be finite"
        )
    if mean <= 0.0 or std <= 0.0:
        raise ValueError(
            f"weibull prior requires mean > 0 and std > 0, got "
            f"mean={mean!r}, std={std!r}"
        )
    cv2 = (std / mean) ** 2

    def _gap(k: float) -> float:
        return math.exp(gammaln(1.0 + 2.0 / k) - 2.0 * gammaln(1.0 + 1.0 / k)) - 1.0 - cv2

    lo, hi = 0.05, 1.0
    for _ in range(200):
        if _gap(hi) < 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - unreachable for finite mean/std
        raise ValueError(
            f"weibull prior: could not bracket the shape for mean={mean!r}, std={std!r}"
        )
    while _gap(lo) < 0.0 and lo > 1e-8:  # pragma: no cover - extreme cv only
        lo /= 2.0
    k = brentq(_gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    return k, mean / math.exp(gammaln(1.0 + 1.0 / k))


def _weibull_mean_std(shape: float, scale: float) -> tuple[float, float]:
    """Inverse of :func:`_weibull_shape_scale`."""
    if not (shape > 0.0 and scale > 0.0):
        raise ValueError(
            f"weibull prior requires shape > 0 and scale > 0, got "
            f"shape={shape!r}, scale={scale!r}"
        )
    g1 = math.exp(gammaln(1.0 + 1.0 / shape))
    g2 = math.exp(gammaln(1.0 + 2.0 / shape))
    mean = scale * g1
    var = scale * scale * (g2 - g1 * g1)
    return mean, math.sqrt(max(var, 0.0))


def _logpdf_weibull(
    x: float, shape: float, scale: float, shift: float = 0.0,
) -> float:
    """log-pdf of the Weibull, Dynare's ``WEIBULL_PDF``."""
    if not (shape > 0.0 and scale > 0.0):
        raise ValueError(
            f"weibull prior requires shape > 0 and scale > 0, got "
            f"shape={shape!r}, scale={scale!r}"
        )
    if x <= shift:
        return -math.inf
    return float(stats.weibull_min.logpdf(x - shift, c=shape, scale=scale))


def _logpdf_weibull_mean_std(
    x: float, mean: float, std: float, shift: float = 0.0,
) -> float:
    """``_logpdf_weibull`` reached through the (mean, std) parameterisation."""
    shape, scale = _weibull_shape_scale(float(mean - shift), float(std))
    return _logpdf_weibull(x, shape, scale, shift)


def _logpdf_uniform(x: float, mean: float, std: float) -> float:
    """log-pdf of Uniform distribution parameterized by mean and std."""
    width = math.sqrt(12.0) * std
    return -math.log(max(width, 1e-12))


_DIST_LOGPDF = {
    "beta":     _logpdf_beta,
    "gamma":    _logpdf_gamma,
    "normal":   _logpdf_normal,
    "invgamma": _logpdf_invgamma,
    "uniform":  _logpdf_uniform,
    "weibull":  _logpdf_weibull_mean_std,
}

# Dynare's inverse-gamma variants. "type1" is lpdfig1 (a prior on a standard
# deviation, x**2 inverse gamma); "type2" is lpdfig2 (a prior on a variance).
_INVGAMMA_KINDS = ("type1", "type2")


def _logpdf_for_spec(spec: dict | Prior, x: float) -> float:
    """Dispatch to the correct log-pdf given a single param spec and a value."""
    lb = spec["lb"]
    ub = spec["ub"]
    if not (lb <= x <= ub):
        return -math.inf
    dist = spec["dist"]
    if dist == "uniform":
        return -math.log(max(ub - lb, 1e-12))

    shift = float(spec.get("shift") or 0.0)

    if dist == "invgamma":
        kind = spec.get("kind") or "type1"
        if kind not in _INVGAMMA_KINDS:
            raise ValueError(
                f"unknown inverse-gamma kind {kind!r}; expected one of "
                f"{list(_INVGAMMA_KINDS)}"
            )
        s_val, nu_val = spec.get("s"), spec.get("nu")
        if s_val is None or nu_val is None:
            solve = _invgamma2_s_nu if kind == "type2" else _invgamma_s_nu
            s_val, nu_val = solve(float(spec["mean"]), float(spec["std"]))
        density = (
            _logpdf_invgamma2_s_nu if kind == "type2" else _logpdf_invgamma_s_nu
        )
        return density(x, float(s_val), float(nu_val))

    if dist == "beta":
        return _logpdf_beta(
            x, float(spec["mean"]), float(spec["std"]),
            shift, float(spec.get("scale") if spec.get("scale") is not None else 1.0),
        )

    if dist == "gamma":
        return _logpdf_gamma(x, float(spec["mean"]), float(spec["std"]), shift)

    if dist == "weibull":
        shape, scale = spec.get("shape"), spec.get("scale")
        if shape is None or scale is None:
            shape, scale = _weibull_shape_scale(
                float(spec["mean"]) - shift, float(spec["std"])
            )
        return _logpdf_weibull(x, float(shape), float(scale), shift)

    try:
        fn = _DIST_LOGPDF[dist]
    except KeyError as exc:
        raise ValueError(f"unknown distribution {dist!r} for prior spec") from exc
    return fn(x, spec["mean"], spec["std"])


class Prior:
    """Base class for Bayesian prior distributions in DSGE models."""

    #: Optional per-family fields a subclass may carry alongside the five
    #: canonical ones (``shift``/``scale`` for a generalised beta or gamma,
    #: ``s``/``nu``/``kind`` for an inverse gamma, ``shape``/``scale`` for a
    #: Weibull).  They reach :func:`_logpdf_for_spec` through ``__getitem__``
    #: and ``to_dict`` exactly as a plain dict spec's keys would.
    _EXTRA_DEFAULTS: dict = {}

    def __init__(
        self,
        dist: str,
        mean: float,
        std: float,
        lb: float = -math.inf,
        ub: float = math.inf,
        **extra,
    ) -> None:
        self.dist = dist
        self.mean = float(mean)
        self.std = float(std)
        self.lb = float(lb)
        self.ub = float(ub)
        self._extra = {**self._EXTRA_DEFAULTS, **extra}

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        if isinstance(x, (int, float, np.floating)):
            return _logpdf_for_spec(self, float(x))
        arr = np.asarray(x, dtype=float)
        out = np.empty_like(arr)
        for idx, val in np.ndenumerate(arr):
            out[idx] = _logpdf_for_spec(self, float(val))
        return out

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return np.exp(self.logpdf(x))

    def __getitem__(self, key: str):
        if key == "dist":
            return self.dist
        if key == "mean":
            return self.mean
        if key == "std":
            return self.std
        if key == "lb":
            return self.lb
        if key == "ub":
            return self.ub
        if key in self._extra:
            return self._extra[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in ("dist", "mean", "std", "lb", "ub") or key in self._extra

    def __getattr__(self, name: str):
        # Per-family fields (`shift`, `scale`, `shape`, `s`, `nu`, `kind`) read
        # as attributes too. Only reached when normal lookup fails, so it never
        # shadows a real attribute.
        try:
            return self.__dict__["_extra"][name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {
            "dist": self.dist,
            "mean": self.mean,
            "std": self.std,
            "lb": self.lb,
            "ub": self.ub,
            **self._extra,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
            f"lb={self.lb}, ub={self.ub})"
        )


class BetaPrior(Prior):
    """Beta prior parameterized by mean and std (Dynare/Pfeifer convention).

    ``shift`` and ``scale`` give Dynare's *generalised* beta: the variable
    lives on ``[shift, shift + scale]`` (``PRIOR_P3`` and ``PRIOR_P4 -
    PRIOR_P3``) and ``mean``/``std`` describe **that** variable.  With the
    defaults ``shift=0, scale=1`` this is the plain Beta, bit-for-bit.

    ``lb``/``ub`` default to the support: ``(1e-4, 0.9999)`` for the plain
    Beta, ``(shift, shift + scale)`` for a generalised one.
    """

    _EXTRA_DEFAULTS = {"shift": 0.0, "scale": 1.0}

    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.1,
        lb: float | None = None,
        ub: float | None = None,
        *,
        shift: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        shift = float(shift)
        scale = float(scale)
        if scale <= 0.0:
            raise ValueError(f"beta prior: scale must be > 0, got {scale!r}")
        generalised = shift != 0.0 or scale != 1.0
        if lb is None:
            lb = shift if generalised else 1e-4
        if ub is None:
            ub = shift + scale if generalised else 0.9999
        super().__init__("beta", mean, std, lb, ub, shift=shift, scale=scale)


class InvGammaPrior(Prior):
    """Type-1 inverse-gamma prior, Dynare's ``INV_GAMMA_PDF``.

    ``mean`` and ``std`` are the **mean and standard deviation of the prior
    itself** (Dynare's ``PRIOR_P1, PRIOR_P2``), matching every other prior
    class here.  The distribution is the one a DSGE puts on a shock
    standard deviation: if ``x`` has this prior then
    ``x**2 ~ InvGamma(shape=nu/2, scale=s/2)``, and the internal Dynare
    pair ``(s, nu)`` is exposed on the ``.s`` / ``.nu`` attributes.

    Passing ``s=`` and ``nu=`` instead gives that internal pair directly;
    ``nu > 2`` is then required for the standard deviation to exist.

    .. versionchanged:: 2.4.1
       ``mean``/``std`` used to be stored verbatim as ``(s, nu)`` and the
       density evaluated was ``invgamma(a=nu/2, scale=s**2*nu/2)`` applied
       to ``x`` rather than ``x**2``.  Neither matched Dynare; both are
       fixed, so log-densities from this class have changed.
    """

    _EXTRA_DEFAULTS = {"kind": "type1"}

    def __init__(
        self,
        mean: float | None = None,
        std: float | None = None,
        lb: float = 1e-4,
        ub: float = math.inf,
        *,
        s: float | None = None,
        nu: float | None = None,
        kind: str = "type1",
    ) -> None:
        if kind not in _INVGAMMA_KINDS:
            raise ValueError(
                f"InvGammaPrior: unknown kind {kind!r}; expected one of "
                f"{list(_INVGAMMA_KINDS)} (Dynare's inv_gamma1_pdf / "
                "inv_gamma2_pdf)."
            )
        if s is not None or nu is not None:
            if s is None or nu is None:
                raise ValueError(
                    "InvGammaPrior: pass s and nu together, or neither "
                    "(use mean= and std= for Dynare's PRIOR_P1/PRIOR_P2)."
                )
            if mean is not None or std is not None:
                raise ValueError(
                    "InvGammaPrior: give either (mean, std) or (s, nu), "
                    "not both."
                )
            moments = _invgamma2_mean_std if kind == "type2" else _invgamma_mean_std
            val_mean, val_std = moments(float(s), float(nu))
            val_s, val_nu = float(s), float(nu)
        else:
            val_mean = 0.1 if mean is None else float(mean)
            val_std = 2.0 if std is None else float(std)
            solve = _invgamma2_s_nu if kind == "type2" else _invgamma_s_nu
            val_s, val_nu = solve(val_mean, val_std)
        super().__init__(
            "invgamma", val_mean, val_std, lb, ub,
            s=float(val_s), nu=float(val_nu), kind=kind,
        )

    @property
    def s(self) -> float:
        """Dynare's internal scale parameter ``s``."""
        return self._extra["s"]

    @property
    def nu(self) -> float:
        """Dynare's internal degrees-of-freedom parameter ``nu``."""
        return self._extra["nu"]


class NormalPrior(Prior):
    """Normal prior parameterized by mean and std."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        lb: float = -math.inf,
        ub: float = math.inf,
    ) -> None:
        super().__init__("normal", mean, std, lb, ub)


class GammaPrior(Prior):
    """Gamma prior parameterized by mean and std.

    ``shift`` is Dynare's ``PRIOR_P3`` (the lower bound).  ``mean`` is the
    mean of the variable itself, so ``x - shift`` has mean ``mean - shift``
    and standard deviation ``std``.  ``shift=0`` is the plain Gamma.
    """

    _EXTRA_DEFAULTS = {"shift": 0.0}

    def __init__(
        self,
        mean: float = 1.0,
        std: float = 0.5,
        lb: float | None = None,
        ub: float = math.inf,
        *,
        shift: float = 0.0,
    ) -> None:
        shift = float(shift)
        if lb is None:
            lb = shift if shift != 0.0 else 1e-4
        super().__init__("gamma", mean, std, lb, ub, shift=shift)


class WeibullPrior(Prior):
    """Weibull prior, Dynare's ``WEIBULL_PDF``.

    ``mean``/``std`` are Dynare's ``PRIOR_P1, PRIOR_P2`` as in every other
    family here, with ``shift`` the optional lower bound ``PRIOR_P3``; the
    internal ``(shape, scale)`` pair is solved for by
    :func:`_weibull_shape_scale` and exposed on ``.shape`` / ``.scale``.
    Passing ``shape=`` and ``scale=`` instead gives that pair directly.
    """

    _EXTRA_DEFAULTS = {"shift": 0.0}

    def __init__(
        self,
        mean: float | None = None,
        std: float | None = None,
        lb: float | None = None,
        ub: float = math.inf,
        *,
        shape: float | None = None,
        scale: float | None = None,
        shift: float = 0.0,
    ) -> None:
        shift = float(shift)
        if shape is not None or scale is not None:
            if shape is None or scale is None:
                raise ValueError(
                    "WeibullPrior: pass shape and scale together, or neither "
                    "(use mean= and std= for Dynare's PRIOR_P1/PRIOR_P2)."
                )
            if mean is not None or std is not None:
                raise ValueError(
                    "WeibullPrior: give either (mean, std) or (shape, scale), "
                    "not both."
                )
            val_shape, val_scale = float(shape), float(scale)
            base_mean, val_std = _weibull_mean_std(val_shape, val_scale)
            val_mean = base_mean + shift
        else:
            val_mean = 1.0 if mean is None else float(mean)
            val_std = 0.5 if std is None else float(std)
            val_shape, val_scale = _weibull_shape_scale(val_mean - shift, val_std)
        if lb is None:
            lb = shift
        super().__init__(
            "weibull", val_mean, val_std, lb, ub,
            shape=val_shape, scale=val_scale, shift=shift,
        )


class UniformPrior(Prior):
    """Uniform prior on [lb, ub]."""

    def __init__(self, lb: float = 0.0, ub: float = 1.0) -> None:
        mean = (lb + ub) / 2.0
        std = (ub - lb) / math.sqrt(12.0)
        super().__init__("uniform", mean, std, lb, ub)


def ensure_prior(spec: dict | Prior) -> Prior:
    """Convert dict prior spec or return Prior instance."""
    if isinstance(spec, Prior):
        return spec
    dist = spec["dist"].lower()
    mean = spec.get("mean", 0.0)
    std = spec.get("std", 1.0)
    lb = spec.get("lb", -math.inf)
    ub = spec.get("ub", math.inf)
    shift = spec.get("shift")
    if dist == "beta":
        scale = spec.get("scale")
        # A generalised beta with no declared truncation takes its support as
        # the bounds; a plain one keeps the +/-inf a bare dict spec implies,
        # so param_bounds() is unchanged for every pre-2.6.0 spec.
        if (shift or scale) and "lb" not in spec and "ub" not in spec:
            lb = ub = None
        return BetaPrior(
            mean=mean, std=std, lb=lb, ub=ub,
            shift=0.0 if shift is None else shift,
            scale=1.0 if scale is None else scale,
        )
    elif dist in ("invgamma", "inv_gamma"):
        s_val = spec.get("s")
        nu_val = spec.get("nu")
        kind = spec.get("kind") or "type1"
        if s_val is not None and nu_val is not None:
            return InvGammaPrior(lb=lb, ub=ub, s=s_val, nu=nu_val, kind=kind)
        return InvGammaPrior(mean=mean, std=std, lb=lb, ub=ub, kind=kind)
    elif dist == "normal":
        return NormalPrior(mean=mean, std=std, lb=lb, ub=ub)
    elif dist == "gamma":
        if shift and "lb" not in spec:
            lb = None
        return GammaPrior(
            mean=mean, std=std, lb=lb, ub=ub,
            shift=0.0 if shift is None else shift,
        )
    elif dist == "weibull":
        shape_val = spec.get("shape")
        scale_val = spec.get("scale")
        kw = dict(ub=ub, shift=0.0 if shift is None else shift)
        if "lb" in spec:
            kw["lb"] = lb
        if shape_val is not None and scale_val is not None:
            return WeibullPrior(shape=shape_val, scale=scale_val, **kw)
        return WeibullPrior(mean=spec["mean"], std=spec["std"], **kw)
    elif dist == "uniform":
        return UniformPrior(lb=lb, ub=ub)
    return Prior(dist, mean, std, lb, ub)


def _validate_priors(priors: dict, *, caller: str = "estimate_dsge") -> None:
    """Fail fast on a prior spec that no distribution can satisfy.

    Catches, per parameter: an unknown ``dist``; ``lb >= ub``; a
    ``mean``/``std`` pair outside the family's feasible set (a Beta with
    ``std**2 >= mean*(1-mean)``, a Gamma/InvGamma with a non-positive mean
    or std, a Normal with a non-positive std).  Without it these surface
    as a NaN log-prior that the driver turns into ``-inf``, and the user
    is told to "pick a better starting point" for a prior that is simply
    impossible.

    Raises ``ValueError`` naming ``caller`` and the offending parameter.
    Emits a ``UserWarning`` (not an error) when the prior mean falls
    outside the declared ``[lb, ub]`` truncation, which is legal but is
    almost always a transcription slip.
    """
    if not priors:
        raise ValueError(f"{caller}: priors is empty; nothing to estimate")
    for name, spec in priors.items():
        try:
            dist = spec["dist"]
            lb = float(spec["lb"])
            ub = float(spec["ub"])
            mean = float(spec["mean"])
            std = float(spec["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{caller}: prior spec for {name!r} is malformed "
                f"({type(exc).__name__}: {exc}); it needs "
                "'dist', 'mean', 'std', 'lb' and 'ub'."
            ) from exc
        if dist not in _DIST_LOGPDF:
            raise ValueError(
                f"{caller}: unknown distribution {dist!r} for prior "
                f"{name!r}; supported: {sorted(_DIST_LOGPDF)}"
            )
        if not (lb < ub):
            raise ValueError(
                f"{caller}: prior {name!r} has lb={lb!r} >= ub={ub!r}; "
                "the support is empty."
            )
        shift = float(spec.get("shift") or 0.0)
        try:
            if dist == "beta":
                scale = spec.get("scale")
                scale = 1.0 if scale is None else float(scale)
                if scale <= 0.0:
                    raise ValueError(f"beta prior: scale must be > 0, got {scale!r}")
                _beta_ab((mean - shift) / scale, std / scale)
            elif dist == "gamma":
                _logpdf_gamma(max(mean, 1e-12), mean, std, shift)
            elif dist == "weibull":
                _weibull_shape_scale(mean - shift, std)
            elif dist == "invgamma":
                kind = spec.get("kind") or "type1"
                if kind not in _INVGAMMA_KINDS:
                    raise ValueError(
                        f"unknown inverse-gamma kind {kind!r}; expected one of "
                        f"{list(_INVGAMMA_KINDS)}"
                    )
                if spec.get("s") is not None and spec.get("nu") is not None:
                    density = (
                        _logpdf_invgamma2_s_nu if kind == "type2"
                        else _logpdf_invgamma_s_nu
                    )
                    density(max(mean, 1e-12), float(spec["s"]), float(spec["nu"]))
                else:
                    solve = _invgamma2_s_nu if kind == "type2" else _invgamma_s_nu
                    solve(mean, std)
            elif dist == "normal":
                _logpdf_normal(mean, mean, std)
        except ValueError as exc:
            raise ValueError(f"{caller}: prior {name!r}: {exc}") from exc
        if not math.isfinite(std):
            warnings.warn(
                f"{caller}: prior {name!r} has a non-finite standard deviation "
                f"({std!r}). The density is well defined, but the Metropolis "
                "proposal falls back to diag(prior_stds**2) when the Hessian "
                "is not usable, and that fallback would be infinite. Give the "
                "prior a finite std, or supply a proposal covariance.",
                UserWarning,
                stacklevel=2,
            )
        if dist != "uniform" and not (lb <= mean <= ub):
            warnings.warn(
                f"{caller}: prior {name!r} has mean={mean!r} outside its "
                f"declared support [{lb!r}, {ub!r}]; the truncated prior "
                "will have a very different mean from the one you wrote.",
                UserWarning,
                stacklevel=2,
            )


def log_prior(params: dict, priors: dict) -> float:
    """Sum of log-prior densities across all parameters in ``priors``.

    Returns -inf if any parameter value is missing, non-finite, or outside
    its declared ``[lb, ub]`` support. Raises ``ValueError`` if any spec
    declares an unsupported ``dist``.
    """
    total = 0.0
    for name, spec in priors.items():
        if name not in params:
            return -math.inf
        x = params[name]
        if not math.isfinite(x):
            return -math.inf
        contrib = _logpdf_for_spec(spec, x)
        if contrib == -math.inf:
            return -math.inf
        total += contrib
    return total


def prior_means(priors: dict) -> dict[str, float]:
    """Return ``{name: mean}`` in priors-dict insertion order."""
    return {name: spec["mean"] for name, spec in priors.items()}


def prior_stds(priors: dict) -> dict[str, float]:
    """Return ``{name: std}`` in priors-dict insertion order."""
    return {name: spec["std"] for name, spec in priors.items()}


def param_bounds(priors: dict) -> list[tuple[float, float]]:
    """Return ``[(lb, ub), ...]`` in priors-dict insertion order.

    Matches the shape scipy.optimize.minimize expects for ``bounds``.
    """
    return [(spec["lb"], spec["ub"]) for spec in priors.values()]


def param_names(priors: dict) -> tuple[str, ...]:
    """Return parameter names in priors-dict insertion order."""
    return tuple(priors.keys())


__all__ = [
    "Prior",
    "BetaPrior",
    "InvGammaPrior",
    "NormalPrior",
    "GammaPrior",
    "UniformPrior",
    "ensure_prior",
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
