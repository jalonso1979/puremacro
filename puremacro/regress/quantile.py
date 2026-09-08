"""Quantile regression by iteratively reweighted least squares.

Reproduces ``statsmodels.regression.quantile_regression.QuantReg(y, X).fit(q=τ)``
— the IRLS loop, the Hall-Sheather bandwidth, the Epanechnikov kernel and the
Koenker-Machado / Powell sparsity sandwich — in pure numpy + scipy, so a
notebook that reads ``params`` / ``bse`` / ``pvalues`` off a ``smf.quantreg``
fit can drop statsmodels and still run under Pyodide.

How this differs from the two quantile fitters puremacro already ships
-------------------------------------------------------------------
There are now three, and they are **not** interchangeable. Pick deliberately.

``puremacro.regress.quantile.quantreg`` (this module)
    A *cross-sectional* quantile regression of ``y`` on a design matrix you
    build yourself. The optimiser is IRLS on the check function with the
    residual magnitude floored at ``1e-6``, which is what statsmodels does;
    it converges to the Koenker-Bassett optimum but is not the exact
    linear-programming solution — see "Notes" for the measured gap. It is
    the only one of the three that returns standard errors, t statistics,
    p values, confidence intervals and a covariance matrix.
    **Use it when you need statsmodels-comparable inference on a
    cross-section**, or when you are porting a ``smf.quantreg`` call site and
    the published number must not move.

``puremacro.lp.quantile._qreg`` (private)
    Solves the *exact* Koenker-Bassett linear program through
    ``scipy.optimize.linprog(method="highs")``. It returns coefficients only —
    no covariance, no standard errors — and ``np.full(k, nan)`` if the LP
    fails. Being an exact simplex solution it is, strictly, the better point
    estimate: at the optimum exactly ``k`` residuals are zero, whereas IRLS
    only approaches that configuration. **Use it when you want the LP optimum
    and will get your uncertainty from a bootstrap**, and when bit-agreement
    with statsmodels does not matter.

``puremacro.lp.lp_quantile`` and ``puremacro.gar.qar``
    Public wrappers around ``_qreg`` that impose a *time-series* shape — a
    local projection over horizons, and a quantile autoregression,
    respectively. Both build their own lag structure, loop over quantiles and
    horizons, and report **bootstrap** bands (``LPResult.se`` is documented as
    all-NaN for ``lp_quantile``). They are growth-at-risk tools, not
    general-purpose quantile regressions. **Use them when the object of
    interest is a conditional-quantile impulse response or forecast path**,
    not a single cross-sectional slope.

The estimand is the same in all three; the optimiser, the inference and the
data shape are not. This module never calls the other two, and none of them
calls this one, so the LP and IRLS answers can be compared against each other
directly — ``tests/test_regress_quantile_parity.py`` does exactly that and
records the size of the gap.

References
----------
Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46(1).
Koenker, R. and Machado, J.A.F. (1999). Goodness of fit and related inference
    processes for quantile regression. JASA 94(448).
Powell, J.L. (1991). Estimation of monotonic regression models under quantile
    restrictions. In Nonparametric and Semiparametric Methods in Econometrics.
Hall, P. and Sheather, S.J. (1988). On the distribution of the studentized
    quantile. JRSS-B 50(3).
Greene, W.H. (2008). Econometric Analysis, 6th ed., pp. 407-408.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from .._linalg import inv_xtx

__all__ = ["quantreg", "QuantRegResult"]


# --------------------------------------------------------------------------
# Kernels and bandwidth rules.
#
# Transcribed from statsmodels 0.14.6
# regression/quantile_regression.py:226-262 rather than from the papers,
# because parity is the point: the published formulae differ from this code
# in normalisation (the Parzen kernel below is Greene's Table 14.2 form, not
# the textbook 1/(2π)∫ form), and a "corrected" kernel would move every
# standard error in the corpus.
# --------------------------------------------------------------------------
def _parzen(u: np.ndarray) -> np.ndarray:
    """Parzen kernel, exactly as statsmodels writes it (``kernels['par']``)."""
    z = np.where(
        np.abs(u) <= 0.5,
        4.0 / 3 - 8.0 * u ** 2 + 8.0 * np.abs(u) ** 3,
        8.0 * (1 - np.abs(u)) ** 3 / 3.0,
    )
    z[np.abs(u) > 1] = 0
    return z


KERNELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "biw": lambda u: 15.0 / 16 * (1 - u ** 2) ** 2 * np.where(np.abs(u) <= 1, 1, 0),
    "cos": lambda u: np.where(np.abs(u) <= 0.5, 1 + np.cos(2 * np.pi * u), 0),
    "epa": lambda u: 3.0 / 4 * (1 - u ** 2) * np.where(np.abs(u) <= 1, 1, 0),
    "gau": norm.pdf,
    "par": _parzen,
}


def hall_sheather(n: float, q: float, alpha: float = 0.05) -> float:
    """Hall-Sheather (1988) bandwidth rule, statsmodels' ``'hsheather'``.

    ``h = n^(-1/3) · Φ⁻¹(1-α/2)^(2/3) · [1.5·φ(Φ⁻¹(q))² / (2·Φ⁻¹(q)² + 1)]^(1/3)``
    """
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z) ** 2.0
    den = 2.0 * z ** 2.0 + 1
    return n ** (-1.0 / 3) * norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3) * (num / den) ** (1.0 / 3)


def bofinger(n: float, q: float) -> float:
    """Bofinger (1975) bandwidth rule, statsmodels' ``'bofinger'``."""
    num = 9.0 / 2 * norm.pdf(2 * norm.ppf(q)) ** 4
    den = (2 * norm.ppf(q) ** 2 + 1) ** 2
    return n ** (-1.0 / 5) * (num / den) ** (1.0 / 5)


def chamberlain(n: float, q: float, alpha: float = 0.05) -> float:
    """Chamberlain (1994) bandwidth rule, statsmodels' ``'chamberlain'``."""
    return norm.ppf(1 - alpha / 2) * np.sqrt(q * (1 - q) / n)


BANDWIDTH_RULES: dict[str, Callable[[float, float], float]] = {
    "hsheather": hall_sheather,
    "bofinger": bofinger,
    "chamberlain": chamberlain,
}


def _score_at_percentile(a: np.ndarray, per: float) -> float:
    """``scipy.stats.scoreatpercentile(a, per)`` with ``'fraction'`` interpolation.

    Re-implemented rather than called because ``scoreatpercentile`` is
    documented by scipy as "will become obsolete in the future", and because
    ``np.percentile(a, per)`` — the recommended replacement — is *not* a
    bit-identical substitute: it takes a different floating-point path and
    disagrees in the last bit (measured: 4.4e-16 on the interquartile range
    of a 240-observation residual vector). That bit propagates into the
    bandwidth, the sparsity and every standard error, so the substitution
    would cost parity for nothing.

    Follows ``scipy/stats/_stats_py.py::_compute_qth_percentile`` for the
    scalar, one-dimensional, no-``limit`` case.
    """
    sorted_ = np.sort(np.asarray(a))
    idx = per / 100.0 * (sorted_.shape[0] - 1)
    i = int(idx)
    if i == idx:
        return float(sorted_[i])
    # Linear interpolation between the bracketing order statistics. The
    # division by ``sumval`` looks redundant (the weights sum to one in exact
    # arithmetic) but is not in floating point, and scipy does it.
    weights = np.array([(i + 1 - idx), (idx - i)], float)
    sumval = weights.sum()
    return float(np.add.reduce(sorted_[i:i + 2] * weights) / sumval)


def _make_exog_names(exog: np.ndarray) -> tuple[str, ...]:
    """Column names statsmodels invents for an *unlabelled* design.

    Transcribed from ``statsmodels/base/data.py::_make_exog_names`` (:643).
    The columns are ``x1 … xk`` — *except* that a zero-variance column is
    called ``const`` and the numbering closes up around it, so
    ``sm.QuantReg(y, np.column_stack([ones, x])).model.exog_names`` is
    ``['const', 'x1']`` and not ``['x1', 'x2']``. The corpus reads
    ``params['const']`` off exactly that kind of design, so the off-by-one is
    load-bearing rather than cosmetic.

    ``exog`` is assumed finite: ``_prepare`` calls this only after the
    non-finite refusal, which is also the order statsmodels evaluates them in
    (``xnames`` is a lazy property; ``_handle_constant`` raises first).
    """
    exog_var = exog.var(0)
    if (exog_var == 0).any():
        # statsmodels' own comment here reads "assumes one constant in first
        # or last position / avoid exception if more than one constant".
        const_idx = int(exog_var.argmin())
        names = [f"x{i}" for i in range(1, exog.shape[1])]
        names.insert(const_idx, "const")
    else:
        names = [f"x{i}" for i in range(1, exog.shape[1] + 1)]
    return tuple(names)


def _label_names(X) -> tuple | None:
    """Column labels statsmodels *keeps*, or ``None`` when it invents them.

    ``ModelData._get_names`` (base/data.py:410) hands back
    ``list(df.columns)`` **unconverted**: integer labels stay integers, so a
    caller who built the design with ``pd.DataFrame(np.column_stack([...]))``
    can still read ``params[1]``. Stringifying them here — which this function
    used to do — turned that into a ``KeyError``. A ``MultiIndex`` is the one
    label type statsmodels rewrites, joining the levels with underscores
    (``('lev', 'const')`` becomes ``'lev_const'``).

    A ``Series`` contributes its ``name``, but only when the name is *truthy*:
    statsmodels writes ``if arr.name:``, so a Series called ``0`` or ``''``
    falls through to :func:`_make_exog_names` exactly as an unnamed one does.
    Reproduced rather than corrected, because a call site that indexes by name
    has to find the same name here that it found there.
    """
    if isinstance(X, pd.DataFrame):
        if isinstance(X.columns, pd.MultiIndex):
            return tuple("_".join(level for level in c if level) for c in X.columns)
        return tuple(X.columns)
    if isinstance(X, pd.Series):
        return (X.name,) if X.name else None
    return None


# Refusal threshold for the condition number of X'X. The number is the one
# ``_linalg.inv_xtx`` names in its own error text: past cond(X'X) ≈ 1e14 fewer
# than three of the sixteen decimal digits in a float64 coefficient survive,
# which is below what any published number quotes.
_COND_MAX = 1e14


def _reject_ill_conditioned(exog: np.ndarray, names, *, name: str = "quantreg") -> float:
    """Refuse a design whose ``X'X`` is too ill-conditioned to estimate.

    This runs *in addition to* :func:`puremacro._linalg.inv_xtx`, not instead
    of it, because the two gates catch different things and neither is a
    superset of the other.

    ``inv_xtx`` compares the smallest and largest entries on the diagonal of
    the Cholesky factor. That ratio is a lower bound on ``cond(L)``, and a
    weak one: it is driven by the *scale* of the columns rather than by their
    dependence. The archetypal macro design — ``[1, t, t²]`` for ``t`` a
    calendar year — has a diagonal ratio of 2.7e-3, comfortably inside the
    1e-6 gate, at ``cond(X'X) = 1.7e21``. That design passed, and the
    coefficients that came back were 82% wrong on the quadratic term with
    standard errors understated by a factor of 108: silent garbage, which is
    the one thing this package promises not to return. Hence a real condition
    number here.

    Computed from the singular values of ``X`` rather than of ``X'X``:
    forming ``X'X`` squares the condition number *and* loses half the
    available precision, so ``np.linalg.cond(X.T @ X)`` is itself unreliable
    exactly where the answer matters — it reports 2.0e21 for the design
    above, against the 1.7e21 the singular values of ``X`` give. Nothing
    hinges on which of the two is right, seven orders past the gate, but the
    more accurate one is no more expensive.

    Parameters
    ----------
    exog : ndarray, shape (n, k)
        The design, already finite and with at least one column.
    names : sequence
        Column labels, used only to make the error message readable.
    name : str
        Caller label prefixed onto the error message.

    Returns
    -------
    float
        ``cond(X'X)``, for callers that want to report it.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``cond(X'X)`` exceeds ``1e14``.
    """
    sv = np.linalg.svd(exog, compute_uv=False)
    smin, smax = float(sv.min()), float(sv.max())
    cond = np.inf if smin <= 0.0 else (smax / smin) ** 2
    if cond <= _COND_MAX:
        return float(cond)

    # Only now pay for the singular vectors: the last one spans the direction
    # X'X is nearly blind to, and its largest loadings name the columns that
    # are nearly dependent.
    _, _, vh = np.linalg.svd(exog, full_matrices=False)
    loadings = np.abs(vh[-1])
    culprits = [names[i] for i in np.argsort(loadings)[::-1][:2]]
    raise np.linalg.LinAlgError(
        f"{name}: X'X is numerically singular at full rank {exog.shape[1]} "
        f"(condition number {cond:.3g}, gate {_COND_MAX:.0e}). The columns "
        f"carrying the near-dependency are {culprits}. Centre or rescale "
        f"them — a polynomial in calendar year is the usual culprit, and "
        f"subtracting the mean of t fixes it. statsmodels does not refuse "
        f"here: it goes through pinv and returns coefficients whose leading "
        f"digits are rounding error, with standard errors to match."
    )


def _k_constant(X: np.ndarray) -> int:
    """Number of constant columns as statsmodels counts them (0 or 1).

    Mirrors ``statsmodels/base/data.py::_handle_constant`` for the
    ``hasconst=None`` path, which is what ``QuantReg(y, X)`` takes: a column
    whose max equals its min is constant; if several are, the one that is a
    column of ones wins, else the first non-zero one; if none is, an
    *implicit* constant is detected by a rank comparison against the
    design augmented with a column of ones.

    Only ``df_model`` depends on this, but ``df_model`` is on the
    statsmodels result surface, so it is reproduced rather than guessed.

    ``X`` is assumed finite; ``_prepare`` has already rejected it otherwise,
    which is the same refusal statsmodels makes inside ``_handle_constant``.
    """
    if X.size == 0:
        return 0
    exog_max = np.max(X, axis=0)
    exog_min = np.min(X, axis=0)
    const_idx = np.where(exog_max == exog_min)[0].squeeze()
    k_const = int(const_idx.size)

    check_implicit = False
    if k_const == 1:
        if X[:, const_idx].mean() == 0:
            check_implicit = True  # a column of zeros is not a constant
    elif k_const > 1:
        values = []
        for idx in np.atleast_1d(const_idx):
            value = X[:, idx].mean()
            if value == 1:
                return 1
            values.append(value)
        pos = np.array(values) != 0
        if pos.any():
            return 1
        check_implicit = True
    else:
        check_implicit = True

    if check_implicit:
        augmented = np.column_stack((np.ones(X.shape[0]), X))
        return int(np.linalg.matrix_rank(X) == np.linalg.matrix_rank(augmented))
    return k_const


@dataclass(frozen=True)
class QuantRegResult:
    """Result of :func:`quantreg`, with statsmodels' attribute names.

    A call site that reads ``q90.params["sigma_s"]``, ``q90.bse``,
    ``q90.pvalues`` or ``q90.conf_int()`` off a ``smf.quantreg`` fit changes
    only its constructor line.

    Attributes
    ----------
    params : pandas.Series or ndarray
        Coefficients, shape ``(k,)``. A ``Series`` indexed by column name when
        **either** ``y`` or ``X`` was pandas, an ndarray only when neither was
        — statsmodels' convention, which is decided by
        ``handle_data_class_factory(endog, exog)`` and therefore by both
        arguments, not by the design alone.
    bse, tvalues, pvalues : pandas.Series or ndarray
        Standard errors, ``params/bse``, and two-sided p values from
        ``t(df_resid)`` (``use_t`` is ``True``, as in statsmodels).
    resid, fittedvalues : pandas.Series or ndarray
        ``y - X @ params`` and ``X @ params``, shape ``(n,)``. Indexed by
        ``X``'s row labels when ``X`` is pandas, else by ``y``'s.
    vcov : ndarray
        The ``(k, k)`` sandwich; also reachable as :meth:`cov_params`. The
        array is **read-only**: ``bse``, ``tvalues`` and ``pvalues`` are
        already derived from it, so a write here would leave two accessors
        describing different covariances. Copy it if you need to modify one.
    nobs, df_resid, df_model : float
        ``n``, ``n - rank(X)``, ``rank(X) - k_constant``. Floats, as in
        statsmodels.
    rank : int
        ``matrix_rank(X)``.
    q : float
        The quantile that was fitted.
    iterations : int
        IRLS iterations actually run.
    converged : bool
        ``True`` when the loop stopped because ``max|Δβ| ≤ tol``. statsmodels
        has no such flag; it only warns.
    sparsity : float
        ``1/f̂(0)``, the estimated reciprocal density of the residuals at zero.
    bandwidth : float
        The *rescaled* kernel bandwidth ``h`` actually used, i.e.
        ``min(std(y), IQR(e)/1.34) · (Φ⁻¹(q+h₀) - Φ⁻¹(q-h₀))``, not the raw
        rule output ``h₀``. Matches ``QuantRegResults.bandwidth``.
    prsquared : float
        Koenker-Machado pseudo-R², ``1 - Σρ_q(e) / Σρ_q(y - Q_q(y))``.
    scale : float
        Always ``1.0``. The sandwich is already a covariance, and statsmodels
        multiplies ``normalized_cov_params`` by this same ``1.0``.
    cov_type : str
        ``'robust'`` or ``'iid'`` — the ``vcov`` that was requested. **This is
        a deliberate divergence**: statsmodels reports ``'nonrobust'`` here
        whatever you pass, because ``QuantReg.fit`` smuggles the sandwich in
        through ``normalized_cov_params`` and never sets ``cov_type``. The
        number it labels is a robust sandwich; the label is wrong.
    use_t : bool
        Always ``True``, matching statsmodels.
    kernel, bandwidth_rule : str
        The density-estimation choices behind ``sparsity`` and ``bandwidth``.
    exog_names : tuple
        Column labels, in design order — **not** necessarily strings. A
        DataFrame's labels are carried over untouched (integer labels stay
        integers, a MultiIndex is flattened with underscores, as statsmodels
        does); an unlabelled design gets statsmodels' invented names, which
        are ``x1 … xk`` with any zero-variance column renamed ``const``.

    Notes
    -----
    ``rsquared``, ``rsquared_adj``, ``llf``, ``aic`` and ``bic`` are exposed
    and return ``nan``. That is not a shortcut — ``QuantRegResults`` overrides
    all five to ``nan`` too (quantile_regression.py:284-302), because a
    least-squares R² and a Gaussian likelihood are undefined for a check-loss
    fit. Use ``prsquared``.
    """

    params: pd.Series | np.ndarray
    bse: pd.Series | np.ndarray
    tvalues: pd.Series | np.ndarray
    pvalues: pd.Series | np.ndarray
    resid: pd.Series | np.ndarray
    fittedvalues: pd.Series | np.ndarray
    vcov: np.ndarray
    nobs: float
    df_resid: float
    df_model: float
    rank: int
    q: float
    iterations: int
    converged: bool
    sparsity: float
    bandwidth: float
    prsquared: float
    cov_type: str
    use_t: bool
    kernel: str
    bandwidth_rule: str
    exog_names: tuple
    scale: float = 1.0

    @property
    def _as_pandas(self) -> bool:
        """Whether the caller passed a labelled design.

        Derived from ``params`` rather than stored, so it does not become a
        dataclass field: ``tests/test_public_api.py`` snapshots every field of
        every public ``*Result`` class, and a private flag has no business in
        a public-API freeze.
        """
        return isinstance(self.params, pd.Series)

    # -- statsmodels result surface -------------------------------------
    def cov_params(self) -> pd.DataFrame | np.ndarray:
        """Parameter covariance, ``(k, k)``.

        A DataFrame labelled by ``exog_names`` when ``X`` was a DataFrame,
        an ndarray otherwise — statsmodels' convention.
        """
        if self._as_pandas:
            return pd.DataFrame(self.vcov, index=list(self.exog_names),
                                columns=list(self.exog_names))
        return self.vcov

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame | np.ndarray:
        """Two-sided ``1-alpha`` interval, ``params ± t_{1-α/2}(df_resid)·bse``.

        Columns are labelled ``0`` and ``1``, which is what statsmodels
        returns for a DataFrame design (``conf_int()[0]`` is the lower bound).
        """
        crit = stats.t.ppf(1 - alpha / 2.0, self.df_resid)
        b = np.asarray(self.params, dtype=float)
        se = np.asarray(self.bse, dtype=float)
        out = np.column_stack([b - crit * se, b + crit * se])
        if self._as_pandas:
            return pd.DataFrame(out, index=list(self.exog_names), columns=[0, 1])
        return out

    @property
    def rsquared(self) -> float:
        """``nan`` — undefined for a check-loss fit, as in statsmodels."""
        return float("nan")

    @property
    def rsquared_adj(self) -> float:
        """``nan`` — see :attr:`rsquared`."""
        return float("nan")

    @property
    def llf(self) -> float:
        """``nan`` — no Gaussian likelihood here, as in statsmodels."""
        return float("nan")

    @property
    def aic(self) -> float:
        """``nan`` — follows from :attr:`llf`."""
        return float("nan")

    @property
    def bic(self) -> float:
        """``nan`` — follows from :attr:`llf`."""
        return float("nan")

    def summary(self) -> str:
        """A plain-text coefficient table.

        Returns a ``str``, not a ``statsmodels.iolib.summary.Summary``: there
        is no Summary class in puremacro, and a string prints identically.
        """
        b = np.asarray(self.params, dtype=float)
        se = np.asarray(self.bse, dtype=float)
        t = np.asarray(self.tvalues, dtype=float)
        p = np.asarray(self.pvalues, dtype=float)
        ci = np.asarray(self.conf_int(), dtype=float)
        # str() because a label need not be one: a DataFrame with the default
        # integer columns gives exog_names = (0, 1, 2), which statsmodels
        # keeps as integers and so do we.
        labels = [str(n) for n in self.exog_names]
        width = max([len(n) for n in labels] + [8])
        lines = [
            f"QuantReg (q = {self.q:.3g}), IRLS, {self.cov_type} sparsity covariance",
            f"  n = {self.nobs:.0f}   df_resid = {self.df_resid:.0f}   "
            f"df_model = {self.df_model:.0f}   pseudo-R2 = {self.prsquared:.4f}",
            f"  iterations = {self.iterations} ({'converged' if self.converged else 'NOT converged'})"
            f"   bandwidth = {self.bandwidth:.4g}   sparsity = {self.sparsity:.4g}",
            "",
            f"{'':<{width}}  {'coef':>12} {'std err':>10} {'t':>9} {'P>|t|':>8} "
            f"{'[0.025':>11} {'0.975]':>11}",
        ]
        for i, name in enumerate(labels):
            lines.append(
                f"{name:<{width}}  {b[i]:>12.6f} {se[i]:>10.5f} {t[i]:>9.3f} "
                f"{p[i]:>8.3f} {ci[i, 0]:>11.5f} {ci[i, 1]:>11.5f}"
            )
        return "\n".join(lines) + "\n"


def _resolve_bandwidth(bandwidth) -> tuple[Callable[[float, float], float], str]:
    """Map the ``bandwidth=`` argument onto a rule callable and its label."""
    if bandwidth is None:
        return BANDWIDTH_RULES["hsheather"], "hsheather"
    if isinstance(bandwidth, str):
        if bandwidth not in BANDWIDTH_RULES:
            raise ValueError(
                f"quantreg: bandwidth must be one of "
                f"{sorted(BANDWIDTH_RULES)}, a positive float, or None "
                f"(= 'hsheather'); got {bandwidth!r}."
            )
        return BANDWIDTH_RULES[bandwidth], bandwidth
    h0 = float(bandwidth)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError(
            f"quantreg: a numeric bandwidth must be finite and strictly "
            f"positive; got {bandwidth!r}."
        )
    return (lambda n, q: h0), f"fixed({h0:g})"


def _prepare(y, X, missing: str):
    """Coerce ``(y, X)`` to float arrays, apply ``missing``, recover names."""
    if missing not in ("none", "drop", "raise"):
        raise ValueError(
            f"quantreg: missing must be 'none', 'drop' or 'raise'; got {missing!r}."
        )

    # Which container comes back is decided by BOTH arguments, not by X alone.
    # statsmodels picks its data handler with
    # ``handle_data_class_factory(endog, exog)`` (base/data.py:666), which
    # selects ``PandasData`` as soon as *either* input is pandas
    # (``tools/data.py::_is_using_pandas``). Keying off X alone returned bare
    # ndarrays for the commonest porting shape of all —
    # ``X = np.column_stack([...])`` with ``y = df["col"]`` — where
    # statsmodels returns a Series and the call site reads
    # ``params["const"]``.
    y_is_pandas = isinstance(y, (pd.Series, pd.DataFrame))
    x_is_pandas = isinstance(X, (pd.DataFrame, pd.Series))
    as_pandas = y_is_pandas or x_is_pandas

    y_index = y.index if y_is_pandas else None
    x_index = X.index if x_is_pandas else None
    if y_index is not None and x_index is not None and not y_index.equals(x_index):
        # ``PandasData._check_integrity`` (base/data.py:543-551) refuses this
        # outright, and it is right to: joining two labelled objects by
        # position when their labels disagree regresses y on somebody else's
        # regressors and hands back a residual series wearing X's labels over
        # y's values. Nothing about the answer looks wrong.
        absent = int(np.sum(~y_index.isin(x_index)))
        if absent == 0 and len(y_index) == len(x_index):
            detail = "they carry the same labels in a different order"
        else:
            detail = (
                f"{absent} of y's {len(y_index)} labels are absent from X's "
                f"{len(x_index)}"
            )
        raise ValueError(
            "quantreg: the indices for endog and exog are not aligned "
            f"({detail}). statsmodels raises here too. Align them yourself — "
            "`y = y.reindex(X.index)`, or `y, X = y.align(X, join='inner', "
            "axis=0)` — so the join is the one you intended rather than a "
            "positional one."
        )

    row_index = x_index if x_index is not None else y_index
    names = _label_names(X)

    if isinstance(X, pd.Series):
        Xa = X.to_numpy(dtype=float).reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        Xa = X.to_numpy(dtype=float)
    else:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)

    ya = np.asarray(y, dtype=float).ravel()

    if Xa.ndim != 2:
        raise ValueError(f"quantreg: X must be 2-dimensional; got shape {Xa.shape}.")
    if Xa.shape[1] == 0:
        # Otherwise this surfaces from inside the singularity gate as a bare
        # "zero-size array to reduction operation minimum", which names
        # neither the caller nor the problem.
        raise ValueError(
            f"quantreg: X has no columns (shape {Xa.shape}); there is nothing "
            "to estimate. Pass a design with at least one column — including "
            "the constant, which is not added for you."
        )
    if ya.shape[0] != Xa.shape[0]:
        raise ValueError(
            f"quantreg: y has {ya.shape[0]} rows but X has {Xa.shape[0]}."
        )

    if missing != "none":
        # NaN, not "non-finite". statsmodels' ``_nan_rows`` (base/data.py:38)
        # is built on ``pandas.isnull``, which does not consider an infinity
        # missing, and dropping inf rows here moved numbers: a y with one inf
        # returned a plausible finite fit from 199 rows where statsmodels
        # reports nobs=200 and params=[nan, nan]. PARITY_SPEC §7 — do not
        # quietly ship a number that moved.
        bad = np.isnan(ya) | np.isnan(Xa).any(axis=1)
        if bad.any():
            if missing == "raise":
                raise ValueError(
                    f"quantreg: {int(bad.sum())} of {bad.size} rows contain "
                    f"NaN in y or X (missing='raise')."
                )
            ya = ya[~bad]
            Xa = Xa[~bad]
            if row_index is not None:
                row_index = row_index[~bad]

    # A non-finite X is refused whatever `missing` says — after 'drop' there
    # may still be an infinity, and this is exactly where statsmodels raises
    # MissingDataError (base/data.py::_handle_constant). Doing it here rather
    # than downstream keeps the message useful: an inf in the design
    # otherwise surfaces as "SVD did not converge" from the singularity gate.
    if not np.isfinite(Xa).all():
        raise ValueError(
            "quantreg: X contains inf or NaN. statsmodels raises "
            "MissingDataError here (an inf is not 'missing' to it, so "
            "missing='drop' does not remove it either); drop or replace the "
            "offending rows yourself."
        )

    if Xa.shape[0] == 0:
        raise ValueError("quantreg: no observations left after applying missing=.")

    if not np.isfinite(ya).all():
        # statsmodels checks exog and not endog, and quietly returns a vector
        # of NaNs for a non-finite y. Match the numbers, say so out loud.
        # Reached under missing='none' (NaN or inf) and under 'drop' (inf
        # only — NaN rows are gone by now).
        warnings.warn(
            f"quantreg: y contains non-finite values after missing={missing!r}, "
            "so every coefficient will be NaN. missing='drop' removes NaN "
            "rows, matching statsmodels; an infinity is not missing to "
            "statsmodels and is never dropped, so filter or replace it "
            "yourself.",
            UserWarning,
            stacklevel=3,
        )

    if not names:
        # Computed from the *post-drop* design, because statsmodels' ``xnames``
        # is a lazy property reading ``self.exog``, which handle_missing has
        # already filtered. Only matters when dropping rows changes which
        # columns are constant.
        names = _make_exog_names(Xa)
    return ya, Xa, names, row_index, as_pandas


def quantreg(
    y,
    X,
    q: float = 0.5,
    *,
    missing: str = "none",
    max_iter: int = 1000,
    tol: float = 1e-6,
    vcov: str = "robust",
    kernel: str = "epa",
    bandwidth=None,
) -> QuantRegResult:
    """Fit a quantile regression by IRLS, matching statsmodels' ``QuantReg``.

    Solves ``min_β Σ_i ρ_q(y_i - x_i'β)`` with ``ρ_q(u) = u(q - 1{u<0})`` by
    iteratively reweighted least squares, then estimates the asymptotic
    covariance with the Powell / Koenker-Machado sparsity sandwich (Greene
    2008, pp. 407-408) — a kernel density of the residuals at zero,
    bandwidth from the Hall-Sheather rule by default.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response. A ``pandas.Series`` is accepted; its index is carried onto
        ``resid`` and ``fittedvalues`` when ``X`` has no index of its own, and
        a pandas ``y`` is enough on its own to make ``params`` a Series.
        If **both** ``y`` and ``X`` are pandas their indices must be equal —
        a mismatch is refused rather than joined by position.
    X : array-like, shape (n, k)
        Design matrix, **including the constant column if you want one** —
        there is no formula parser and nothing is added for you. Pass a
        ``pandas.DataFrame`` to choose the coefficient labels: the column
        labels become the index of ``params`` / ``bse`` / ``pvalues``
        unchanged, so a call site porting from
        ``smf.quantreg("y ~ x + I(x**2)", d)`` can keep reading
        ``params["I(x ** 2)"]`` by naming the column that, and a design built
        with ``pd.DataFrame(np.column_stack([...]))`` keeps working with
        ``params[1]``. An unlabelled design gets statsmodels' invented names
        (``x1 … xk``, with a zero-variance column renamed ``const``).
    q : float, default 0.5
        Quantile, strictly between 0 and 1. ``q=0.5`` is least absolute
        deviations.
    missing : {'none', 'drop', 'raise'}, default 'none'
        Row handling for **NaN**, with statsmodels' semantics. ``'none'`` does
        no filtering; ``'drop'`` drops rows with a NaN in ``y`` or ``X``;
        ``'raise'`` raises. An infinity is *not* missing — statsmodels'
        ``_nan_rows`` is built on ``pandas.isnull``, so an inf is never
        dropped by any of the three. An inf in ``X`` is refused outright
        (statsmodels raises ``MissingDataError`` in ``_handle_constant``); an
        inf in ``y`` makes every coefficient NaN, which statsmodels does
        silently and this function warns about.
    max_iter : int, default 1000
        IRLS iteration cap. statsmodels' ``max_iter``.
    tol : float, default 1e-6
        Convergence tolerance on ``max|Δβ|``. statsmodels' ``p_tol``.
    vcov : {'robust', 'iid'}, default 'robust'
        ``'robust'`` gives the heteroskedasticity-robust sandwich
        ``(X'X)⁻¹ X'DX (X'X)⁻¹`` with ``D = diag(q²/f̂₀²)`` above the fit and
        ``((1-q)/f̂₀)²`` below it; ``'iid'`` gives Stata's
        ``(1/f̂₀)² q(1-q) (X'X)⁻¹``.
    kernel : {'epa', 'biw', 'cos', 'gau', 'par'}, default 'epa'
        Kernel for the residual density at zero. ``'epa'`` (Epanechnikov) is
        statsmodels' default.
    bandwidth : None, str or float, default None
        ``None`` means ``'hsheather'`` (Hall-Sheather), statsmodels' default;
        ``'bofinger'`` and ``'chamberlain'`` are the other two rules. A
        positive float replaces the *rule output* ``h₀`` — the quantity that
        goes into ``Φ⁻¹(q ± h₀)`` — not the rescaled bandwidth reported as
        ``result.bandwidth``. That is the same slot statsmodels' rules plug
        into, so a fixed value stays comparable with a rule-chosen one.

    Returns
    -------
    QuantRegResult
        Frozen dataclass carrying statsmodels' attribute names; see its
        docstring for the full surface.

    Raises
    ------
    ValueError
        If ``q`` is not strictly inside ``(0, 1)``; if ``kernel``,
        ``bandwidth``, ``vcov`` or ``missing`` is not a recognised choice; if
        ``y`` and ``X`` disagree on length; if ``y`` and ``X`` are both
        pandas and their indices are not equal; if ``X`` has no columns; if
        ``X`` holds a non-finite value; or if ``missing='raise'`` and any row
        holds a NaN.
    numpy.linalg.LinAlgError
        If ``X'X`` is singular or near-singular. Two gates run, both before
        any arithmetic: :func:`puremacro._linalg.inv_xtx` for rank deficiency
        and a bad Cholesky, then a condition-number test that refuses
        ``cond(X'X) > 1e14``. The second exists because the first is a ratio
        of Cholesky pivots, which measures column scale more than column
        dependence and passed a raw calendar-year polynomial at
        ``cond(X'X) = 2e21``; see the Notes.
        **statsmodels does not raise for either** — it goes through ``pinv``
        and silently returns the minimum-norm solution. Refusing is the house
        contract (CONTRIBUTING, "Diagnostic error contract"), and it is the
        one input class on which this function deliberately does not match
        statsmodels.

    Warns
    -----
    UserWarning
        On hitting ``max_iter``; on a detected convergence cycle; when the
        bandwidth rescaling puts ``q ± h₀`` outside ``(0, 1)`` (which makes
        every standard error NaN, silently, in statsmodels); when the kernel
        density at zero comes out non-positive; and when non-finite values
        survive in ``y`` (a NaN under ``missing='none'``, an infinity under
        any setting).

    Notes
    -----
    **Parity.** The IRLS loop, the bandwidth, the kernel and both covariance
    branches are transcribed from statsmodels 0.14.6
    ``regression/quantile_regression.py:138-213`` operation for operation,
    including the floor that resets residuals with ``|e| < 1e-6`` to
    ``±1e-6`` and the ``pinv`` (not ``solve``) used at each step. On a
    240-observation, 3-regressor design the agreement with
    ``sm.QuantReg(y, X).fit(q=τ)`` is **exact in double precision** — 0.0
    maximum absolute difference in ``params``, ``bse``, ``bandwidth`` and
    ``sparsity``, and the same iteration count — at every
    ``τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}``. The test file asserts a weaker
    ``atol=1e-6``, because bit-exactness across BLAS builds is not something
    to promise.

    **The IRLS answer is not the LP answer.** IRLS floors the residual
    magnitude at ``1e-6`` to keep the weights finite, so it converges to a
    neighbourhood of the Koenker-Bassett optimum rather than landing on the
    exact simplex vertex. Measured against ``puremacro.lp.quantile._qreg``,
    which solves that LP exactly:

    ==========================  =================  ===================
    design                      coefficient gap    check-loss excess
    ==========================  =================  ===================
    n=200, k=2, q ∈ {.1,.5,.9}  3e-7 … 2e-6        ~4e-7 absolute
    n=120, k=3, q ∈ {.1 … .9}   7e-7 … 9.3e-5      7e-7 … 2.1e-6
                                                   (2e-8 … 9e-8 relative)
    ==========================  =================  ===================

    The LP is, as it must be, the better optimiser, by an amount far below
    any standard error (the smallest ``bse`` on the second design is O(0.05),
    three to four orders above the gap). Two consequences worth stating:
    the coefficient gap is much larger than the objective gap, because the
    check loss is flat near its optimum; and where a published number is a
    *coefficient ratio* the gap is amplified — the turning point
    ``-β₁/(2β₂)`` of a quadratic divides one small number by another. Check
    a ratio before quoting it to more than three significant figures.

    **Ill-conditioned designs are refused, and the commonest one is a trend.**
    ``[1, t, t²]`` on raw calendar years has ``cond(X'X) = 1.7e21``: the fit
    statsmodels returns for it has 82% error on the quadratic coefficient,
    standard errors 108 times too small, and a *worse* check loss than the
    same model fitted on ``t - t.mean()``. It looks like an estimate. This
    function refuses it, names the columns, and tells you to centre them.
    The threshold is ``cond(X'X) > 1e14``, measured from the singular values
    of ``X`` (forming ``X'X`` first would square the conditioning and lose
    half the precision available to judge it). Centring or rescaling costs
    nothing statistically — the fitted values are identical, only the
    coordinates change — so the fix is always available.

    This function refuses a little more than :func:`puremacro.regress.ols.ols`
    does, and the difference is deliberate. ``ols`` scales its columns to unit
    norm and estimates in that space, so it accepts a design that is merely
    badly *scaled* — an intercept beside a population in persons, where
    ``cond(X'X)`` is 1.1e16 raw and 3.9 equilibrated. ``quantreg`` cannot do
    that: bit-parity with statsmodels means running statsmodels' IRLS on the
    array the caller passed, and the residual floor of 1e-6 in that loop is
    in the units of the raw data, so rescaling internally would change the
    answer. Estimating in raw coordinates is therefore the constraint, and a
    design too ill-conditioned for raw coordinates is refused rather than
    fitted — statsmodels fits that population design and returns an intercept
    of 5e-16 for a true 0.86, at a check loss 20% worse than the same model
    on scaled columns. Rescale the design yourself before calling, and the
    coefficients divide back out exactly.

    **Inference is asymptotic.** ``use_t`` is ``True`` and the p values come
    from ``t(n - rank)``, which is what statsmodels does, but the sandwich is
    asymptotic in all three factors — the density at zero especially. In
    small samples the sparsity estimate is the weak link, not the Student-t
    correction.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x + rng.normal(size=n)
    >>> design = pd.DataFrame({"const": 1.0, "x": x})
    >>> res = quantreg(y, design, q=0.9)
    >>> float(res.params["x"]).__round__(3)
    0.501
    >>> res.nobs
    300.0

    A quadratic, named the way patsy would name it, so a ported call site
    keeps working:

    >>> design2 = pd.DataFrame({"const": 1.0, "x": x, "I(x ** 2)": x ** 2})
    >>> res2 = quantreg(y, design2, q=0.9)
    >>> turning_point = -res2.params["x"] / (2 * res2.params["I(x ** 2)"])
    >>> bool(np.isfinite(turning_point))
    True
    """
    if not (0 < q < 1):
        # statsmodels raises a bare Exception here; ValueError is the same
        # refusal with a catchable type.
        raise ValueError(f"quantreg: q must be strictly between 0 and 1; got {q!r}.")
    if kernel not in KERNELS:
        raise ValueError(
            f"quantreg: kernel must be one of {sorted(KERNELS)}; got {kernel!r}."
        )
    if vcov not in ("robust", "iid"):
        raise ValueError(f"quantreg: vcov must be 'robust' or 'iid'; got {vcov!r}.")
    kern = KERNELS[kernel]
    bw_rule, bw_label = _resolve_bandwidth(bandwidth)

    endog, exog, names, row_index, as_pandas = _prepare(y, X, missing)
    nobs = float(exog.shape[0])

    # Diagnostic gate, in two parts. inv_xtx raises a LinAlgError naming the
    # columns most aligned with the null space; statsmodels would min-norm
    # through pinv and hand back garbage that looks like an estimate. The
    # inverse it returns is deliberately discarded: the sandwich below uses
    # pinv so the arithmetic stays bit-identical to statsmodels on designs
    # that pass the gate. See ARCHITECTURE.md, "Diagnostic errors over silent
    # garbage".
    #
    # inv_xtx alone is not enough, and the second call is not belt-and-braces.
    # Its test is a ratio of Cholesky pivots, which tracks column *scale*
    # rather than column dependence, so it let a raw calendar-year polynomial
    # through at cond(X'X)=2e21 — seven orders past the threshold its own
    # error message names. _reject_ill_conditioned closes that hole for this
    # estimator; the shared helper still has it for every other caller.
    inv_xtx(exog, name="quantreg")
    _reject_ill_conditioned(exog, names, name="quantreg")

    exog_rank = int(np.linalg.matrix_rank(exog))
    k_const = _k_constant(exog)  # only df_model depends on this
    df_model = float(exog_rank - k_const)
    df_resid = nobs - exog_rank

    # ---------------- IRLS ------------------------------------------------
    # statsmodels quantile_regression.py:145-192, transcribed. Two details
    # that look like accidents and are not:
    #   * beta starts at ones and xstar starts at exog, so the FIRST step is
    #     plain OLS and the initial beta only ever enters the convergence
    #     check.
    #   * the weights use ρ with q and 1-q swapped relative to the check
    #     function. That is correct, not a bug: at the fixed point the normal
    #     equation Σ xᵢ sign(rᵢ)/c̃ᵢ = 0 rearranges to Koenker-Bassett's
    #     Σ xᵢ (q - 1{rᵢ<0}) = 0.
    n_iter = 0
    xstar = exog
    beta = np.ones(exog.shape[1])
    diff = 10.0
    cycle = False
    history: list[np.ndarray] = []

    while n_iter < max_iter and diff > tol and not cycle:
        n_iter += 1
        beta0 = beta
        xtx = np.dot(xstar.T, exog)
        xty = np.dot(xstar.T, endog)
        beta = np.dot(np.linalg.pinv(xtx), xty)
        resid = endog - np.dot(exog, beta)

        # Floor |resid| away from zero so 1/resid stays finite. This floor is
        # exactly why IRLS is not the LP optimum.
        mask = np.abs(resid) < 0.000001
        resid[mask] = ((resid[mask] >= 0) * 2 - 1) * 0.000001
        resid = np.where(resid < 0, q * resid, (1 - q) * resid)
        resid = np.abs(resid)
        xstar = exog / resid[:, np.newaxis]
        diff = np.max(np.abs(beta - beta0))
        history.append(beta)

        if (n_iter >= 300) and (n_iter % 100 == 0):
            # A two-to-nine step limit cycle: the iteration is bouncing
            # between vertices instead of settling on one.
            for ii in range(2, 10):
                if ii <= len(history) and np.all(beta == history[-ii]):
                    cycle = True
                    warnings.warn(
                        f"quantreg: convergence cycle detected after {n_iter} "
                        f"iterations at q={q:g}; the reported coefficients are "
                        f"one vertex of the cycle.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

    if n_iter == max_iter:
        warnings.warn(
            f"quantreg: hit max_iter={max_iter} at q={q:g} with "
            f"max|Δβ|={diff:.3e} > tol={tol:g}; coefficients are not converged.",
            UserWarning,
            stacklevel=2,
        )
    converged = bool(diff <= tol) and not cycle

    # ---------------- sparsity and covariance ----------------------------
    # Greene (2008, pp. 407-408) as Stata 12 implements it, which is what
    # statsmodels ported: rescale the rule's h by the smaller of the response
    # scale and a robust residual scale, then read the residual density off
    # an Epanechnikov kernel at zero.
    e = endog - np.dot(exog, beta)
    iqre = _score_at_percentile(e, 75) - _score_at_percentile(e, 25)
    h0 = bw_rule(nobs, q)
    if not (0.0 < q - h0 and q + h0 < 1.0):
        warnings.warn(
            f"quantreg: bandwidth rule '{bw_label}' returned h0={h0:.4g} at "
            f"q={q:g}, so q±h0 falls outside (0, 1) and Φ⁻¹ is undefined. "
            f"Every standard error will be NaN (statsmodels does this "
            f"silently). n={nobs:.0f} is probably too small for q={q:g}; try "
            f"bandwidth='bofinger' or a smaller fixed bandwidth.",
            UserWarning,
            stacklevel=2,
        )
    h = min(np.std(endog), iqre / 1.34) * (norm.ppf(q + h0) - norm.ppf(q - h0))
    # A collapsed bandwidth (constant y, or a rule that returned an h0 the
    # rescaling drove to zero) would make numpy shout about division by zero
    # on top of the explicit, more informative warning below. Silence numpy,
    # not the diagnosis.
    with np.errstate(divide="ignore", invalid="ignore"):
        fhat0 = 1.0 / (nobs * h) * np.sum(kern(e / h))
    if not (fhat0 > 0):
        warnings.warn(
            f"quantreg: estimated residual density at zero is {float(fhat0):.6g} "
            f"(kernel='{kernel}', bandwidth={h:.4g}). Standard errors are not "
            f"usable. Too few residuals fall inside the kernel support, or "
            f"the bandwidth collapsed.",
            UserWarning,
            stacklevel=2,
        )

    # pinv, not inv_xtx: parity. The singularity gate already ran above.
    xtxi = np.linalg.pinv(np.dot(exog.T, exog))
    with np.errstate(divide="ignore", invalid="ignore"):
        if vcov == "robust":
            d = np.where(e > 0, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            V = xtxi @ xtdx @ xtxi
        else:  # 'iid'
            V = (1.0 / fhat0) ** 2 * q * (1 - q) * xtxi

        bse = np.sqrt(np.diag(V))
        tvalues = beta / bse
        sparsity = float(1.0 / fhat0)
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)

    # Koenker-Machado pseudo-R²: check loss of the fit against the check loss
    # of the unconditional q-quantile. QuantRegResults.prsquared:268-278.
    def _check_loss(u: np.ndarray) -> np.floating:
        # Deliberately a numpy scalar, not a Python float: a degenerate y
        # makes the denominator zero, and statsmodels' numpy arithmetic
        # yields -inf there while float division would raise. Parity means
        # returning -inf.
        w = np.where(u < 0, (1 - q) * u, q * u)
        return np.sum(np.abs(w))

    ered = endog - _score_at_percentile(endog, q * 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        prsquared = 1 - _check_loss(e) / _check_loss(ered)

    # The result is advertised as a frozen value, so the one array reachable
    # through two accessors that could disagree — .vcov and .cov_params() —
    # is made read-only rather than merely documented. bse, tvalues and
    # pvalues are already computed above, so an outside write to V could
    # otherwise leave cov_params() and bse describing different covariances.
    V.setflags(write=False)

    fitted = np.dot(exog, beta)
    if as_pandas:
        idx = list(names)
        params_out: pd.Series | np.ndarray = pd.Series(beta, index=idx)
        bse_out: pd.Series | np.ndarray = pd.Series(bse, index=idx)
        t_out: pd.Series | np.ndarray = pd.Series(tvalues, index=idx)
        p_out: pd.Series | np.ndarray = pd.Series(pvalues, index=idx)
        resid_out: pd.Series | np.ndarray = pd.Series(e, index=row_index)
        fitted_out: pd.Series | np.ndarray = pd.Series(fitted, index=row_index)
    else:
        params_out, bse_out, t_out, p_out = beta, bse, tvalues, pvalues
        resid_out, fitted_out = e, fitted

    return QuantRegResult(
        params=params_out,
        bse=bse_out,
        tvalues=t_out,
        pvalues=p_out,
        resid=resid_out,
        fittedvalues=fitted_out,
        vcov=V,
        nobs=nobs,
        df_resid=df_resid,
        df_model=df_model,
        rank=exog_rank,
        q=float(q),
        iterations=n_iter,
        converged=converged,
        sparsity=sparsity,
        bandwidth=float(h),
        prsquared=float(prsquared),
        cov_type=vcov,
        use_t=True,
        kernel=kernel,
        bandwidth_rule=bw_label,
        exog_names=tuple(names),
    )
