"""Maximum-likelihood binary-choice and count regression (Newton / IRLS).

The pure-numpy replacement for the two statsmodels entry points a
Pyodide notebook cannot import: ``sm.Logit(y, X).fit(disp=False)`` and
``sm.GLM(y, X, family=sm.families.Poisson()).fit(...)``. Both estimators
here are the *same* Fisher-scoring loop — for a canonical link Newton-
Raphson and IRLS are algebraically identical — so the two public
functions differ only in their link, variance function and
log-likelihood.

The design goal is that a call site changes only its constructor line.
:class:`DiscreteResult` therefore carries statsmodels' attribute names
(``params``, ``bse``, ``tvalues``, ``pvalues``, ``conf_int()``, ``nobs``,
``llf``, ``pearson_chi2``, ``df_resid``, ...), and ``params`` / ``bse`` /
``pvalues`` come back as **pandas Series indexed by column name**
whenever ``X`` is a DataFrame, because the corpus indexes them by name
(``logit_famine.params["nile_failure"]``, ``m_base.bse["dry"]``) and one
site registers those values with ``exp.claim``.

Relation to N19's hand-rolled ``poisson_hac``
---------------------------------------------
``N19_ming_collapse_1644_asian_megadrought.py:838`` defines::

    def poisson_hac(d, regressors=("dry", "log_nonfamine", "trend"),
                    lags=HAC_LAGS):
        X = sm.add_constant(d[list(regressors)])
        return sm.GLM(d["famine"], X, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})

Despite the name, that helper is not a hand-rolled estimator: it is a
two-line wrapper that builds the design and hands the whole problem to
statsmodels, and the notebook then reads ``.nobs``, ``.params``,
``.bse``, ``.conf_int()``, ``.pvalues``, ``.pearson_chi2`` and
``.df_resid`` off the statsmodels result (``N19:849-863``). This module
replaces the *inside* of that wrapper, not the wrapper: the drop-in is

>>> from puremacro.regress.ols import add_constant          # doctest: +SKIP
>>> from puremacro.regress.discrete import poisson          # doctest: +SKIP
>>> poisson(d["famine"], add_constant(d[list(regressors)]),
...         cov_type="HAC", cov_kwds={"maxlags": lags})     # doctest: +SKIP

and every attribute N19 reads survives with the same name and the same
number. The one number that does **not** survive to machine precision is
the *non-robust* ``bse``, which N19 never asks for — see Notes on
:func:`poisson`.

What is reproduced, and where the bodies are buried
---------------------------------------------------
Every default below was read out of statsmodels 0.14.6 source and then
confirmed by running it; the traps are all silent if you get them wrong.

* ``use_t`` is ``False`` for *both* models and for *every* ``cov_type``,
  including ``nonrobust``. This is the opposite of ``sm.OLS``, whose
  non-robust fit defaults to ``use_t=True``. p-values are normal-tail
  (``statsmodels/base/model.py:1359`` leaves ``_use_t = False`` and
  neither ``DiscreteResults`` nor ``GLMResults`` overrides it;
  ``genmod/generalized_linear_model.py:1639-1640`` sets it explicitly).
* The non-robust covariance is the inverse **observed information**
  ``(X' W X)^-1`` evaluated at the converged coefficients, with
  ``W = diag(p(1-p))`` for the logit and ``W = diag(mu)`` for the
  Poisson. There is no ``scale`` factor: the Poisson family has fixed
  scale 1.
* ``cov_type`` in ``{'HC0','HC1','HC2','HC3'}`` all produce the **same**
  matrix for these two model classes. ``base/covtype.py:237-244`` looks
  for a ``cov_HC1``-style attribute on the results object, finds none
  (only ``RegressionResults`` defines them) and falls back to
  ``sandwich_covariance.cov_white_simple(self, use_correction=False)``
  for every one of the four. So there is no ``n/(n-k)`` factor and no
  leverage correction here, unlike OLS. Verified numerically: all four
  give bit-identical ``bse``.
* ``cov_type='HAC'`` defaults to ``use_correction=False``
  (``base/covtype.py:255``) while ``cov_type='cluster'`` defaults to
  ``use_correction=True`` (``:273``) with the factor
  ``G/(G-1) * (n-1)/(n-k)``. A wrapper that hardcodes one convention
  moves half of any real corpus.
* ``cluster`` sets ``df_resid_inference = G - 1``
  (``base/covtype.py:364-366``). Because ``use_t`` is ``False`` that
  changes no p-value by default, but it is what ``conf_int()`` would use
  if you passed ``use_t=True``, so it is carried on the result.

Covariance machinery: what is borrowed and what is local
--------------------------------------------------------
The cluster sandwich is borrowed from :mod:`puremacro.regress.ols`
(``_cluster_cov``), whose signature ``(xu, bread, groups,
use_correction)`` is already generic in the score matrix and the bread,
so it applies unchanged to an MLE sandwich. ``_default_maxlags`` is
borrowed for the same reason: the HAC bandwidth fallback must be
literally the same expression in both modules or the two will drift.

The White and Newey-West meats are built locally, deliberately:

* ``ols._hc_cov`` cannot be reused because it implements the *OLS* HC
  family (``n/(n-k)`` for HC1, hat-matrix leverages for HC2/HC3), and
  statsmodels applies none of that to Logit/GLM results — reusing it
  would produce a confidently wrong HC1.
* ``inference._ols_helpers.ols_hac`` cannot be reused because it refits
  OLS internally and wraps its meat in the ``(X'X)^-1`` bread; an MLE
  sandwich needs the inverse information matrix ``(X'WX)^-1`` as bread
  and a score matrix that is not ``X * residual``.
* ``inference.dk.driscoll_kraay(xu, np.arange(n), lags)`` *would* return
  exactly the Newey-West meat (one observation per "period"), and is
  validated as such in ``validation/cases_inference.py``, but it forms
  one boolean mask per row and is therefore O(n^2). The six-line
  Bartlett loop below is the same object in O(n L).

Every matrix inversion goes through :func:`puremacro._linalg.inv_xtx`,
so a rank-deficient design fails with a message naming the collinear
columns rather than returning garbage (``CONTRIBUTING.md``,
"Diagnostic-error contract").

Failure policy
--------------
The two failure modes are treated differently, on purpose:

* **Perfect (or quasi-complete) separation raises**
  :class:`PerfectSeparationError`. The MLE does not exist; statsmodels
  warns and hands back whatever the 35th Newton iterate happened to be
  (a coefficient of -671 with a standard error of 150 507 in the test
  case in ``tests/test_regress_discrete_parity.py``). Returning that is
  exactly the "silent garbage" the package forbids.
* **Non-convergence warns** with :class:`ConvergenceWarning` and returns
  the last iterate with ``converged=False``, matching statsmodels. The
  caller can see the flag; the estimate is a real point on the
  likelihood surface, just not a stationary one.

Declared divergences from statsmodels
-------------------------------------
PARITY_SPEC §7: where a difference is principled, declare it rather than
ship it quietly. Every one of these is a *refusal* where statsmodels
returns a number, or a *diagnosis* where statsmodels returns the wrong
one; none of them moves a number that statsmodels computes correctly.

Each entry is *input* -- what statsmodels does -- what happens here.

* **A separated design** -- warns and returns the 35th Newton iterate --
  :class:`PerfectSeparationError`.
* **A** ``cov_kwds`` **key the covariance never reads** -- silently
  ignored, a TODO in its own source -- ``ValueError`` naming the key.
* **A float** ``cov_kwds['maxlags']`` -- ``TypeError`` from the ``range``
  in its kernel loop -- ``TypeError`` naming the key.
* **cov_type='cluster' with one cluster** -- ``ZeroDivisionError``, or,
  under ``use_correction=False``, standard errors at machine epsilon with
  no warning (measured: ``2.7e-17``) -- ``ValueError`` naming the cluster
  count.
* **n == k** -- fits the saturated, separated model -- ``ValueError``:
  not identified.
* **A non-finite** ``y`` **or** ``X`` **under** ``missing='none'`` --
  ``MissingDataError`` for a non-finite design, a family error for a
  non-finite response -- ``ValueError`` naming the array and the first
  offending row.
* **missing='drop'** alongside ``cov_kwds['groups']`` -- ``ValueError``
  from the mismatched lengths -- the labels are subset to the kept rows
  and the fit proceeds.

A design in awkward units — a population in persons beside an intercept —
is **not** on that list. Every inversion equilibrates the columns to unit
norm first (:func:`_equilibrated_bread`), which makes the package's
conditioning gate scale-free, so such a design is fitted and agrees with
statsmodels to ``1e-10`` rather than being refused or, worse, reported as
separation.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from puremacro.regress.discrete import logit, poisson
>>> rng = np.random.default_rng(0)
>>> n = 400
>>> X = pd.DataFrame({"const": 1.0, "x": rng.standard_normal(n)})
>>> y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + X["x"])))).astype(float)
>>> res = logit(y, X)
>>> bool(res.converged), res.use_t, list(res.params.index)
(True, False, ['const', 'x'])
>>> counts = rng.poisson(np.exp(0.5 + 0.2 * X["x"])).astype(float)
>>> pres = poisson(counts, X, cov_type="HAC", cov_kwds={"maxlags": 4})
>>> pres.cov_type
'HAC'

References
----------
McCullagh, P. and Nelder, J.A. (1989). Generalized Linear Models,
    2nd ed. Chapman and Hall. (IRLS; the canonical-link identity between
    Fisher scoring and Newton-Raphson.)
Albert, A. and Anderson, J.A. (1984). On the existence of maximum
    likelihood estimates in logistic regression models. Biometrika
    71(1), 1-10. (Separation.)
Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica 55(3), 703-708.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from .._linalg import inv_xtx
from .ols import _cluster_cov, _default_maxlags

__all__ = [
    "logit",
    "poisson",
    "DiscreteResult",
    "PerfectSeparationError",
    "ConvergenceWarning",
]


# Threshold at which "the model reproduces the data exactly" is declared.
# statsmodels' callback uses np.allclose(fitted - endog, 0), whose default
# atol is 1e-8 (discrete/discrete_model.py:216-227); the same number is
# used here so the two agree about *which* designs are separated, even
# though they then do different things about it.
_SEPARATION_ATOL = 1e-8

_HC_KINDS = ("HC0", "HC1", "HC2", "HC3")


class PerfectSeparationError(np.linalg.LinAlgError):
    """Raised when the likelihood has no interior maximum.

    Subclasses :class:`numpy.linalg.LinAlgError` so that callers already
    catching the package's diagnostic linear-algebra failures (see
    ``CONTRIBUTING.md``, "Diagnostic-error contract") catch this too, and
    so that a bare ``except np.linalg.LinAlgError`` around a batch of
    specifications keeps working.

    Raised in three situations, distinguishable by the message:

    * the fitted values reproduce the outcome to within ``1e-8`` for
      every observation (complete separation), or the linear predictor
      overflows;
    * the weighted cross-product ``X' W X`` goes singular *and* the IRLS
      weights have collapsed (``min(w) <= 1e-8 max(w)``);
    * Fisher scoring hits ``maxiter`` *and* the weights have collapsed.

    The last two are the two ways quasi-complete separation shows —
    a subset of observations is perfectly classified, and the
    corresponding coefficient diverges — and both are conjunctions
    rather than shortcuts, because each half occurs innocently on its
    own. ``X'WX`` can go singular from near-collinearity with the
    weights perfectly healthy, which needs a different remedy (drop or
    orthogonalise, not "the model is separated") and raises a plain
    diagnostic :class:`numpy.linalg.LinAlgError` saying so. And the
    weights alone collapse on any count model with a wide exposure —
    measured at ``min(w)/max(w) = 9.4e-09`` on a converging fit — so
    that half is only evidence alongside a loop that failed to settle.
    """


class ConvergenceWarning(UserWarning):
    """Warned when the Fisher-scoring loop hits ``maxiter``.

    The returned :class:`DiscreteResult` carries ``converged=False`` and
    the last iterate. Standard errors are conditional on a point that is
    not a stationary point of the log-likelihood and should not be
    reported.
    """


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
#
# Each family supplies the three things Fisher scoring needs (mean,
# working weight, log-likelihood) plus the two deviance-family
# diagnostics the corpus reads off the Poisson fit. Written as plain
# functions rather than classes because there are two of them and they
# are never user-extensible.


def _logit_mean_weight(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``p = expit(eta)`` and the Fisher weight ``p(1-p)``."""
    mu = special.expit(eta)
    return mu, mu * (1.0 - mu)


def _logit_loglike(y: np.ndarray, eta: np.ndarray, mu: np.ndarray) -> float:
    """``sum log F(q eta)`` with ``q = 2y - 1``, as ``Logit.loglike``.

    Written on the linear predictor rather than on ``mu`` because that is
    what ``discrete_model.py:2416-2417`` does, and it is the numerically
    stable form: for ``eta = -40`` the ``log(1 - mu)`` route loses every
    significant digit while ``log F(-eta)`` does not.
    """
    q = 2.0 * y - 1.0
    return float(np.sum(np.log(special.expit(q * eta))))


def _poisson_mean_weight(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``mu = exp(eta)``; for the log link the Fisher weight is ``mu``.

    Overflow is silenced here and diagnosed one level up: an infinite
    ``mu`` always means the coefficients are diverging, and
    :func:`_check_mu` says so in a sentence, which is more use than
    numpy's ``RuntimeWarning: overflow encountered in exp``.
    """
    with np.errstate(over="ignore"):
        mu = np.exp(eta)
    return mu, mu


def _poisson_loglike(y: np.ndarray, eta: np.ndarray, mu: np.ndarray) -> float:
    """``sum y log(mu) - mu - log(y!)``, as ``families.Poisson.loglike_obs``.

    ``special.xlogy`` rather than ``y * np.log(mu)`` so that a zero count
    contributes exactly zero instead of ``0 * -inf``; for ``mu > 0``,
    which the exponential link guarantees, the two agree bit for bit.
    """
    return float(np.sum(special.xlogy(y, mu) - mu - special.gammaln(y + 1.0)))


_FAMILY = {
    "Logit": (_logit_mean_weight, _logit_loglike),
    "Poisson": (_poisson_mean_weight, _poisson_loglike),
}


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """``2 sum [y log(y/mu) - (y - mu)]``.

    Matches ``families.Poisson._resid_dev``, which clips ``y/mu`` away
    from zero before taking the log so that ``y = 0`` contributes
    ``2 mu``; ``special.xlogy`` reaches the same value without the clip.
    """
    return float(2.0 * np.sum(special.xlogy(y, y / mu) - (y - mu)))


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _prepare(y, X, offset, exposure, missing, name):
    """Normalise ``(y, X, offset)`` to float arrays and recover column names.

    Returns ``(y, X, offset, names, keep)`` where ``names`` is a tuple of
    column labels when ``X`` carried them and ``None`` otherwise, and
    ``keep`` is the boolean row mask ``missing='drop'`` applied — or
    ``None`` when no row was dropped, which is what lets the caller
    re-align a ``cov_kwds['groups']`` vector that was built against the
    *undropped* sample. The ``names is None`` case is what makes the
    result object hand back bare ndarrays, exactly as statsmodels does
    for ndarray input.

    Non-finite data is rejected under ``missing='none'`` rather than
    propagated. That is not extra strictness invented here: statsmodels'
    ``missing='none'`` also raises on a non-finite design
    (``base/data.py::_handle_constant`` → ``MissingDataError('exog
    contains inf or nans')``, which runs whatever ``missing`` says),
    and a NaN response reaches its family code and dies there too. The
    difference is only that this raise arrives with a sentence saying
    which array holds the missing values and which ``missing`` setting
    would deal with them, instead of surfacing several hundred lines
    later as ``LinAlgError('SVD did not converge')`` out of a rank check
    or as a spurious separation diagnosis out of the IRLS loop. Both of
    those were the observed behaviour before this guard existed.
    """
    names = None
    if isinstance(X, pd.DataFrame):
        names = tuple(str(c) for c in X.columns)
        X_arr = X.to_numpy(dtype=float)
    elif isinstance(X, pd.Series):
        names = (str(X.name),) if X.name is not None else None
        X_arr = X.to_numpy(dtype=float).reshape(-1, 1)
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError(f"{name}: X must be 2-D, got shape {X_arr.shape}")

    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(
            f"{name}: y has {y_arr.shape[0]} observations but X has "
            f"{X_arr.shape[0]} rows"
        )

    n = X_arr.shape[0]
    off = np.zeros(n, dtype=float)
    # Length is checked before the addition, not after: numpy's own
    # broadcast error ("operands could not be broadcast together") does
    # not say which argument was wrong.
    for label, arr in (("offset", offset), ("exposure", exposure)):
        if arr is None:
            continue
        vec = np.asarray(arr, dtype=float).ravel()
        if vec.shape[0] != n:
            raise ValueError(
                f"{name}: offset/exposure length {vec.shape[0]} does not "
                f"match the {n} rows of X ({label})"
            )
        if label == "exposure":
            # `<= 0` rather than `not (> 0)` on purpose: a NaN exposure is
            # missing data, not a non-positive exposure, and belongs to the
            # `missing` machinery below so that missing='drop' can drop it.
            if np.any(vec <= 0.0):
                raise ValueError(f"{name}: exposure must be strictly positive")
            # statsmodels folds exposure into the offset as log(exposure)
            # (genmod/generalized_linear_model.py, GLM.__init__); passing
            # both is legal there and adds them, so it is legal here too.
            vec = np.log(vec)
        off = off + vec

    if missing not in ("none", "drop", "raise"):
        raise ValueError(
            f"{name}: missing must be 'none', 'drop' or 'raise'; "
            f"got {missing!r}"
        )
    # Computed for every `missing` setting, not just the two that act on
    # it: 'none' means "do not *drop*", not "do not *look*". Letting a NaN
    # through here does not propagate it to a NaN answer — nothing
    # downstream propagates. `np.linalg.matrix_rank` dies on it with a bare
    # LAPACK message, and an inf reaches the IRLS loop and is reported as a
    # diverging linear predictor, i.e. as separation. Both are wrong
    # diagnoses of a data problem the caller can fix in one line.
    keep = None
    bad = ~np.isfinite(y_arr) | ~np.isfinite(off)
    bad |= ~np.isfinite(X_arr).all(axis=1)
    if bad.any():
        n_bad = int(bad.sum())
        if missing == "raise":
            raise ValueError(
                f"{name}: {n_bad} of {n} rows contain NaN or inf "
                "(missing='raise')"
            )
        if missing == "none":
            where = ", ".join(
                part for part, flag in (
                    ("y", bool(np.any(~np.isfinite(y_arr)))),
                    ("X", bool(np.any(~np.isfinite(X_arr)))),
                    ("offset/exposure", bool(np.any(~np.isfinite(off)))),
                ) if flag
            )
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{name}: {n_bad} of {n} rows contain NaN or inf (in "
                f"{where}; first at position {first}). Maximum likelihood "
                "has no meaning on a missing observation, so missing='none' "
                "does not silently carry them into the fit. Pass "
                "missing='drop' to fit on the complete rows, or clean the "
                "input."
            )
        # Column labels are unaffected: `missing` drops rows, never
        # columns, so `names` carries through untouched.
        keep = ~bad
        y_arr, X_arr, off = y_arr[keep], X_arr[keep], off[keep]
    if y_arr.shape[0] == 0:
        raise ValueError(f"{name}: no observations left after missing handling")
    if y_arr.shape[0] <= X_arr.shape[1]:
        # Declared divergence (see the Notes of `logit` / `poisson`):
        # statsmodels fits n == k and returns the saturated, separated
        # answer. There is no information left for a standard error.
        raise ValueError(
            f"{name}: {y_arr.shape[0]} observations for {X_arr.shape[1]} "
            "parameters — the model is not identified"
        )
    return y_arr, X_arr, off, names, keep


def _align_groups(groups, keep, name):
    """Subset a cluster-label vector by the rows ``missing='drop'`` kept.

    ``cov_kwds['groups']`` is built by the caller against the frame they
    passed in, so after ``missing='drop'`` removes rows it is one entry
    per *original* observation while everything else is one entry per
    *retained* observation. statsmodels does not reconcile the two and
    dies inside its weighting code ("The weights and list don't have the
    same length"); this module lines them up instead.

    The mapping is unambiguous because the two lengths cannot collide:
    ``keep`` is only non-``None`` when at least one row was dropped, so
    ``keep.size`` (pre-drop) is strictly greater than ``keep.sum()``
    (post-drop). A vector matching neither length is left alone and
    reaches the length check in :func:`_covariance`, which names both
    numbers.
    """
    arr = groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
    if arr.ndim >= 1 and arr.shape[0] == keep.size:
        return arr[keep]
    return groups


def _wrap(values: np.ndarray, names):
    """A Series indexed by column name, or the bare array when unnamed."""
    if names is None:
        return values
    return pd.Series(values, index=list(names))


# ---------------------------------------------------------------------------
# The Fisher-scoring loop
# ---------------------------------------------------------------------------


def _check_mu(y, eta, mu, model, name, n_iter):
    """Reject a diverged or degenerate fit before it becomes a number."""
    if not np.all(np.isfinite(mu)):
        raise PerfectSeparationError(
            f"{name}: the linear predictor overflowed at iteration "
            f"{n_iter} (max |eta| = {np.max(np.abs(eta)):.3e}). The "
            "coefficients are diverging, which means the likelihood has "
            "no interior maximum — check for a regressor that separates "
            "the outcome, or rescale the design."
        )
    if model == "Logit" and np.max(np.abs(mu - y)) < _SEPARATION_ATOL:
        raise PerfectSeparationError(
            f"{name}: complete separation at iteration {n_iter} — the "
            "fitted probabilities reproduce every outcome to within "
            f"{_SEPARATION_ATOL:g}. The maximum-likelihood estimate does "
            "not exist (Albert-Anderson 1984); the coefficients diverge "
            "and their standard errors are meaningless. Drop the "
            "separating regressor, merge categories, or use a penalised "
            "(Firth) fit."
        )


def _fit_core(y, X, off, model, maxiter, tol, name, *, warn=True):
    """Iteratively reweighted least squares to the MLE of a canonical GLM.

    For a canonical link the observed and expected information coincide,
    so this single loop is simultaneously statsmodels' ``method='newton'``
    (``base/optimizer.py:385``, what ``sm.Logit.fit`` uses) and its IRLS
    (``genmod._fit_irls``, what ``sm.GLM.fit`` uses). The update is
    written in IRLS form,

        ``beta+ = (X'WX)^-1 X' W [ (lin - offset) + (y - mu)/g'(mu)^-1 ]``

    which for ``lin = X beta + offset`` collapses to the Newton step
    ``beta + (X'WX)^-1 X'(y - mu)`` — the working-response weight and the
    link derivative cancel for both canonical links. The IRLS form is
    kept because it is the only one that can express statsmodels'
    *starting point* for the Poisson.

    **The two entry points start in different places, and it matters.**
    ``sm.Logit.fit`` is Newton from ``beta = 0`` (``base/model.py:520``).
    ``sm.GLM.fit`` is IRLS from ``mu0 = (y + ybar)/2`` with
    ``lin_pred = log(mu0)`` and no offset added
    (``family.starting_mu``; ``_fit_irls``). Both are reproduced. Starting
    the Poisson from ``beta = 0`` instead means starting from an implied
    mean of 1, and on a count series with a spike in the thousands — a
    famine chronicle, say — the first Newton step is then long enough
    that the loop can fail to come back at all, while statsmodels
    converges. That failure was observed before this start was adopted.

    statsmodels' Newton adds ``ridge_factor=1e-10`` to the Hessian
    diagonal *during* the iteration (``optimizer.py:447-448``). That
    perturbs the path, never the fixed point — the loop still stops where
    the score is zero — and statsmodels itself drops the ridge when it
    computes the Hessian it reports. It is therefore not reproduced.

    Returns ``(beta, eta, mu, w, bread, converged, n_iter)``.
    """
    mean_weight, _ = _FAMILY[model]
    k = X.shape[1]

    # Design gate up front, so that a design problem is reported as a
    # design problem — naming the collinear columns — rather than being
    # mistaken for separation once the weights enter.
    #
    # The gate is `inv_xtx` itself, unconditionally, and NOT the cheaper
    # `matrix_rank(X) < k` test it used to be. Those two ask different
    # questions: `matrix_rank` is a *rank* test, while `inv_xtx` rejects on
    # *conditioning*, so a design that passed the rank test could still
    # fail inside the loop — where the only diagnosis on offer was
    # separation. Running the real gate here, on the same equilibrated
    # matrix the loop will use, means the two can no longer disagree.
    _equilibrated_bread(X, f"{name} design")  # raises, naming the columns

    beta = np.zeros(k)
    if model == "Poisson":
        mu = (y + float(np.mean(y))) / 2.0
        if np.any(mu <= 0.0):
            raise PerfectSeparationError(
                f"{name}: every count is zero, so the IRLS starting mean "
                "is zero and the log link is undefined. There is nothing "
                "to explain."
            )
        lin = np.log(mu)          # statsmodels adds no offset here
        w = mu
    else:
        lin = off.copy()
        mu, w = mean_weight(lin)

    converged = False
    n_iter = 0
    step_size = np.inf
    eta = lin

    for n_iter in range(1, maxiter + 1):
        _check_mu(y, eta, mu, model, name, n_iter)
        bread = _information_inverse(X, w, name, n_iter)
        # Working response, weighted. w*(y-mu)*g'(mu) is (y - mu) for both
        # canonical links, so only the first term needs the weights.
        rhs = X.T @ (w * (lin - off)) + X.T @ (y - mu)
        beta_new = bread @ rhs
        step_size = float(np.max(np.abs(beta_new - beta)))
        beta = beta_new
        lin = eta = X @ beta + off
        mu, w = mean_weight(eta)
        if step_size < tol:
            converged = True
            break

    # Re-evaluate everything at the returned coefficients. statsmodels
    # does the same for the reported Hessian (base/model.py:583 inverts
    # a Hessian recomputed at xopt), which is why the HAC and cluster
    # sandwiches below agree with it to machine precision even where the
    # non-robust one does not -- see poisson's Notes.
    _check_mu(y, eta, mu, model, name, n_iter)
    bread = _information_inverse(X, w, name, n_iter)

    # Separation that the information matrix no longer catches.
    #
    # Equilibrating the columns before inverting (see `_equilibrated_bread`)
    # is right for a badly *scaled* design, but it also rescues the matrix
    # in the one case where its collapse was the diagnosis: a regressor
    # non-zero only where the count is zero drives `w` to zero on exactly
    # those rows, the weighted column shrinks toward nothing, and unit-norm
    # scaling then stretches it back to a direction the Cholesky accepts.
    # The fit that comes out is the coefficient a hundred units down the
    # road to minus infinity, flagged only as non-convergence -- which is
    # the "silent garbage" reading of a model whose MLE does not exist.
    #
    # The test is a conjunction, and both halves are needed. Weight
    # collapse alone is not evidence: a Poisson with a genuine exposure
    # spanning eight orders of magnitude reaches min(w)/max(w) = 9.4e-09
    # and converges perfectly well. Non-convergence alone is not evidence
    # either; that is what `maxiter` is for. Together they are the
    # signature of a likelihood with no interior maximum, and nothing else
    # produces them: the coefficient marches by a fixed step per iteration
    # and the weights it kills stay killed.
    w_min, w_max = float(np.min(w)), float(np.max(w))
    if not converged and w_min <= _SEPARATION_ATOL * w_max:
        worst = int(np.argmax(np.abs(beta)))
        raise PerfectSeparationError(
            f"{name}: quasi-complete separation. Fisher scoring did not "
            f"converge in {maxiter} iterations (last max |step| = "
            f"{step_size:.3e}) and the IRLS weights have collapsed, "
            f"spanning {w_min:.3e} to {w_max:.3e}: some observations are "
            "perfectly predicted, so the coefficient explaining them has "
            f"no finite maximum-likelihood value. Column {worst} is the "
            f"furthest along, at {float(beta[worst]):.4g}. Drop the "
            "separating regressor, merge categories, or use a penalised "
            "(Firth) fit."
        )

    if not converged and warn:
        warnings.warn(
            f"{name}: Fisher scoring did not converge in {maxiter} "
            f"iterations (last max |step| = {step_size:.3e}, tol = {tol:g}). "
            "The returned coefficients are the last iterate and "
            "`converged` is False; every standard error is conditional on "
            "a point that is not a stationary point of the log-likelihood.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return beta, eta, mu, w, bread, converged, n_iter


def _equilibrated_bread(M, name):
    """``(M'M)^-1`` via ``inv_xtx``, with the columns scaled to unit norm.

    The equilibration is load-bearing, not cosmetic, and it is the same
    move ``regress.ols`` makes for the same reason. ``inv_xtx`` gates on
    the ratio of the smallest to the largest Cholesky pivot of ``M'M``,
    and those pivots carry the column *norms* — so an un-equilibrated
    design is refused for holding a regressor in levels (a population in
    persons beside an intercept) rather than for being collinear. On the
    logit design that made this visible, ``cond(X'X)`` is ``3.2e14``
    before equilibration and ``33`` after: the entire "ill-conditioning"
    was the unit of one column. Refusing that design is wrong twice over
    — statsmodels fits it, and the numbers are perfectly recoverable —
    and diagnosing it as separation, which is what happened before, is
    wrong a third time.

    Scaling by ``D = diag(||m_j||)`` gives ``Ms = M D^-1``, so
    ``(M'M)^-1 = D^-1 (Ms'Ms)^-1 D^-1`` exactly; the division below is
    that identity. What survives the gate afterwards is only genuine
    near-collinearity, which is scale-free and which rescaling cannot
    fix.

    Parameters
    ----------
    M : (n, k) ndarray
        Design, or ``sqrt(W) X`` for a weighted information matrix.
    name : str
        Caller label for the error message.

    Returns
    -------
    numpy.ndarray
        ``(M'M)^-1``, shape ``(k, k)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``M`` is rank deficient (the message names the columns most
        aligned with the null space) or near-collinear on the scale-free
        measure.
    """
    norms = np.sqrt(np.einsum("ij,ij->j", M, M))
    # An all-zero column would divide by zero; leave it alone and let the
    # rank branch of `inv_xtx` name it, which is the better message anyway.
    norms = np.where(norms > 0.0, norms, 1.0)
    Ms = M / norms
    try:
        bread = inv_xtx(Ms, name=name)
    except np.linalg.LinAlgError as err:
        # Which of inv_xtx's two refusals was it? Asked by re-deriving the
        # rank rather than by matching its message text, so that rewording
        # `_linalg.py` cannot silently disable this branch. Rank deficiency
        # is the case where inv_xtx has already named the culprit columns,
        # and nothing here improves on that.
        if np.linalg.matrix_rank(Ms) < M.shape[1]:
            raise
        # The other case is full-rank conditioning, where inv_xtx's advice
        # ("try rescaling regressors") is already spent: the columns it saw
        # were unit-norm.
        raise np.linalg.LinAlgError(
            f"{name}: X'X is numerically singular at full rank "
            f"{M.shape[1]} even with the columns scaled to unit norm "
            f"(cond ~ {np.linalg.cond(Ms.T @ Ms):.2e}). The regressors are "
            "near-collinear on a scale-free measure, so rescaling will not "
            "help: drop one, or orthogonalise them (centre a time trend "
            "before taking its powers)."
        ) from err
    return bread / np.outer(norms, norms)


def _information_inverse(X, w, name, n_iter):
    """``(X' W X)^-1``, with the failure diagnosed rather than assumed.

    :func:`_equilibrated_bread` wants a design, not a cross-product, so it
    is handed ``sqrt(W) X`` — whose cross-product is exactly ``X' W X``.

    A failure here has two possible causes and they call for opposite
    actions, so the message is chosen by evidence. ``_fit_core`` has
    already put ``X`` itself through the same gate, so what ``W`` adds on
    top is bounded by the spread of the weights,
    ``cond(X'WX) <= cond(X'X) * max(w)/min(w)``. Hence:

    * **weights collapsed** (``min(w) <= 1e-8 max(w)``) — for the logit
      that means a fitted probability within ~2.5e-9 of a boundary, for
      the Poisson a fitted mean eight orders below the largest. That is
      quasi-complete separation: some observations are perfectly
      predicted and the corresponding coefficient is running off.
    * **weights intact** — the spread cannot explain the singularity, so
      what failed is the design's own scale-free conditioning. Saying
      "separation" there would send the caller to drop a regressor on
      evidence that does not support it, so the underlying
      :class:`numpy.linalg.LinAlgError` is re-raised with the arithmetic
      that rules separation out.

    The threshold is ``_SEPARATION_ATOL`` rather than a second magic
    number, because it is the same statement the complete-separation
    check makes about ``mu``.
    """
    Xw = X * np.sqrt(w)[:, None]
    try:
        return _equilibrated_bread(Xw, f"{name} information matrix")
    except np.linalg.LinAlgError as err:
        w_min, w_max = float(np.min(w)), float(np.max(w))
        if w_min <= _SEPARATION_ATOL * w_max:
            raise PerfectSeparationError(
                f"{name}: the information matrix X'WX went singular at "
                f"iteration {n_iter} while X itself inverts, and the IRLS "
                f"weights have collapsed (they span {w_min:.3e} to "
                f"{w_max:.3e}). That is quasi-complete separation: some "
                "observations are perfectly predicted and the "
                "corresponding coefficient is diverging. Drop the "
                "separating regressor, merge categories, or use a "
                f"penalised (Firth) fit. Underlying diagnostic: {err}"
            ) from err
        raise np.linalg.LinAlgError(
            f"{name}: the information matrix X'WX is numerically singular "
            f"at iteration {n_iter}, and this is not separation — X itself "
            f"inverts and the IRLS weights are intact (they span "
            f"{w_min:.3e} to {w_max:.3e}, a factor of "
            f"{w_max / w_min:.3g}, far too little to account for it). "
            f"Underlying diagnostic: {err}"
        ) from err


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def _bartlett_meat(xu: np.ndarray, maxlags: int, kernel: str) -> np.ndarray:
    """Newey-West meat ``G_0 + sum_l w_l (G_l + G_l')`` of a score matrix.

    ``kernel='bartlett'`` gives ``w_l = 1 - l/(L+1)``; ``'uniform'`` gives
    ``w_l = 1``. Those are the two kernels ``sandwich_covariance.kernel_dict``
    exposes, and the two the ``cov_kwds['kernel']`` argument accepts.
    """
    S = xu.T @ xu
    n = xu.shape[0]
    for lag in range(1, maxlags + 1):
        if lag >= n:
            break
        w = 1.0 - lag / (maxlags + 1.0) if kernel == "bartlett" else 1.0
        G = xu[lag:].T @ xu[:-lag]
        S = S + w * (G + G.T)
    return S


def _normalise_cov_type(cov_type: str) -> str:
    """Map a user string onto the canonical covariance name."""
    if not isinstance(cov_type, str):
        raise TypeError(f"cov_type must be a string, got {type(cov_type)!r}")
    lowered = cov_type.lower()
    if lowered == "nonrobust":
        return "nonrobust"
    if cov_type.upper() in _HC_KINDS:
        return cov_type.upper()
    if lowered == "hac":
        return "HAC"
    if lowered == "cluster":
        return "cluster"
    raise ValueError(
        f"cov_type {cov_type!r} not recognised. Available for a "
        "maximum-likelihood fit: 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', "
        "'HAC', 'cluster'. The panel covariances ('hac-panel', "
        "'hac-groupsum') are implemented for OLS only, in "
        "puremacro.regress.ols."
    )


# The `cov_kwds` keys each covariance actually reads. Anything else is a
# typo or a statsmodels option this module does not implement, and either
# way silently ignoring it hands back a number the caller did not ask for
# -- see `_check_cov_kwds`.
_COV_KWDS_ALLOWED = {
    "HAC": ("kernel", "maxlags", "use_correction"),
    "cluster": ("df_correction", "groups", "use_correction"),
}


def _check_cov_kwds(cov_type, kwds, name):
    """Reject ``cov_kwds`` keys the requested covariance does not read.

    statsmodels does not do this, and says so: ``base/covtype.py``'s own
    docstring carries ``.. todo:: Currently there is no check for extra or
    misspelled keywords, except in the case of cov_type `HCx```. The
    consequence is that ``cov_kwds={'maxlag': 10}`` — one letter — is
    accepted and every option silently takes its default. In statsmodels
    the HAC case at least crashes, because ``maxlags`` is read with
    ``kwds['maxlags']`` and a missing key is a ``KeyError``; this module
    deliberately makes ``maxlags`` optional (the bandwidth falls back to
    ``floor(4 (n/100)^(2/9))``), which removes that accidental guard and
    would leave the typo returning a plausible, wrong standard error. The
    measured cost of not checking was a 4% error in ``bse`` with no
    warning. So the check is explicit, and it covers the cluster keys
    too, where statsmodels never crashed and the silent default has
    always been the whole story.
    """
    allowed = _COV_KWDS_ALLOWED[cov_type]
    unknown = sorted(set(kwds) - set(allowed))
    if unknown:
        raise ValueError(
            f"{name}: cov_kwds key(s) {unknown} not recognised for "
            f"cov_type={cov_type!r}. Accepted: {list(allowed)}. (An "
            "unrecognised key is not ignored here, because ignoring it "
            "returns a covariance built from defaults the caller did not "
            "choose.)"
        )


def _as_maxlags(value, name):
    """Coerce ``cov_kwds['maxlags']`` to a non-negative Python ``int``.

    A float is rejected rather than truncated. ``int(4.7)`` is 4, which is
    a bandwidth nobody asked for; statsmodels raises ``TypeError: 'float'
    object cannot be interpreted as an integer`` from the ``range`` in its
    kernel loop, so refusing it here is parity as well as good manners.
    ``bool`` passes because it is an ``int`` subclass and statsmodels
    accepts it for the same reason (``maxlags=True`` means one lag,
    there and here).
    """
    if not isinstance(value, (int, np.integer, np.bool_)):
        raise TypeError(
            f"{name}: cov_kwds['maxlags'] must be an integer; got "
            f"{type(value).__name__} ({value!r})"
        )
    maxlags = int(value)
    if maxlags < 0:
        raise ValueError(f"{name}: cov_kwds['maxlags'] must be >= 0")
    return maxlags


def _covariance(cov_type, kwds, xu, bread, n, k, df_resid, name):
    """Dispatch to the requested covariance.

    Returns ``(cov, n_groups, df_resid_inference, kwds_used)``.
    ``n_groups`` is ``None`` unless the estimator has a group dimension.
    """
    if cov_type == "nonrobust":
        if kwds:
            raise ValueError(
                f"{name}: cov_type='nonrobust' takes no cov_kwds; "
                f"got {sorted(kwds)}"
            )
        return bread, None, df_resid, {}

    if cov_type in _HC_KINDS:
        if kwds:
            # statsmodels raises 'heteroscedasticity robust covariance does
            # not use keywords' here (base/covtype.py:238-240).
            raise ValueError(
                f"{name}: heteroskedasticity-robust covariance does not use "
                f"keywords; got {sorted(kwds)}"
            )
        # All four HC names give this same matrix for a Logit/GLM result:
        # base/covtype.py:243 looks for a `cov_HC1`-style attribute, only
        # RegressionResults has one, and the fallback is
        # cov_white_simple(..., use_correction=False) for every kind.
        return bread @ (xu.T @ xu) @ bread, None, df_resid, {}

    if cov_type == "HAC":
        _check_cov_kwds("HAC", kwds, name)
        maxlags = kwds.get("maxlags", None)
        maxlags = (_default_maxlags(n) if maxlags is None
                   else _as_maxlags(maxlags, name))
        kernel = kwds.get("kernel", "bartlett")
        if kernel not in ("bartlett", "uniform"):
            raise ValueError(
                f"{name}: cov_kwds['kernel'] must be 'bartlett' or "
                f"'uniform'; got {kernel!r}"
            )
        cov = bread @ _bartlett_meat(xu, maxlags, kernel) @ bread
        use_correction = bool(kwds.get("use_correction", False))
        if use_correction:
            cov = cov * (n / float(n - k))
        return (cov, None, df_resid,
                {"maxlags": maxlags, "kernel": kernel,
                 "use_correction": use_correction})

    # cluster
    _check_cov_kwds("cluster", kwds, name)
    if "groups" not in kwds:
        raise ValueError(f"{name}: cov_type='cluster' requires cov_kwds['groups']")
    groups = kwds["groups"]
    if hasattr(groups, "to_numpy"):
        groups = groups.to_numpy()
    groups = np.asarray(groups)
    if groups.ndim == 2 and groups.shape[1] == 1:
        groups = groups[:, 0]
    if groups.ndim != 1:
        raise ValueError(
            f"{name}: cov_type='cluster' accepts a single group column here; "
            f"got shape {groups.shape}. Two-way clustering lives in "
            "puremacro.regress.ols."
        )
    if groups.shape[0] != n:
        raise ValueError(
            f"{name}: cov_kwds['groups'] has {groups.shape[0]} entries for "
            f"{n} observations"
        )
    n_unique = int(np.unique(groups).size)
    if n_unique < 2:
        # Not parity: statsmodels divides by G-1 = 0 and raises a bare
        # ZeroDivisionError from sandwich_covariance.py, and with
        # use_correction=False it skips the division and returns a rank-1
        # matrix built from the total score -- which at the MLE is zero, so
        # every standard error comes back at machine epsilon. A named
        # refusal is the package's contract for a degenerate design.
        raise ValueError(
            f"{name}: cov_type='cluster' needs at least two clusters; "
            f"cov_kwds['groups'] contains {n_unique}. With one cluster the "
            "meat matrix is the outer product of the total score, which "
            "vanishes at the maximum, and the small-sample factor "
            "G/(G-1) divides by zero. Use cov_type='HAC' for dependence "
            "within a single unit, or an HC covariance."
        )
    use_correction = bool(kwds.get("use_correction", True))
    cov, n_groups = _cluster_cov(xu, bread, groups, use_correction)
    df_inf = df_resid
    if kwds.get("df_correction", None) is not False:
        # base/covtype.py:364-366 — the t denominator becomes G-1.
        df_inf = float(n_groups - 1)
    return (cov, n_groups, df_inf,
            {"use_correction": use_correction, "n_groups": n_groups})


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscreteResult:
    """Fitted binary-choice or count regression.

    Returned by :func:`logit` and :func:`poisson`. The attribute names
    are statsmodels', so a call site that reads ``.params["x"]``,
    ``.bse["x"]``, ``.pvalues["x"]``, ``.nobs``, ``.conf_int()``,
    ``.pearson_chi2`` or ``.df_resid`` needs no other change.

    Attributes
    ----------
    model : str
        ``'Logit'`` or ``'Poisson'``.
    params : pandas.Series or numpy.ndarray
        Maximum-likelihood coefficients, shape ``(k,)``. A Series indexed
        by column name when ``X`` was a DataFrame, a bare ndarray
        otherwise — the same rule statsmodels follows.
    vcov : numpy.ndarray
        Parameter covariance, shape ``(k, k)``, of the requested
        ``cov_type``. Reached through :meth:`cov_params` in the
        statsmodels-shaped API.
    llf : float
        Log-likelihood at ``params``.
    llnull : float
        Log-likelihood of the intercept-only model fitted with the same
        offset. ``nan`` if that model itself failed to fit.
    nobs : int
        Number of observations used.
    df_model : float
        ``rank(X) - 1``.
    df_resid : float
        ``nobs - rank(X)``.
    df_resid_inference : float
        Denominator degrees of freedom for ``pvalues`` / :meth:`conf_int`
        when ``use_t`` is True. Equals ``df_resid`` except under
        ``cov_type='cluster'``, where statsmodels sets it to ``G - 1``.
    cov_type : str
        Canonical covariance name actually used.
    cov_kwds : dict
        The covariance options in force, including the defaults that were
        filled in (``maxlags``, ``use_correction``, ``n_groups``).
    use_t : bool
        Whether ``pvalues`` and :meth:`conf_int` use Student-t. Defaults
        to False for both estimators and every ``cov_type``.
    converged : bool
        Whether the Fisher-scoring loop met ``tol`` before ``maxiter``.
    n_iter : int
        Iterations actually taken.
    scale : float
        Dispersion. Fixed at 1.0 for both families, as in statsmodels.
    linpred : numpy.ndarray
        ``X @ params + offset``, shape ``(nobs,)``.
    mu : numpy.ndarray
        Fitted mean: ``P(y=1)`` for the logit, ``E[y]`` for the Poisson.
    weights : numpy.ndarray
        Final Fisher weights, ``p(1-p)`` or ``mu``.
    deviance : float
        Poisson deviance; ``nan`` for the logit (where the corpus never
        reads it and statsmodels' binary deviance has a different
        convention).
    null_deviance : float
        Deviance of the intercept-only model; ``nan`` for the logit.
    pearson_chi2 : float
        ``sum (y - mu)^2 / V(mu)``. ``pearson_chi2 / df_resid`` is the
        over-dispersion diagnostic ``N19:863`` registers.
    aic : float
        ``-2 llf + 2 k``, as both statsmodels result classes define it.
    bic : float
        **Different convention per model, because statsmodels' is.** For
        the logit, ``-2 llf + k log n`` (``DiscreteResults.bic``). For
        the Poisson, ``deviance - df_resid log n``
        (``GLMResults.bic``, the deviance form statsmodels still ships
        behind a ``FutureWarning``). Use :attr:`bic_llf` for the
        likelihood form in both cases.
    bic_llf : float
        ``-2 llf + k log n`` for both models.
    endog : numpy.ndarray
        The response actually used, after missing handling.
    exog : numpy.ndarray
        The design actually used, after missing handling.
    offset : numpy.ndarray
        The additive term in the linear predictor actually used: the
        ``offset`` argument plus ``log(exposure)``, and zeros when
        neither was given. The two are held jointly, as they enter the
        model — see :meth:`predict` for the one place that shows.
    names : tuple of str or None
        Column labels, or ``None`` for ndarray input.
    n_groups : int or None
        Number of clusters under ``cov_type='cluster'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.regress.discrete import logit
    >>> rng = np.random.default_rng(1)
    >>> X = pd.DataFrame({"const": 1.0, "x": rng.standard_normal(300)})
    >>> y = (rng.uniform(size=300) < 0.5).astype(float)
    >>> res = logit(y, X)
    >>> res.conf_int().columns.tolist()
    [0, 1]
    """

    model: str
    params: np.ndarray | pd.Series
    vcov: np.ndarray
    llf: float
    llnull: float
    nobs: int
    df_model: float
    df_resid: float
    df_resid_inference: float
    cov_type: str
    cov_kwds: dict = field(repr=False)
    use_t: bool
    converged: bool
    n_iter: int
    scale: float
    linpred: np.ndarray = field(repr=False)
    mu: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    deviance: float
    null_deviance: float
    pearson_chi2: float
    aic: float
    bic: float
    bic_llf: float
    endog: np.ndarray = field(repr=False)
    exog: np.ndarray = field(repr=False)
    offset: np.ndarray = field(repr=False)
    names: tuple[str, ...] | None
    n_groups: int | None

    # -- derived vectors ---------------------------------------------------

    @property
    def _params_arr(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)

    @property
    def bse(self):
        """Standard errors, ``sqrt(diag(cov_params()))``."""
        return _wrap(np.sqrt(np.diag(self.vcov)), self.names)

    @property
    def tvalues(self):
        """``params / bse``. Named ``tvalues`` even when the tail is normal."""
        return _wrap(self._params_arr / np.sqrt(np.diag(self.vcov)), self.names)

    @property
    def pvalues(self):
        """Two-sided p-values, normal-tail unless ``use_t``."""
        t = self._params_arr / np.sqrt(np.diag(self.vcov))
        if self.use_t:
            p = 2.0 * stats.t.sf(np.abs(t), self.df_resid_inference)
        else:
            p = 2.0 * stats.norm.sf(np.abs(t))
        return _wrap(p, self.names)

    @property
    def fittedvalues(self):
        """statsmodels' ``fittedvalues``, which means two different things.

        ``DiscreteResults.fittedvalues`` is the **linear predictor**
        ``X @ params`` (``discrete_model.py``), while
        ``GLMResults.fittedvalues`` is the **mean** ``mu``. That
        inconsistency is reproduced here so that a ported call site keeps
        reading the same quantity; :attr:`linpred` and :attr:`mu` are
        unambiguous and should be preferred in new code.
        """
        return self.linpred if self.model == "Logit" else self.mu

    @property
    def resid_response(self) -> np.ndarray:
        """``y - mu``."""
        return self.endog - self.mu

    @property
    def resid_pearson(self) -> np.ndarray:
        """``(y - mu) / sqrt(V(mu))``."""
        var = self.mu * (1.0 - self.mu) if self.model == "Logit" else self.mu
        return (self.endog - self.mu) / np.sqrt(var)

    @property
    def prsquared(self) -> float:
        """McFadden's ``1 - llf/llnull`` (``sm.Logit(...).prsquared``)."""
        return self.pseudo_rsquared("mcf")

    # -- methods -----------------------------------------------------------

    def pseudo_rsquared(self, kind: str = "mcf") -> float:
        """Pseudo R-squared.

        Parameters
        ----------
        kind : {'mcf', 'cs'}, default 'mcf'
            ``'mcf'`` is McFadden's ``1 - llf/llnull`` — what
            ``sm.Logit(...).prsquared`` reports. ``'cs'`` is Cox-Snell,
            ``1 - exp(2 (llnull - llf)/n)`` — the default of
            ``sm.GLM(...).pseudo_rsquared()``. Neither is the default of
            the other, which is why the argument is explicit here.

        Returns
        -------
        float
        """
        kind = kind.lower()
        if kind.startswith("mcf"):
            return float(1.0 - self.llf / self.llnull)
        if kind.startswith("cs") or kind.startswith("cox"):
            return float(1.0 - np.exp((self.llnull - self.llf) * (2.0 / self.nobs)))
        raise ValueError(f"pseudo_rsquared: kind must be 'mcf' or 'cs'; got {kind!r}")

    def cov_params(self):
        """Parameter covariance, as a DataFrame when the columns are named."""
        if self.names is None:
            return self.vcov
        return pd.DataFrame(self.vcov, index=list(self.names),
                            columns=list(self.names))

    def conf_int(self, alpha: float = 0.05):
        """Two-sided confidence interval at level ``1 - alpha``.

        Parameters
        ----------
        alpha : float, default 0.05
            Tail mass, split evenly.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            Columns ``0`` and ``1`` (lower, upper), indexed by column
            name when the design carried names — the shape statsmodels
            returns, so ``res.conf_int().loc["dry"]`` unpacks into a
            ``(lo, hi)`` pair exactly as it does at ``N19:854``.

        Notes
        -----
        The critical value is normal unless ``use_t`` is True, in which
        case it is Student-t on :attr:`df_resid_inference`.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"conf_int: alpha must be in (0, 1); got {alpha!r}")
        se = np.sqrt(np.diag(self.vcov))
        if self.use_t:
            crit = stats.t.ppf(1.0 - alpha / 2.0, self.df_resid_inference)
        else:
            crit = stats.norm.ppf(1.0 - alpha / 2.0)
        lo = self._params_arr - crit * se
        hi = self._params_arr + crit * se
        if self.names is None:
            return np.column_stack([lo, hi])
        return pd.DataFrame({0: lo, 1: hi}, index=list(self.names))

    def predict(self, exog=None, offset=None, exposure=None,
                which: str = "mean") -> np.ndarray:
        """Predicted mean (or linear predictor) at ``exog``.

        Parameters
        ----------
        exog : array_like, optional
            Design to predict at. Defaults to the estimation design, in
            which case the estimation offset is carried over.
        offset, exposure : array_like, optional
            Offset in the linear predictor; ``exposure`` enters as
            ``log(exposure)``. An ``offset`` given here *replaces* the
            estimation offset; an ``exposure`` given here is *added* to
            it. See Notes.
        which : {'mean', 'linear'}, default 'mean'
            ``'mean'`` returns ``P(y=1)`` or ``E[y]``; ``'linear'``
            returns ``X beta + offset``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``. An ndarray, not a Series, as statsmodels
            returns for these two models.

        Raises
        ------
        ValueError
            If ``offset`` or ``exposure`` has the wrong length, if
            ``exposure`` is not strictly positive, or if ``which`` is
            neither ``'mean'`` nor ``'linear'``.

        Notes
        -----
        The offset rule is ``GLMResults.predict``'s, verbatim
        (``genmod/generalized_linear_model.py:989-1007``): the estimation
        offset survives unless *it* is overridden, and passing an
        ``exposure`` does not override it. Reading that rule as "any of
        the three arguments discards the fitted offset" costs a factor of
        ``exp(offset)`` on every prediction — measured at ``4.8`` on a
        series with mean ``4.06`` — with nothing to show that it
        happened. A new ``exog``, on the other hand, *does* discard both,
        because the estimation offset is one number per estimation row
        and there is no reason to think it lines up with a new design.

        One divergence remains, and it is structural rather than
        arithmetic. statsmodels keeps ``model.offset`` and
        ``model.exposure`` as separate vectors, so ``predict(offset=o)``
        replaces the first and keeps the second. :func:`poisson` folds
        ``log(exposure)`` into :attr:`offset` when the model is built,
        the way the linear predictor does, so on a model fitted with
        ``exposure=`` a later ``predict(offset=...)`` drops the fitted
        exposure with it and ``predict(exposure=...)`` adds to it rather
        than replacing it. Fit with ``offset=np.log(e)`` instead of
        ``exposure=e`` and every path agrees with statsmodels exactly.
        """
        if exog is None:
            X = self.exog
            # statsmodels: `if offset is None and exog is None: offset =
            # self.offset`. Only an explicit `offset` displaces the fitted
            # one -- an `exposure` is a separate additive term.
            base = self.offset if offset is None else None
        else:
            X = np.asarray(
                exog.to_numpy() if hasattr(exog, "to_numpy") else exog, dtype=float
            )
            if X.ndim == 1:
                X = X[:, None]
            base = None
        rows = X.shape[0]
        off = np.zeros(rows) if base is None else base
        for label, arr in (("offset", offset), ("exposure", exposure)):
            if arr is None:
                continue
            vec = np.asarray(arr, dtype=float).ravel()
            if vec.shape[0] != rows:
                raise ValueError(
                    f"predict: {label} has {vec.shape[0]} entries for "
                    f"{rows} rows of exog"
                )
            if label == "exposure":
                if np.any(vec <= 0.0):
                    raise ValueError("predict: exposure must be strictly positive")
                vec = np.log(vec)
            off = off + vec
        eta = X @ self._params_arr + off
        if which == "linear":
            return eta
        if which != "mean":
            raise ValueError(f"predict: which must be 'mean' or 'linear'; got {which!r}")
        return _FAMILY[self.model][0](eta)[0]

    def to_frame(self) -> pd.DataFrame:
        """Coefficient table as a tidy DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``term, coef, std_err, z, p_value, ci_lo, ci_hi``.
        """
        ci = np.asarray(self.conf_int())
        terms = list(self.names) if self.names is not None else [
            f"x{i}" for i in range(len(self._params_arr))
        ]
        return pd.DataFrame({
            "term": terms,
            "coef": self._params_arr,
            "std_err": np.sqrt(np.diag(self.vcov)),
            "z": np.asarray(self.tvalues, dtype=float),
            "p_value": np.asarray(self.pvalues, dtype=float),
            "ci_lo": ci[:, 0],
            "ci_hi": ci[:, 1],
        })

    def summary(self) -> str:
        """Plain-text regression table.

        Returns
        -------
        str
            A header block plus one row per coefficient. This is
            *readable like* statsmodels' ``summary()`` but is not
            byte-identical to it — the corpus prints it
            (``N04:1292``) and never parses it.
        """
        tab = self.to_frame()
        head = "z" if not self.use_t else "t"
        width = 86
        lines = [
            f"{self.model} regression results",
            "=" * width,
            f"No. observations : {self.nobs:<12d} Df model    : {self.df_model:.0f}",
            f"Log-likelihood   : {self.llf:<12.4f} Df residuals: {self.df_resid:.0f}",
            f"Pseudo R-sq (McF): {self.prsquared:<12.4f} AIC         : {self.aic:.4f}",
            f"Covariance       : {self.cov_type:<12s} use_t       : {self.use_t}",
            f"Converged        : {str(self.converged):<12s} Iterations  : {self.n_iter}",
        ]
        if self.model == "Poisson":
            lines.append(
                f"Pearson chi2     : {self.pearson_chi2:<12.4f} "
                f"chi2/df     : {self.pearson_chi2 / self.df_resid:.4f}"
            )
        lines.append("-" * width)
        lines.append(
            f"{'term':<20s}{'coef':>12s}{'std err':>12s}{head:>10s}"
            f"{'P>|' + head + '|':>10s}{'[0.025':>11s}{'0.975]':>11s}"
        )
        lines.append("-" * width)
        for _, r in tab.iterrows():
            lines.append(
                f"{str(r['term']):<20s}{r['coef']:>12.4f}{r['std_err']:>12.4f}"
                f"{r['z']:>10.3f}{r['p_value']:>10.3f}"
                f"{r['ci_lo']:>11.4f}{r['ci_hi']:>11.4f}"
            )
        lines.append("=" * width)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The two public estimators
# ---------------------------------------------------------------------------


def _fit(model, y, X, *, offset, exposure, cov_type, cov_kwds, use_t,
         missing, maxiter, tol):
    """Shared body of :func:`logit` and :func:`poisson`."""
    name = model.lower()
    y_arr, X_arr, off, names, keep = _prepare(
        y, X, offset, exposure, missing, name
    )
    n, k = X_arr.shape

    if model == "Logit":
        # Written as a negated `all`, not as `any(y < 0) or any(y > 1)`.
        # Every comparison against NaN is False, so the `any` form has a
        # hole exactly where a missing value sits; statsmodels writes it
        # this way for the same reason (`discrete_model.py`, "endog must be
        # in the unit interval"). `_prepare` now rejects non-finite input
        # before this point, so the form is belt and braces -- which is
        # what it should be, since the check reads as a range check and
        # will be maintained as one.
        if not np.all((y_arr >= 0.0) & (y_arr <= 1.0)):
            raise ValueError(
                f"{name}: endog must be in the unit interval; found values in "
                f"[{y_arr.min():.4g}, {y_arr.max():.4g}]"
            )
        if not np.any(y_arr > 0.0) or not np.any(y_arr < 1.0):
            raise PerfectSeparationError(
                f"{name}: the outcome is constant "
                f"({y_arr[0]:.0f} for all {n} observations); there is nothing "
                "to explain and no maximum-likelihood estimate exists."
            )
    else:
        # Negated `all` for the same NaN reason as the logit branch above.
        if not np.all(y_arr >= 0.0):
            raise ValueError(
                f"{name}: the response must be non-negative counts; found "
                f"a minimum of {y_arr.min():.4g}"
            )

    beta, eta, mu, w, bread, converged, n_iter = _fit_core(
        y_arr, X_arr, off, model, maxiter, tol, name
    )

    rank = int(np.linalg.matrix_rank(X_arr))
    df_model = float(rank - 1)
    df_resid = float(n - rank)

    cov_type_c = _normalise_cov_type(cov_type)
    kwds = dict(cov_kwds) if cov_kwds else {}
    if keep is not None and cov_type_c == "cluster" and "groups" in kwds:
        # missing='drop' shortened y/X/offset; the cluster labels were
        # built against the frame the caller passed in, so line them up.
        kwds["groups"] = _align_groups(kwds["groups"], keep, name)
    # xu = score_obs. For a canonical link the score factor is (y - mu)
    # for both families (sandwich_covariance.py:241 reads it off
    # model.score_obs), so a single expression serves both.
    xu = X_arr * (y_arr - mu)[:, None]
    vcov, n_groups, df_inf, kwds_used = _covariance(
        cov_type_c, kwds, xu, bread, n, k, df_resid, name
    )

    loglike = _FAMILY[model][1]
    llf = loglike(y_arr, eta, mu)

    # Intercept-only model, on the same offset, for llnull / null_deviance.
    # Wrapped because an intercept-only fit can itself be degenerate (a
    # Poisson whose counts are all zero, say); the diagnostic then belongs
    # to a quantity nobody asked for, so it becomes nan rather than a
    # raise that kills a well-posed main fit.
    try:
        ones = np.ones((n, 1))
        b0, eta0, mu0, _, _, _, _ = _fit_core(
            y_arr, ones, off, model, maxiter, tol, f"{name} null model",
            warn=False,
        )
        llnull = loglike(y_arr, eta0, mu0)
        null_dev = (_poisson_deviance(y_arr, mu0) if model == "Poisson"
                    else float("nan"))
    except (np.linalg.LinAlgError, ValueError):
        llnull = float("nan")
        null_dev = float("nan")

    if model == "Poisson":
        deviance = _poisson_deviance(y_arr, mu)
        pearson = float(np.sum((y_arr - mu) ** 2 / mu))
        # GLMResults.bic is still the deviance form in 0.14.6.
        bic = deviance - df_resid * np.log(n)
    else:
        deviance = float("nan")
        pearson = float(np.sum((y_arr - mu) ** 2 / (mu * (1.0 - mu))))
        bic = -2.0 * llf + np.log(n) * rank

    return DiscreteResult(
        model=model,
        params=_wrap(beta, names),
        vcov=vcov,
        llf=llf,
        llnull=llnull,
        nobs=int(n),
        df_model=df_model,
        df_resid=df_resid,
        df_resid_inference=float(df_inf),
        cov_type=cov_type_c,
        cov_kwds=kwds_used,
        use_t=bool(use_t) if use_t is not None else False,
        converged=converged,
        n_iter=n_iter,
        scale=1.0,
        linpred=eta,
        mu=mu,
        weights=w,
        deviance=deviance,
        null_deviance=null_dev,
        pearson_chi2=pearson,
        aic=-2.0 * llf + 2.0 * rank,
        bic=float(bic),
        bic_llf=-2.0 * llf + np.log(n) * rank,
        endog=y_arr,
        exog=X_arr,
        offset=off,
        names=names,
        n_groups=n_groups,
    )


def logit(y, X, *, cov_type: str = "nonrobust", cov_kwds: dict | None = None,
          use_t: bool | None = None, missing: str = "none",
          maxiter: int = 100, tol: float = 1e-10) -> DiscreteResult:
    """Binary logistic regression by maximum likelihood.

    The replacement for ``sm.Logit(y, X).fit(disp=False)``. Newton-
    Raphson from ``beta = 0``; the default covariance is the inverse
    observed information ``(X' diag(p(1-p)) X)^-1``; ``use_t`` is False,
    so p-values and confidence intervals are normal-tail.

    Parameters
    ----------
    y : array_like or pandas.Series
        Binary outcome, shape ``(n,)``. Values in ``[0, 1]`` are
        accepted (statsmodels allows a fractional response for this
        model); anything outside raises.
    X : array_like or pandas.DataFrame
        Design matrix, shape ``(n, k)``. **No constant is added** — pass
        one yourself, e.g. with
        :func:`puremacro.regress.ols.add_constant`, exactly as
        ``sm.Logit`` requires. When ``X`` is a DataFrame, ``params`` /
        ``bse`` / ``pvalues`` come back as Series indexed by its columns.
    cov_type : str, default 'nonrobust'
        One of ``'nonrobust'``, ``'HC0'``, ``'HC1'``, ``'HC2'``,
        ``'HC3'``, ``'HAC'``, ``'cluster'``. The four HC names are
        aliases of one another here because statsmodels makes them
        aliases for a maximum-likelihood fit (see Notes).
    cov_kwds : dict, optional
        ``{'maxlags': L, 'kernel': 'bartlett'|'uniform',
        'use_correction': bool}`` for ``'HAC'``;
        ``{'groups': g, 'use_correction': bool, 'df_correction': bool}``
        for ``'cluster'``. **Those lists are exhaustive and are
        enforced**: any other key raises rather than being ignored, and
        ``maxlags`` must be an integer. statsmodels ignores both, which
        turns a one-letter typo into a silently different standard error
        (see the module's "Declared divergences").
    use_t : bool, optional
        Use Student-t rather than normal tails. ``None`` means False,
        which is statsmodels' default for this model under every
        ``cov_type``.
    missing : {'none', 'drop', 'raise'}, default 'none'
        What to do with rows holding a NaN or an inf. ``'none'`` does not
        *drop* them — and does not carry them into the fit either: it
        raises, naming the array and the first offending row, because
        every downstream failure mode (a bare LAPACK error out of the
        rank check, a "diverging linear predictor") is a misdiagnosis of
        what is simply missing data. statsmodels also raises under
        ``'none'``, as ``MissingDataError``, for a non-finite design.
        ``'drop'`` fits on the complete rows; under ``cov_type='cluster'``
        it subsets ``cov_kwds['groups']`` to match, so the two options
        compose.
    maxiter : int, default 100
        Iteration cap. statsmodels' default is 35; 100 is used here
        because the tolerance is tighter.
    tol : float, default 1e-10
        Stop when the largest absolute coefficient step falls below this.
        statsmodels stops at ``1e-8`` on the same criterion
        (``base/optimizer.py:435``); the tighter default costs at most
        one extra Newton step and removes the last iterate's dependence
        on where the loop happened to stop.

    Returns
    -------
    DiscreteResult
        See that class for the full attribute list.

    Raises
    ------
    PerfectSeparationError
        If the design separates the outcome — complete separation
        (fitted values reproduce ``y`` to ``1e-8``), quasi-complete
        separation (``X'WX`` singular, or Fisher scoring stalling at
        ``maxiter``, *with* the IRLS weights collapsed), a diverging
        linear predictor, or a constant outcome. statsmodels warns and
        returns the diverging iterate instead; that is the behaviour this
        raise exists to prevent.
    numpy.linalg.LinAlgError
        If ``X`` is rank deficient — the message names the columns most
        aligned with the null space (:func:`puremacro._linalg.inv_xtx`) —
        or if it is near-collinear once the columns are scaled to unit
        norm, which is a property rescaling cannot fix and the message
        says so. A design that merely *looks* ill-conditioned because a
        column is in awkward units is fitted; see Notes.
    ValueError
        If ``y`` leaves ``[0, 1]``, if any input is non-finite under
        ``missing='none'``, if there are no more observations than
        parameters, if shapes disagree, if ``cov_type`` is unknown, or if
        ``cov_kwds`` holds a key the chosen covariance does not read.
    TypeError
        If ``cov_kwds['maxlags']`` is not an integer.

    Warns
    -----
    ConvergenceWarning
        If ``maxiter`` is reached. The result carries
        ``converged=False``.

    Notes
    -----
    **Parity.** Against ``sm.Logit(y, X).fit(disp=False)`` on a seeded,
    separation-free design (n=300, k=3) the agreement measured in
    ``tests/test_regress_discrete_parity.py`` is at the level of floating
    point: ``params`` to 3.3e-16, ``bse`` to 2.8e-17, ``pvalues`` to
    5.2e-17, and ``llf`` exactly equal. Newton converges quadratically,
    so both loops land on the same stationary point regardless of where
    they stop.

    **``llnull`` is exact here and approximate there.**
    ``DiscreteResults.llnull`` refits the intercept-only model with BFGS,
    and lands about ``1.4e-8`` short of its maximum on the tested design.
    The intercept-only logit has the closed form
    ``n1 log(pbar) + n0 log(1-pbar)``, which the Newton loop here reaches
    bit-exactly. ``prsquared`` inherits an attenuated version of the
    difference (~1e-11) and still agrees to ``1e-8``. The direction is
    worth noting: this is a case where matching statsmodels would mean
    reproducing its optimiser's residual error.

    **HC aliasing is statsmodels', not ours.** For a
    ``LikelihoodModelResults`` there is no ``cov_HC1`` attribute to find,
    so ``base/covtype.py:243-249`` sends ``'HC0'``, ``'HC1'``, ``'HC2'``
    and ``'HC3'`` alike to ``cov_white_simple(..., use_correction=False)``.
    There is no ``n/(n-k)`` factor and no leverage adjustment. This
    module reproduces that; do not read ``cov_type='HC1'`` here as
    meaning what it means for :func:`puremacro.regress.ols.ols`.

    **Where this refuses and statsmodels answers.** Several guards are
    stricter than statsmodels on purpose; the module docstring's
    "Declared divergences" table is the full list, and the one that most
    often surprises is here. ``n == k`` raises rather than fitting: the
    saturated logit is separated, and the coefficients statsmodels
    returns for it (order 1e2, with standard errors to match) are an
    artefact of where its optimiser stopped.

    **Scaling.** ``cond(X)`` is not one of those guards. Every inversion
    scales the columns to unit norm first, so a regressor in persons
    rather than millions changes nothing: the fit is the same to ``1e-10``
    either way, and it is statsmodels' fit. What still raises is
    near-collinearity, which survives that scaling because it is a
    property of the column *directions*.

    **What is not implemented.** ``method`` (only Newton/IRLS),
    ``start_params``, ``fit_regularized``, marginal effects
    (``get_margeff``), ``score_test``, and ``cov_type`` in
    ``{'hac-panel', 'hac-groupsum'}``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.regress.discrete import logit
    >>> rng = np.random.default_rng(4)
    >>> x = rng.standard_normal(500)
    >>> X = pd.DataFrame({"const": 1.0, "x": x})
    >>> y = (rng.uniform(size=500) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(float)
    >>> res = logit(y, X)
    >>> 0.9 < float(res.params["x"]) < 1.6      # true slope is 1.2
    True
    >>> res.nobs, bool(res.converged), res.use_t
    (500, True, False)
    """
    return _fit("Logit", y, X, offset=None, exposure=None, cov_type=cov_type,
                cov_kwds=cov_kwds, use_t=use_t, missing=missing,
                maxiter=maxiter, tol=tol)


def poisson(y, X, *, offset=None, exposure=None, cov_type: str = "nonrobust",
            cov_kwds: dict | None = None, use_t: bool | None = None,
            missing: str = "none", maxiter: int = 100,
            tol: float = 1e-10) -> DiscreteResult:
    """Poisson count regression with a log link, by IRLS.

    The replacement for
    ``sm.GLM(y, X, family=sm.families.Poisson()).fit(...)``. The single
    corpus call site is ``N19:840``, which asks for
    ``cov_type='HAC'``; that path is reproduced exactly.

    Parameters
    ----------
    y : array_like or pandas.Series
        Non-negative counts, shape ``(n,)``. Non-integer values are
        allowed, as in a quasi-Poisson fit; negative values raise.
    X : array_like or pandas.DataFrame
        Design matrix, shape ``(n, k)``. **No constant is added.**
    offset : array_like, optional
        Added to the linear predictor with coefficient 1.
    exposure : array_like, optional
        Strictly positive exposure; enters as ``log(exposure)``. Passing
        both ``offset`` and ``exposure`` adds them, as statsmodels does.
    cov_type : str, default 'nonrobust'
        As :func:`logit`.
    cov_kwds : dict, optional
        As :func:`logit`, including the rejection of unrecognised keys.
        ``cov_type='HAC'`` with no ``maxlags`` falls back to
        ``floor(4 (n/100)^(2/9))`` — the same rule
        :func:`puremacro.regress.ols.ols` uses. This is a deliberate
        divergence: statsmodels indexes ``cov_kwds['maxlags']``
        unconditionally (``base/covtype.py:251``) and raises a bare
        ``KeyError`` instead, so the ``cov_hac_simple`` default its own
        comment promises is unreachable. Note what that costs, and why
        the key check next to it is not optional: statsmodels' ``KeyError``
        was also the only thing standing between ``{'maxlag': 10}`` and a
        covariance built at the fallback bandwidth.
    use_t : bool, optional
        ``None`` means False, statsmodels' default for a GLM.
    missing : {'none', 'drop', 'raise'}, default 'none'
        As :func:`logit`: ``'none'`` raises on non-finite input rather
        than carrying it into the fit, and ``'drop'`` composes with
        ``cov_type='cluster'``.
    maxiter : int, default 100
        Same cap statsmodels' IRLS uses.
    tol : float, default 1e-10
        Stop when the largest absolute coefficient step falls below this.
        statsmodels stops on a *deviance* change below ``1e-8``, which is
        a looser criterion — see Notes.

    Returns
    -------
    DiscreteResult
        Carries ``pearson_chi2``, ``deviance`` and ``df_resid``, so the
        over-dispersion diagnostic ``pearson_chi2 / df_resid`` that
        ``N19:863`` registers reads across unchanged.

    Raises
    ------
    PerfectSeparationError
        If the linear predictor diverges, or the IRLS weights collapse
        while either ``X'WX`` goes singular or Fisher scoring fails to
        converge (separation in the Albert-Anderson sense: a regressor
        that is non-zero only where the count is zero drives its
        coefficient to minus infinity — this is the path a count model
        usually takes, the linear one being the logit's).
    numpy.linalg.LinAlgError
        If ``X`` is rank deficient — the message names the culprit
        columns — or is too ill-conditioned to invert, in which case it
        says to rescale. As :func:`logit`.
    ValueError
        If ``y`` has a negative entry, if ``exposure`` is not strictly
        positive, if any input is non-finite under ``missing='none'``, if
        there are no more observations than parameters, if shapes
        disagree, or if ``cov_kwds`` holds an unknown key or a bad value.
    TypeError
        If ``cov_kwds['maxlags']`` is not an integer.

    Warns
    -----
    ConvergenceWarning
        If ``maxiter`` is reached.

    Notes
    -----
    **Parity — and the one place it is not exact.** ``params``, ``llf``,
    ``deviance``, ``pearson_chi2``, ``df_resid``, and the ``HAC``,
    ``HC0`` and ``cluster`` covariances all reproduce
    ``sm.GLM(y, X, family=sm.families.Poisson()).fit(...)`` to machine
    precision: ``params`` to 4.4e-16, ``llf`` / ``deviance`` /
    ``pearson_chi2`` to 6e-14, and the ``HAC`` ``bse`` to 2.1e-17 with an
    offset and 2.0e-12 without — the second figure being larger only
    because statsmodels' own default IRLS tolerance leaves its
    coefficients ~1.5e-11 from the fixed point on that design, which the
    sandwich then inherits. The **non-robust** ``bse`` does not survive,
    and the reason is worth knowing:

    statsmodels' IRLS stops when the *deviance* stops moving
    (``tol=1e-8``), and the covariance it reports is the one produced by
    the final weighted least-squares step — whose weights were built from
    the mean of the *previous* iterate. Its ``bse`` is therefore
    evaluated one Newton step behind its own coefficients. This module
    evaluates the information matrix at the converged coefficients, which
    is the quantity the standard error is defined to be. On seeded
    designs (n=200, k=3) the gap is **zero to ~1e-12 when statsmodels
    takes five IRLS iterations and up to 1.9e-7 when its deviance
    criterion stops it at four** — i.e. it is a property of where
    statsmodels' loop happened to stop, not a systematic difference.
    Against ``sm.GLM(...).fit(tol=1e-14)``, which forces statsmodels to
    the same fixed point, the agreement is ~6e-17. ``N19`` reports its
    HAC standard error to three decimals and never reads the non-robust
    one, so no registered claim moves; the test file pins both numbers so
    the gap cannot grow unnoticed.

    **bic.** ``GLMResults.bic`` is still the *deviance* form
    ``deviance - df_resid log n`` in statsmodels 0.14.6 (behind a
    ``FutureWarning``), and :attr:`DiscreteResult.bic` reproduces it for
    this estimator. :attr:`DiscreteResult.bic_llf` is the likelihood
    form.

    **offset, exposure and prediction.** Both enter the linear predictor
    additively and are folded together at fit time, ``exposure`` as
    ``log(exposure)``, exactly as ``GLM.__init__`` folds them. The one
    consequence worth knowing is in :meth:`DiscreteResult.predict`, whose
    Notes give the rule and the single case where holding them jointly
    rather than separately shows: fit with ``offset=np.log(e)`` rather
    than ``exposure=e`` if a later ``predict(offset=...)`` on the same
    result has to match statsmodels exactly.

    **Where this refuses and statsmodels answers.** As :func:`logit` —
    see the module docstring's "Declared divergences" table.

    **What is not implemented.** Other families and links, ``var_weights``
    / ``freq_weights``, ``fit_constrained``, ``estimate_scale`` other
    than the fixed 1.0, and the panel covariances.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.regress.discrete import poisson
    >>> rng = np.random.default_rng(11)
    >>> x = rng.standard_normal(400)
    >>> X = pd.DataFrame({"const": 1.0, "x": x})
    >>> y = rng.poisson(np.exp(0.7 + 0.4 * x)).astype(float)
    >>> res = poisson(y, X, cov_type="HAC", cov_kwds={"maxlags": 4})
    >>> res.cov_kwds["use_correction"]        # HAC default, not True
    False
    >>> round(float(res.params["x"]), 1)
    0.4
    """
    return _fit("Poisson", y, X, offset=offset, exposure=exposure,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                missing=missing, maxiter=maxiter, tol=tol)
