"""Cross-section spatial regression: SAR, SEM, SDM, SLX and the LM battery.

Four specifications, all on a single :class:`~puremacro.spatial.SpatialWeights`
matrix ``W``:

======  ===============================================  ==============================
model   equation                                         estimator
======  ===============================================  ==============================
SLX     ``y = X beta + W X_d theta + eps``               OLS
SAR     ``y = rho W y + X beta + eps``                   concentrated ML / KP GMM
SEM     ``y = X beta + u``, ``u = lam W u + eps``        concentrated ML / KP GMM
SDM     ``y = rho W y + X beta + W X_d theta + eps``     concentrated ML / KP GMM
======  ===============================================  ==============================

The default estimator is Gaussian maximum likelihood through the *concentrated*
log-likelihood of Ord (1975) and Anselin (1988, ch. 6): ``beta`` and ``sigma^2``
are profiled out, leaving a smooth one-dimensional problem in the spatial
parameter,

    ``ln L_c(rho) = -(n/2)(ln 2pi + 1) + ln|I - rho W| - (n/2) ln(SSR(rho)/n)``,

maximised by a bounded scalar optimisation over the admissible interval and
cross-checked against the root of the analytic score. For the lag models the
whole profile costs three scalars: with ``e0 = y - X b0`` the OLS residual and
``eL = Wy - X bL`` the residual of ``Wy`` on ``X``, the ML residual at any
``rho`` is ``e0 - rho eL``, so ``SSR(rho) = c0 - 2 rho c1 + rho^2 c2`` and each
evaluation is one quadratic plus one log-determinant. For the error model the
six Gram matrices of ``[X, WX, y, Wy]`` are formed once and every ``lam`` costs
one ``k x k`` solve.

The Jacobian ``ln|I - rho W|`` is **exact**: dense eigenvalues for ``n <= 1000``
(``logdet='eig'``) and a sparse LU factorisation above (``logdet='lu'``); the
two agree to ~1e-12. A Chebyshev approximation (Pace & LeSage 2004) is
available on explicit request for very large symmetrisable ``W``. There is no
stochastic log-determinant, no Hutchinson probe and no seeded numeric anywhere
in the estimators: every trace and determinant in this module is computed
exactly, and where that forces a dense ``O(n^2)`` object the code emits a
``RuntimeWarning`` naming the function and the size and proceeds.

The alternative estimator is Kelejian-Prucha GMM: spatial two-stage least
squares for the lag models with ``[X, WX, ..., W^q X]`` as instruments
(Kelejian & Prucha 1998), and the three-moment estimator of Kelejian & Prucha
(1999) followed by spatially filtered GLS for the error model. GMM admits
heteroskedasticity-robust standard errors, which ML does not: under
heteroskedasticity of unknown form the Gaussian SAR ML estimator of ``rho`` is
itself *inconsistent* (Lin & Lee 2010), so ``vcov='robust'`` with
``method='ml'`` raises rather than wrapping a sandwich around the wrong number.

Because a spatial-lag coefficient is not a marginal effect,
:func:`spatial_effects` implements the LeSage-Pace (2009, sec. 2.7)
decomposition ``S_r(W) = (I - rho W)^{-1} (I beta_r + W theta_r)`` into average
direct, indirect and total impacts. It reduces exactly to four scalars,

    ``tau = tr(A^-1)``, ``t_W = tr(A^-1 W)``, ``s = 1'A^-1 1``, ``s_W = 1'A^-1 W 1``

with ``direct_r = (beta_r tau + theta_r t_W)/n`` and
``total_r = (beta_r s + theta_r s_W)/n``. ``tau`` and ``t_W`` come from the
cached spectrum, ``s`` and ``s_W`` from the exact power series
``m_k = 1'W^k 1`` (cross-checked once against a sparse solve), so a
1000-draw simulation costs no extra factorisation.

:func:`ols_spatial` and :func:`lm_spatial_tests` give the specification battery
that chooses among the four: the Lagrange-multiplier tests for a spatial lag
and a spatial error, their robust versions (Anselin, Bera, Florax & Yoon 1996),
the joint LM-SARMA, and the Cliff-Ord (1972) Moran's I of the OLS residuals.

Pure numpy / scipy / pandas: no PySAL at runtime.

Deliberately omitted
--------------------
* **SARAR / SAC** (``y = rho W1 y + X beta + u``, ``u = lam W2 u + eps``).
  There is no ``error_weights=`` argument anywhere in this module. ML would
  need a two-dimensional optimisation over ``(rho, lam)`` with two Jacobians,
  and the GMM three-step needs the Kelejian-Prucha (2010) ``Psi`` matrices for
  an honest ``lam`` standard error. Out of scope for this release; use
  :func:`sdm`, which nests the SEM common-factor restriction and is testable
  against it.
* **Heteroskedasticity-robust GMM** of Kelejian & Prucha (2010) /
  Arraiz, Drukker, Kelejian & Prucha (2010). ``vcov='robust'`` here is the
  White sandwich around the *homoskedastic* KP moment conditions; it corrects
  the standard errors, not the moment conditions themselves, and it does not
  make the GMM ``lam`` estimator heteroskedasticity-consistent.
* **A standard error for the GMM error parameter.** ``sem(method='gmm')``
  reports ``lam`` with ``bse['lambda'] = NaN`` for exactly that reason. The
  ``beta`` standard errors remain asymptotically valid: the feasible
  generalised spatial estimator has the same limiting distribution as the one
  with ``lam`` known (Kelejian & Prucha 1998, Thm. 2).
* **Stochastic (Monte-Carlo / Hutchinson) log-determinants and traces**
  (Barry & Pace 1999). Every determinant and trace here is exact. ``'lu'``
  costs ~0.2 s for 20 evaluations at ``n = 6400``, which covers anything a
  regional panel realistically needs, and a randomised Jacobian buys a jumpy
  optimiser and an irreproducible ``rho``.
* **Spatial-HAC covariance in** :func:`ols_spatial`. The LM battery is derived
  under homoskedastic Gaussian OLS residuals, so pairing it with a Conley
  covariance would mix two inconsistent assumptions in one printout. Call
  :func:`puremacro.spatial.conley_cov` directly for that.
* **Unbalanced / panel data.** :mod:`puremacro.spatial.models` is cross-section
  only.

References
----------
Ord, J.K. (1975). Estimation methods for models of spatial interaction.
    Journal of the American Statistical Association 70(349), 120-126.
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.
Anselin, L., Bera, A.K., Florax, R. and Yoon, M.J. (1996). Simple diagnostic
    tests for spatial dependence. Regional Science and Urban Economics 26(1),
    77-104.
Cliff, A.D. and Ord, J.K. (1972). Testing for spatial autocorrelation among
    regression residuals. Geographical Analysis 4(3), 267-284.
Kelejian, H.H. and Prucha, I.R. (1998). A generalized spatial two-stage least
    squares procedure for estimating a spatial autoregressive model with
    autoregressive disturbances. Journal of Real Estate Finance and Economics
    17(1), 99-121.
Kelejian, H.H. and Prucha, I.R. (1999). A generalized moments estimator for the
    autoregressive parameter in a spatial model. International Economic Review
    40(2), 509-533.
Kelejian, H.H. and Prucha, I.R. (2010). Specification and estimation of spatial
    autoregressive models with autoregressive and heteroskedastic disturbances.
    Journal of Econometrics 157(1), 53-67.
Pace, R.K. and Barry, R. (1997). Quick computation of regressions with a
    spatially autoregressive dependent variable. Geographical Analysis 29(3),
    232-247.
Pace, R.K. and LeSage, J.P. (2004). Chebyshev approximation of log-determinants
    of spatial weight matrices. Computational Statistics & Data Analysis 45(2),
    179-196.
LeSage, J.P. and Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC.
Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood
    estimators for spatial autoregressive models. Econometrica 72(6),
    1899-1925.
Lin, X. and Lee, L.-F. (2010). GMM estimation of spatial autoregressive models
    with unknown heteroskedasticity. Journal of Econometrics 157(1), 34-52.
Vijverberg, W. (2016). SLX models. (See LeSage & Pace 2009, sec. 2.4, for the
    spatially lagged X specification and Halleck Vega & Elhorst 2015.)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import optimize, stats
from scipy.sparse.linalg import splu, spsolve

from .._linalg import inv_xtx
from .weights import SpatialWeights

__all__ = [
    "sar",
    "sem",
    "sdm",
    "slx",
    "ols_spatial",
    "lm_spatial_tests",
    "spatial_effects",
    "SpatialModelResult",
    "SpatialOLSResult",
    "SpatialLMResult",
    "SpatialEffectsResult",
]

_DENSE_WARN_N = 2000
"""Above this many units the exact dense routes (``A^-1``, the OLS projector
``M``) allocate an ``n x n`` array; the code warns and proceeds rather than
switching to a randomised estimate."""

_LOGDET_METHODS = ("auto", "eig", "lu", "chebyshev")
_MAX_CHEBYSHEV_ORDER = 8


# ---------------------------------------------------------------------------
# Presentation shim (mirrors spatial/diagnostics.py; the integrator may hoist
# it into puremacro.reports)
# ---------------------------------------------------------------------------
def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst
    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


def _zstats(params: np.ndarray, bse: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(z, p, lo, hi)`` from asymptotic standard errors; NaN-safe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(bse > 0, params / bse, np.nan)
    p = 2.0 * stats.norm.sf(np.abs(z))
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lo = params - crit * bse
    hi = params + crit * bse
    return z, p, lo, hi


def _psd_inverse(info: np.ndarray, *, caller: str) -> np.ndarray:
    """Invert an information matrix, raising a message naming the caller."""
    info = 0.5 * (info + info.T)
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise np.linalg.LinAlgError(
            f"{caller}: the information matrix is singular, so no covariance can be "
            "reported. This usually means the spatial parameter sits on a bound or a "
            "regressor is collinear with its own spatial lag."
        ) from exc


# ---------------------------------------------------------------------------
# Design preparation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Design:
    y: np.ndarray
    X: np.ndarray
    names: tuple[str, ...]
    y_name: str
    obs_w: np.ndarray | None
    const_index: int | None

    @property
    def n(self) -> int:
        return self.y.shape[0]

    @property
    def k(self) -> int:
        return self.X.shape[1]


def _check_weights(weights: Any, caller: str) -> SpatialWeights:
    if not isinstance(weights, SpatialWeights):
        raise TypeError(
            f"{caller}: `weights` must be a puremacro.spatial.SpatialWeights, got "
            f"{type(weights).__name__}. Build one with contiguity_weights(), "
            "knn_weights(), distance_weights() or economic_weights()."
        )
    return weights


def _align(obj: Any, ids: tuple, caller: str, what: str) -> Any:
    """Reindex a pandas object onto ``ids`` by label, raising on a gap."""
    missing = [u for u in ids if u not in obj.index]
    if missing:
        raise KeyError(
            f"{caller}: {len(missing)} unit(s) of the weights are missing from the "
            f"{what} index, e.g. {missing[:3]}"
        )
    return obj.loc[list(ids)]


def _prepare_design(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    obs_weights: Any = None,
    caller: str = "spatial model",
) -> _Design:
    """Resolve ``(y, X)`` against ``weights`` into aligned finite arrays.

    Two input modes, and only two:

    * ``data=`` is a DataFrame and ``y`` / ``X`` are **column names** (a string
      and a string or sequence of strings). The frame is aligned onto
      ``weights.ids`` by label.
    * ``data=None`` and ``y`` / ``X`` are arrays, Series or DataFrames. Pandas
      inputs are aligned onto ``weights.ids`` by label; plain arrays are taken
      in ``weights.ids`` order.

    Mixing the two (a ``data=`` frame with array ``y``) raises, so the
    precedence question never arises. Explicit ``x_names`` always wins over
    DataFrame column names.
    """
    ids = weights.ids
    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{caller}: `data` must be a pandas DataFrame, got {type(data).__name__}")
        if not isinstance(y, str):
            raise ValueError(
                f"{caller}: with `data=` given, `y` must be a column name (str), got "
                f"{type(y).__name__}. Pass arrays with data=None instead."
            )
        if isinstance(X, str):
            xcols = [X]
        elif isinstance(X, (list, tuple)) and all(isinstance(c, str) for c in X):
            xcols = list(X)
        else:
            raise ValueError(
                f"{caller}: with `data=` given, `X` must be a column name or a sequence "
                "of column names. Pass arrays with data=None instead."
            )
        for col in [y, *xcols]:
            if col not in data.columns:
                raise KeyError(
                    f"{caller}: column {col!r} is not in `data` (columns: "
                    f"{list(data.columns)[:8]})"
                )
        frame = _align(data, ids, caller, "`data`")
        y_arr = frame[y].to_numpy(dtype=float)
        X_arr = frame[xcols].to_numpy(dtype=float)
        y_label = str(y)
        base_names = list(xcols)
    else:
        if isinstance(y, pd.Series):
            y_label = str(y.name) if y.name is not None else "y"
            y_arr = _align(y, ids, caller, "`y`").to_numpy(dtype=float)
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"{caller}: `y` must be one column, got {y.shape[1]}")
            y_label = str(y.columns[0])
            y_arr = _align(y, ids, caller, "`y`").to_numpy(dtype=float).ravel()
        else:
            y_label = "y"
            y_arr = np.asarray(y, dtype=float).ravel()
        if isinstance(X, pd.DataFrame):
            base_names = [str(c) for c in X.columns]
            X_arr = _align(X, ids, caller, "`X`").to_numpy(dtype=float)
        elif isinstance(X, pd.Series):
            base_names = [str(X.name) if X.name is not None else "x1"]
            X_arr = _align(X, ids, caller, "`X`").to_numpy(dtype=float).reshape(-1, 1)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            base_names = [f"x{i + 1}" for i in range(X_arr.shape[1])]

    if X_arr.ndim != 2:
        raise ValueError(f"{caller}: `X` must be two-dimensional, got shape {X_arr.shape}")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(
            f"{caller}: y has {y_arr.shape[0]} observations but X has {X_arr.shape[0]}"
        )
    if y_arr.shape[0] != weights.n:
        raise ValueError(
            f"{caller}: y has {y_arr.shape[0]} observations but the weights describe "
            f"{weights.n} units"
        )
    if x_names is not None:
        x_names = [str(s) for s in x_names]
        if len(x_names) != X_arr.shape[1]:
            raise ValueError(
                f"{caller}: x_names has {len(x_names)} entries for {X_arr.shape[1]} columns"
            )
        base_names = list(x_names)

    if not np.all(np.isfinite(y_arr)):
        raise ValueError(f"{caller}: y contains NaN or inf")
    bad = [base_names[j] for j in range(X_arr.shape[1]) if not np.all(np.isfinite(X_arr[:, j]))]
    if bad:
        raise ValueError(f"{caller}: X contains NaN or inf in column(s) {bad}")

    const_cols = [
        j for j in range(X_arr.shape[1])
        if np.ptp(X_arr[:, j]) == 0.0 and X_arr[0, j] != 0.0
    ]
    if add_constant:
        if const_cols:
            raise ValueError(
                f"{caller}: X already contains a constant column "
                f"({[base_names[j] for j in const_cols]}) and add_constant=True. "
                "Pass add_constant=False, or drop the column."
            )
        X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        names = tuple(["const", *base_names])
        const_index: int | None = 0
    else:
        names = tuple(base_names)
        const_index = const_cols[0] if const_cols else None
    if len(set(names)) != len(names):
        raise ValueError(f"{caller}: duplicate regressor names in {names}")

    n, k = X_arr.shape
    if n <= k + 2:
        raise ValueError(
            f"{caller}: n = {n} observations for k = {k} regressors; the concentrated "
            "likelihood needs n > k + 2"
        )

    # A degenerate outcome must be rejected here, not silently fitted: with no
    # residual variation the concentrated likelihood is flat (or sigma^2 is
    # zero), and the estimators answer with a confident number anyway — `sem`
    # reports a significant lambda, `sar`/`sdm` pin rho to the boundary and
    # `ols_spatial` divides by zero.
    tail = ("the spatial parameter is not identified"
            if caller in ("sar", "sdm", "sem") else "nothing is identified")
    if np.ptp(y_arr) == 0.0:
        raise ValueError(f"{caller}: the outcome is constant, so {tail}")
    beta_ols = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    resid_ols = y_arr - X_arr @ beta_ols
    if float(resid_ols @ resid_ols) <= n * float(np.finfo(float).eps) * float(y_arr @ y_arr):
        raise ValueError(
            f"{caller}: the outcome has no variation left after the regressors, so {tail}"
        )

    ow: np.ndarray | None = None
    if obs_weights is not None:
        if isinstance(obs_weights, (pd.Series, pd.DataFrame)):
            ow = _align(obs_weights, ids, caller, "`obs_weights`").to_numpy(dtype=float).ravel()
        else:
            ow = np.asarray(obs_weights, dtype=float).ravel()
        if ow.shape[0] != n:
            raise ValueError(
                f"{caller}: obs_weights has {ow.shape[0]} entries for {n} observations"
            )
        if not np.all(np.isfinite(ow)):
            raise ValueError(f"{caller}: obs_weights contains NaN or inf")
        if np.any(ow <= 0):
            raise ValueError(f"{caller}: obs_weights must be strictly positive")
        ow = ow / ow.mean()

    return _Design(y=y_arr, X=X_arr, names=names, y_name=y_label, obs_w=ow, const_index=const_index)


# ---------------------------------------------------------------------------
# Spectrum, admissible interval, log-determinant
# ---------------------------------------------------------------------------
def _symmetrizing_scale(weights: SpatialWeights, *, tol: float = 1e-9) -> np.ndarray | None:
    """Positive ``d`` with ``D^{1/2} W D^{-1/2}`` symmetric, or ``None``.

    A row-standardised contiguity, distance-band or Gaussian-decay ``W`` is
    ``D_r^{-1} B`` with ``B`` symmetric, hence similar to a symmetric matrix and
    guaranteed real eigenvalues. A k-nearest-neighbour or economic-flow ``W`` is
    not, and genuinely has complex eigenvalues.

    The scale is propagated over a breadth-first traversal of each connected
    component (``d_j = d_i W_ij / W_ji``) and then **verified**: the function
    returns ``d`` only when ``max |S - S'| < tol * max(1, max|S|)`` for
    ``S = D^{1/2} W D^{-1/2}``. The sparsity pattern is a cheap early-out, never
    the acceptance test — a row-standardised economic-flow matrix built from a
    dense flow table has a perfectly symmetric pattern and still has complex
    eigenvalues.

    Islands (zero rows and columns) take ``d_i = 1``.
    """
    W = weights.W
    n = weights.n
    if n == 0 or W.nnz == 0:
        return np.ones(n)
    coo = W.tocoo()
    fwd: dict[int, dict[int, float]] = {}
    for i, j, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        fwd.setdefault(i, {})[j] = v
    # cheap early-out: an entry with no transpose partner cannot be scaled
    for i, row in fwd.items():
        for j in row:
            if fwd.get(j, {}).get(i, 0.0) <= 0.0:
                return None
    d = np.zeros(n)
    for start in range(n):
        if d[start] != 0.0:
            continue
        if start not in fwd:
            d[start] = 1.0
            continue
        d[start] = 1.0
        stack = [start]
        while stack:
            i = stack.pop()
            for j, w_ij in fwd[i].items():
                w_ji = fwd[j][i]
                cand = d[i] * w_ij / w_ji
                if not np.isfinite(cand) or cand <= 0.0:
                    return None
                if d[j] == 0.0:
                    d[j] = cand
                    stack.append(j)
                elif abs(d[j] - cand) > tol * max(1.0, abs(cand)):
                    return None  # inconsistent around a cycle
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return None
    root = np.sqrt(d)
    S = sp.diags(root) @ W @ sp.diags(1.0 / root)
    asym = abs(S - S.T)
    scale = float(abs(S).max()) if S.nnz else 1.0
    if float(asym.max() if asym.nnz else 0.0) > tol * max(1.0, scale):
        return None
    return d


def _spectrum(weights: SpatialWeights, *, escape: str | None = None) -> tuple[np.ndarray, bool]:
    """Eigenvalues of ``W`` and whether the symmetric route was used.

    Returns a complex array (real dtype when the symmetric route ran). The
    dense eigendecomposition is ``O(n^3)``; above ``_DENSE_WARN_N`` units a
    ``RuntimeWarning`` names the size. There is no randomised alternative
    by design.

    ``escape`` is the caller-supplied sentence naming the arguments that would
    actually avoid this decomposition *in the call that triggered it*. It is
    caller-supplied because the answer differs by route: ``logdet='lu'`` is no
    escape at all when the spectrum is being built for ``rho_bounds='auto'``,
    and it is not even a legal argument under ``method='gmm'``. Never advertise
    an action the caller cannot take (build rule G3).
    """
    n = weights.n
    if n > _DENSE_WARN_N:
        advice = escape or (
            "Pass an explicit rho_bounds/lambda_bounds pair and logdet='lu' to avoid it."
        )
        warnings.warn(
            f"spatial models: computing the exact dense spectrum of a {n} x {n} weights "
            f"matrix (O(n^2) memory, O(n^3) time). {advice}",
            RuntimeWarning, stacklevel=3,
        )
    scale = _symmetrizing_scale(weights)
    if scale is not None:
        root = np.sqrt(scale)
        S = (sp.diags(root) @ weights.W @ sp.diags(1.0 / root)).toarray()
        S = 0.5 * (S + S.T)
        return np.linalg.eigvalsh(S).astype(float), True
    return np.linalg.eigvals(weights.W.toarray()), False


def _admissible_bounds(
    spectrum: np.ndarray, *, pad: float = 1e-8
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(searched, singularity)`` intervals for the spatial parameter.

    ``singularity`` is where ``I - rho W`` is non-singular: ``rho`` is real, so
    ``det(I - rho W) = 0`` requires a **real** eigenvalue, and the interval
    containing zero is ``(1/lam_min, 1/lam_max)`` over the real eigenvalues
    (infinite on a side with no eigenvalue of that sign).

    ``searched`` intersects that with ``(-1/max|lam|, 1/max|lam|)`` taken over
    the **full complex** spectrum: outside it the Neumann expansion
    ``A^-1 = sum rho^k W^k`` diverges and the spatial-multiplier reading of the
    model is gone. For any row-standardised ``W`` the Perron root is exactly 1,
    so this is ``(-1, 1)``. Reporting only the singularity interval would hand
    the optimiser a bracket ten units wide on an economic-flow ``W``.
    """
    lam = np.asarray(spectrum)
    real = lam.real[np.abs(lam.imag) <= 1e-12 * np.maximum(1.0, np.abs(lam.real))]
    pos = real[real > 1e-12]
    neg = real[real < -1e-12]
    hi_sing = float(1.0 / pos.max()) if pos.size else float("inf")
    lo_sing = float(1.0 / neg.min()) if neg.size else float("-inf")
    radius = float(np.abs(lam).max()) if lam.size else 0.0
    if radius > 0:
        lo = max(lo_sing, -1.0 / radius)
        hi = min(hi_sing, 1.0 / radius)
    else:  # every unit an island: W = 0, any rho is admissible
        lo, hi = -1.0, 1.0
    width = hi - lo
    return (lo + pad * width, hi - pad * width), (lo_sing, hi_sing)


def _chebyshev_traces(weights: SpatialWeights, order: int) -> np.ndarray:
    """``tr T_j(W)`` for ``j = 0 .. order`` by the sparse Chebyshev recurrence.

    ``T_0 = I``, ``T_1 = W``, ``T_{j+1} = 2 W T_j - T_{j-1}``. The matrices fill
    in: on a 30x30 rook lattice ``W`` is 0.4% dense, ``W^5`` about 3% and
    ``W^8`` about 7%, which is why the order is capped at
    ``_MAX_CHEBYSHEV_ORDER``.
    """
    n = weights.n
    W = weights.W.tocsr()
    tr = np.empty(order + 1)
    prev = sp.identity(n, format="csr")
    tr[0] = float(n)
    if order == 0:
        return tr
    cur = W.copy()
    tr[1] = float(cur.diagonal().sum())
    for j in range(2, order + 1):
        nxt = (2.0 * (W @ cur) - prev).tocsr()
        prev, cur = cur, nxt
        tr[j] = float(cur.diagonal().sum())
    return tr


class _LogDet:
    """Exact (or, on request, Chebyshev) ``ln|I - rho W|`` as a callable.

    Parameters
    ----------
    weights : SpatialWeights
    method : {'auto', 'eig', 'lu', 'chebyshev'}
        ``'auto'`` resolves to ``'eig'`` for ``n <= 1000`` and ``'lu'`` above.
    order : int
        Chebyshev degree (ignored otherwise).
    spectrum : ndarray, optional
        A pre-computed spectrum to reuse.
    caller : str
        Prefix for error messages.

    Notes
    -----
    ``'eig'`` uses ``sum_i 0.5 ln((1 - rho Re lam_i)^2 + (rho Im lam_i)^2)``,
    which is exactly ``sum_i ln|1 - rho lam_i|`` and stays correct for the
    complex-conjugate pairs an asymmetric ``W`` produces. ``'lu'`` uses
    ``sum ln|diag(U)|`` from a sparse LU of ``I - rho W`` (Pace & Barry 1997);
    ``L`` is unit lower triangular so only ``U`` contributes.
    """

    def __init__(
        self,
        weights: SpatialWeights,
        *,
        method: str = "auto",
        order: int = 5,
        spectrum: np.ndarray | None = None,
        caller: str = "spatial model",
    ) -> None:
        if method not in _LOGDET_METHODS:
            raise ValueError(
                f"{caller}: logdet must be one of {_LOGDET_METHODS}, got {method!r}"
            )
        resolved = method
        if method == "auto":
            resolved = "eig" if weights.n <= 1000 else "lu"
        self.weights = weights
        self.method = resolved
        self.order = int(order)
        self.spectrum: np.ndarray | None = spectrum
        self._eye = sp.identity(weights.n, format="csc")
        self._Wcsc = weights.W.tocsc()
        self._cheb_traces: np.ndarray | None = None
        if resolved == "eig" and self.spectrum is None:
            self.spectrum, _ = _spectrum(weights)
        if resolved == "chebyshev":
            if not (1 <= self.order <= _MAX_CHEBYSHEV_ORDER):
                raise ValueError(
                    f"{caller}: logdet_order must lie in [1, {_MAX_CHEBYSHEV_ORDER}] for "
                    f"logdet='chebyshev', got {self.order}"
                )
            if _symmetrizing_scale(weights) is None:
                raise ValueError(
                    f"{caller}: logdet='chebyshev' needs a symmetric or symmetrisable W "
                    f"(this one, kind={weights.kind!r}, is not; its eigenvalues are "
                    "complex). Use logdet='eig' or logdet='lu'."
                )
            rowmax = float(weights.row_sums().max()) if weights.n else 0.0
            if rowmax > 1.0 + 1e-8:
                raise ValueError(
                    f"{caller}: logdet='chebyshev' assumes the eigenvalues of W lie in "
                    f"[-1, 1]; the largest row sum here is {rowmax:.4f}. Row-standardise "
                    "the weights with W.standardize(), or use logdet='eig'/'lu'."
                )
            self._cheb_traces = _chebyshev_traces(weights, self.order)

    def __call__(self, rho: float) -> float:
        if self.method == "eig":
            lam = np.asarray(self.spectrum)
            re = 1.0 - rho * lam.real
            im = rho * lam.imag if np.iscomplexobj(lam) else 0.0
            val = re * re + (im * im if np.ndim(im) else 0.0)
            if np.any(val <= 0.0):
                return float("-inf")
            return float(0.5 * np.sum(np.log(val)))
        if self.method == "lu":
            A = (self._eye - rho * self._Wcsc).tocsc()
            try:
                lu = splu(A, permc_spec="COLAMD")
            except RuntimeError:
                return float("-inf")
            diag = lu.U.diagonal()
            if np.any(diag == 0.0):
                return float("-inf")
            return float(np.sum(np.log(np.abs(diag))))
        return self._chebyshev(rho)

    def _chebyshev(self, rho: float) -> float:
        """Pace & LeSage (2004) degree-``q`` expansion."""
        q = self.order
        traces = self._cheb_traces
        assert traces is not None
        m = q + 1
        kk = np.arange(1, m + 1)
        nodes = np.cos(np.pi * (kk - 0.5) / m)
        vals = np.log1p(-rho * nodes)
        j = np.arange(m)
        basis = np.cos(np.pi * j[:, None] * (kk[None, :] - 0.5) / m)
        c = (2.0 / m) * (basis @ vals)
        return float(np.dot(c, traces[:m]) - 0.5 * self.weights.n * c[0])


# ---------------------------------------------------------------------------
# Estimation cores
# ---------------------------------------------------------------------------
def _weighted(arr: np.ndarray, obs_w: np.ndarray | None) -> np.ndarray:
    """``P a`` with ``P = diag(sqrt(w))`` (identity when ``obs_w is None``)."""
    if obs_w is None:
        return arr
    root = np.sqrt(obs_w)
    return arr * (root[:, None] if arr.ndim == 2 else root)


def _maximise_profile(
    objective,
    bounds: tuple[float, float],
    xtol: float,
) -> tuple[float, bool, int]:
    """Bounded maximisation of a smooth scalar ``objective``."""
    lo, hi = bounds
    res = optimize.minimize_scalar(
        lambda r: -objective(float(r)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": xtol, "maxiter": 500},
    )
    return float(res.x), bool(res.success), int(getattr(res, "nfev", 0))


def _lag_ml_core(
    y: np.ndarray,
    Z: np.ndarray,
    weights: SpatialWeights,
    *,
    obs_w: np.ndarray | None,
    logdet: _LogDet,
    bounds: tuple[float, float],
    xtol: float,
    caller: str,
) -> dict[str, Any]:
    """Concentrated Gaussian ML for ``y = rho W y + Z delta + eps``.

    The ``c0 / c1 / c2`` profile trick: two auxiliary least-squares fits give
    ``e0`` and ``eL`` once, after which ``SSR(rho) = c0 - 2 rho c1 + rho^2 c2``.
    With observation weights the spatial lag is formed **first** and weighted
    **second** (``P(Wy)``, never ``W(Py)``).
    """
    n = y.shape[0]
    Wy = weights.W @ y
    Zw = _weighted(Z, obs_w)
    yw = _weighted(y, obs_w)
    Wyw = _weighted(Wy, obs_w)
    ZtZ_inv = inv_xtx(Zw, name=caller)
    b0 = ZtZ_inv @ (Zw.T @ yw)
    bL = ZtZ_inv @ (Zw.T @ Wyw)
    e0 = yw - Zw @ b0
    eL = Wyw - Zw @ bL
    c0 = float(e0 @ e0)
    c1 = float(e0 @ eL)
    c2 = float(eL @ eL)
    const = -0.5 * n * (np.log(2.0 * np.pi) + 1.0)
    logw = 0.5 * float(np.sum(np.log(obs_w))) if obs_w is not None else 0.0

    def ssr(rho: float) -> float:
        return c0 - 2.0 * rho * c1 + rho * rho * c2

    def objective(rho: float) -> float:
        s = ssr(rho)
        if s <= 0.0:
            return float("-inf")
        return const + logdet(rho) - 0.5 * n * np.log(s / n) + logw

    rho_hat, converged, n_iter = _maximise_profile(objective, bounds, xtol)

    score_root: float | None = None
    spectrum = logdet.spectrum
    if spectrum is not None:
        lam = np.asarray(spectrum)

        def score(rho: float) -> float:
            denom = 1.0 - rho * lam
            return float(-np.sum(np.real(lam / denom)) - n * (rho * c2 - c1) / ssr(rho))

        # The bracket must be WIDER than the disagreement it is meant to
        # detect, or the warning below is unreachable dead code: a root found
        # inside [rho_hat - h, rho_hat + h] is trivially within h of rho_hat.
        # gate = 100 * xtol (the optimiser's own precision, floored at 1e-6);
        # bracket = 100 * gate, so a genuine disagreement is both findable and
        # reportable.
        gate = max(100.0 * xtol, 1e-6)
        half = 100.0 * gate
        interior = min(abs(rho_hat - bounds[0]), abs(rho_hat - bounds[1])) > 10.0 * xtol
        a, b = max(bounds[0], rho_hat - half), min(bounds[1], rho_hat + half)
        try:
            if score(a) * score(b) < 0.0:
                score_root = float(optimize.brentq(score, a, b, xtol=1e-14, rtol=1e-15))
        except (ValueError, RuntimeError):  # pragma: no cover - defensive
            score_root = None
        if score_root is not None:
            if abs(score_root - rho_hat) > gate:
                warnings.warn(
                    f"{caller}: the bounded optimum ({rho_hat:.10f}) and the root of the "
                    f"analytic score ({score_root:.10f}) disagree by "
                    f"{abs(score_root - rho_hat):.2e} (> {gate:g}); keeping the bounded "
                    "optimum.",
                    RuntimeWarning, stacklevel=3,
                )
            else:
                rho_hat = score_root
        elif interior:
            # The genuinely diagnostic case: an interior optimum whose score
            # does not change sign across the bracket is not a stationary point.
            warnings.warn(
                f"{caller}: the analytic score does not change sign across "
                f"[{a:.10f}, {b:.10f}] around the interior bounded optimum "
                f"({rho_hat:.10f}), so it could not be refined and score_root is None. "
                "The concentrated likelihood may be flat or non-smooth there; treat "
                "inference on the spatial parameter with care.",
                RuntimeWarning, stacklevel=3,
            )

    lo, hi = bounds
    at_bound = min(abs(rho_hat - lo), abs(rho_hat - hi)) <= 10.0 * xtol
    if at_bound:
        warnings.warn(
            f"{caller}: the spatial parameter ({rho_hat:.6f}) sits on the boundary of the "
            f"admissible interval ({lo:.6f}, {hi:.6f}); the likelihood is not interior, so "
            "the asymptotic standard error understates the true uncertainty.",
            RuntimeWarning, stacklevel=3,
        )

    beta = b0 - rho_hat * bL
    resid_w = e0 - rho_hat * eL
    sigma2 = float(resid_w @ resid_w) / n
    resid = (y - rho_hat * Wy) - Z @ beta
    return {
        "rho": float(rho_hat),
        "beta": beta,
        "resid": resid,
        "sigma2": sigma2,
        "loglik": objective(rho_hat),
        "loglik_at_zero": objective(0.0),
        "objective": objective,
        "converged": converged,
        "n_iter": n_iter,
        "at_bound": at_bound,
        "score_root": score_root,
    }


def _error_ml_core(
    y: np.ndarray,
    X: np.ndarray,
    weights: SpatialWeights,
    *,
    obs_w: np.ndarray | None,
    logdet: _LogDet,
    bounds: tuple[float, float],
    xtol: float,
    caller: str,
) -> dict[str, Any]:
    """Concentrated Gaussian ML for ``y = X beta + u``, ``u = lam W u + eps``.

    Unlike the lag model there is no scalar shortcut — ``beta(lam)`` is the
    spatially filtered GLS estimator and must be recomputed at every ``lam`` —
    but the ten Gram matrices of ``[X, WX, y, Wy]`` are formed once, so each
    evaluation is ``O(k^2)`` with no pass over the data.

    The Gram solve is gated by :func:`puremacro._linalg.inv_xtx` (build rule
    G7): once up front on the unfiltered design, which is what makes ``sem``
    honour the same "the message names the collinear columns" promise as
    ``sar``/``sdm``/``slx``, and again on the filtered design if a solve
    inside the profile loop ever fails at a non-zero ``lambda``.
    """
    n = y.shape[0]
    WX = weights.W @ X
    Wy = weights.W @ y
    Xw, WXw = _weighted(X, obs_w), _weighted(WX, obs_w)
    yw, Wyw = _weighted(y, obs_w), _weighted(Wy, obs_w)
    inv_xtx(Xw, name=caller)  # G7 gate: raises naming the collinear columns
    XtX = Xw.T @ Xw
    XtWX = Xw.T @ WXw
    WXtWX = WXw.T @ WXw
    Xty = Xw.T @ yw
    XtWy = Xw.T @ Wyw
    WXty = WXw.T @ yw
    WXtWy = WXw.T @ Wyw
    yty = float(yw @ yw)
    ytWy = float(yw @ Wyw)
    WytWy = float(Wyw @ Wyw)
    const = -0.5 * n * (np.log(2.0 * np.pi) + 1.0)
    logw = 0.5 * float(np.sum(np.log(obs_w))) if obs_w is not None else 0.0

    def pieces(lam: float) -> tuple[np.ndarray, np.ndarray, float]:
        XsXs = XtX - lam * (XtWX + XtWX.T) + lam * lam * WXtWX
        Xsys = Xty - lam * XtWy - lam * WXty + lam * lam * WXtWy
        ysys = yty - 2.0 * lam * ytWy + lam * lam * WytWy
        return XsXs, Xsys, ysys

    def fit(lam: float) -> tuple[np.ndarray, float, np.ndarray]:
        XsXs, Xsys, ysys = pieces(lam)
        try:
            beta = np.linalg.solve(XsXs, Xsys)
        except np.linalg.LinAlgError:
            # Re-raise through inv_xtx so the message names the columns of the
            # spatially filtered design that went collinear at this lambda.
            inv_xtx(Xw - lam * WXw, name=f"{caller} (filtered at lambda={lam:.6f})")
            raise
        ssr = ysys - float(Xsys @ beta)
        return beta, max(ssr, 0.0), XsXs

    def objective(lam: float) -> float:
        _, ssr, _ = fit(lam)
        if ssr <= 0.0:
            return float("-inf")
        return const + logdet(lam) - 0.5 * n * np.log(ssr / n) + logw

    lam_hat, converged, n_iter = _maximise_profile(objective, bounds, xtol)
    lo, hi = bounds
    at_bound = min(abs(lam_hat - lo), abs(lam_hat - hi)) <= 10.0 * xtol
    if at_bound:
        warnings.warn(
            f"{caller}: the spatial error parameter ({lam_hat:.6f}) sits on the boundary of "
            f"the admissible interval ({lo:.6f}, {hi:.6f}).",
            RuntimeWarning, stacklevel=3,
        )
    beta, ssr, XsXs = fit(lam_hat)
    sigma2 = ssr / n
    resid = (y - lam_hat * Wy) - (X - lam_hat * WX) @ beta
    return {
        "lam": float(lam_hat),
        "beta": beta,
        "resid": resid,
        "sigma2": sigma2,
        "XsXs": XsXs,
        "loglik": objective(lam_hat),
        "loglik_at_zero": objective(0.0),
        "objective": objective,
        "converged": converged,
        "n_iter": n_iter,
        "at_bound": at_bound,
        "score_root": None,
    }


def _dense_multiplier(weights: SpatialWeights, rho: float, *, caller: str) -> np.ndarray:
    """``W (I - rho W)^-1`` as a dense array (exact; warns above the threshold)."""
    n = weights.n
    if n > _DENSE_WARN_N:
        warnings.warn(
            f"{caller}: forming the dense {n} x {n} spatial multiplier W(I - rho W)^-1 for "
            "the analytic covariance (O(n^2) memory). This is exact by design — there is "
            "no randomised trace estimator in this module.",
            RuntimeWarning, stacklevel=3,
        )
    A = (sp.identity(n, format="csc") - rho * weights.W.tocsc()).toarray()
    Ainv = np.linalg.solve(A, np.eye(n))
    return np.asarray(weights.W @ Ainv)


def _trace_pair(WA: np.ndarray, spectrum: np.ndarray | None, rho: float,
                obs_w: np.ndarray | None) -> tuple[float, float, float]:
    """``(tr WA, tr(WA WA), tr(Om^-1 WA Om WA'))`` for the information matrix."""
    if spectrum is not None:
        lam = np.asarray(spectrum)
        g = lam / (1.0 - rho * lam)
        tr_wa = float(np.real(np.sum(g)))
        tr_wawa = float(np.real(np.sum(g * g)))
    else:  # pragma: no cover - the spectrum is always available in practice
        tr_wa = float(np.trace(WA))
        tr_wawa = float(np.sum(WA * WA.T))
    if obs_w is None:
        tr_mixed = float(np.sum(WA * WA))
    else:
        ratio = obs_w[:, None] / obs_w[None, :]
        tr_mixed = float(np.sum(ratio * WA * WA))
    return tr_wa, tr_wawa, tr_mixed


def _lag_information(
    Z: np.ndarray,
    weights: SpatialWeights,
    *,
    beta: np.ndarray,
    rho: float,
    sigma2: float,
    obs_w: np.ndarray | None,
    spectrum: np.ndarray | None,
    caller: str,
) -> np.ndarray:
    """Anselin (1988, eq. 6.14) information matrix in the order (beta, rho, sigma2)."""
    n, k = Z.shape
    WA = _dense_multiplier(weights, rho, caller=caller)
    Zb = Z @ beta
    v = WA @ Zb
    w = obs_w if obs_w is not None else np.ones(n)
    tr_wa, tr_wawa, tr_mixed = _trace_pair(WA, spectrum, rho, obs_w)
    info = np.zeros((k + 2, k + 2))
    info[:k, :k] = (Z.T * w) @ Z / sigma2
    info[:k, k] = (Z.T * w) @ v / sigma2
    info[k, :k] = info[:k, k]
    info[k, k] = tr_wawa + tr_mixed + float((v * w) @ v) / sigma2
    info[k, k + 1] = tr_wa / sigma2
    info[k + 1, k] = info[k, k + 1]
    info[k + 1, k + 1] = n / (2.0 * sigma2 * sigma2)
    return info


def _error_information(
    Xs: np.ndarray,
    weights: SpatialWeights,
    *,
    lam: float,
    sigma2: float,
    obs_w: np.ndarray | None,
    spectrum: np.ndarray | None,
    caller: str,
) -> np.ndarray:
    """Anselin (1988, eq. 6.28) information matrix in the order (beta, lambda, sigma2).

    Block diagonal between ``beta`` and ``(lambda, sigma2)``, which is why
    ``SE(beta) = sqrt(diag(sigma2 (Xs'Xs)^-1))`` is exact here and not a
    plug-in approximation.
    """
    n, k = Xs.shape
    WB = _dense_multiplier(weights, lam, caller=caller)
    w = obs_w if obs_w is not None else np.ones(n)
    tr_wb, tr_wbwb, tr_mixed = _trace_pair(WB, spectrum, lam, obs_w)
    info = np.zeros((k + 2, k + 2))
    info[:k, :k] = (Xs.T * w) @ Xs / sigma2
    info[k, k] = tr_wbwb + tr_mixed
    info[k, k + 1] = tr_wb / sigma2
    info[k + 1, k] = info[k, k + 1]
    info[k + 1, k + 1] = n / (2.0 * sigma2 * sigma2)
    return info


def _numerical_hessian_vcov(neg_loglik, params: np.ndarray) -> np.ndarray:
    """Observed-information covariance by central differences.

    The step is **relative** (``h_i = 1e-5 max(1, |x_i|)``): a fixed absolute
    step evaluates ``ln sigma^2`` at a negative argument whenever the data are
    scaled so that ``sigma^2`` is small. The sigma2 coordinate is differentiated
    in ``ln sigma^2`` by the caller and mapped back with the Jacobian.
    """
    p = params.shape[0]
    h = 1e-5 * np.maximum(1.0, np.abs(params))
    H = np.zeros((p, p))
    f0 = neg_loglik(params)
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = h[i]
            ej = np.zeros(p); ej[j] = h[j]
            if i == j:
                H[i, i] = (neg_loglik(params + ei) - 2.0 * f0 + neg_loglik(params - ei)) / (h[i] ** 2)
            else:
                val = (
                    neg_loglik(params + ei + ej) - neg_loglik(params + ei - ej)
                    - neg_loglik(params - ei + ej) + neg_loglik(params - ei - ej)
                ) / (4.0 * h[i] * h[j])
                H[i, j] = H[j, i] = val
    return H


def _build_instruments(
    X_included: np.ndarray,
    X_ex: np.ndarray,
    weights: SpatialWeights,
    q: int,
    *,
    caller: str,
) -> np.ndarray:
    """``H = [X_included, W X_ex, ..., W^q X_ex]`` with redundant columns dropped.

    Kelejian & Prucha (1998) instrument ``Wy`` with the spatial lags of the
    exogenous regressors. For a row-standardised ``W`` the higher lags become
    nearly collinear with the lower ones; a column whose residual after
    projection on the columns already kept has norm below ``1e-10`` times its
    own norm is dropped, with a ``RuntimeWarning`` naming the lag order.
    """
    blocks = [X_included]
    orders = [0] * X_included.shape[1]
    cur = X_ex
    for order in range(1, q + 1):
        cur = weights.W @ cur
        blocks.append(cur)
        orders.extend([order] * cur.shape[1])
    H = np.column_stack(blocks)
    n_included = X_included.shape[1]
    keep: list[int] = []
    basis: list[np.ndarray] = []
    n_basis_included = 0
    dropped: list[int] = []

    def _residual(col: np.ndarray, upto: int) -> np.ndarray:
        res = col.copy()
        for b in basis[:upto]:
            res = res - float(b @ res) * b
        return res

    for j in range(H.shape[1]):
        col = H[:, j].astype(float)
        nrm = float(np.linalg.norm(col))
        tol = 1e-10 * max(nrm, 1.0)
        if j < n_included:
            res = _residual(col, len(basis))
            if nrm == 0.0 or float(np.linalg.norm(res)) < tol:
                continue  # rank failure in the design itself is inv_xtx's job
            basis.append(res / float(np.linalg.norm(res)))
            n_basis_included = len(basis)
            keep.append(j)
            continue
        # A lagged regressor that is already an INCLUDED regressor (the normal
        # case for the SDM, whose Durbin block is W X_d) cannot instrument for
        # W y. That is structural, not a diagnostic, so it is dropped silently.
        if float(np.linalg.norm(_residual(col, n_basis_included))) < tol:
            continue
        res = _residual(col, len(basis))
        if float(np.linalg.norm(res)) < tol:
            dropped.append(orders[j])
            continue
        basis.append(res / float(np.linalg.norm(res)))
        keep.append(j)
    if dropped:
        warnings.warn(
            f"{caller}: {len(dropped)} instrument column(s) at spatial lag order(s) "
            f"{sorted(set(dropped))} were linearly dependent on the lower-order lags and "
            "were dropped. Lower gmm_lags to silence this.",
            RuntimeWarning, stacklevel=3,
        )
    return H[:, keep]


def _lag_gmm_core(
    y: np.ndarray,
    Z_exog: np.ndarray,
    X_ex: np.ndarray,
    weights: SpatialWeights,
    *,
    gmm_lags: int,
    vcov: str,
    caller: str,
) -> dict[str, Any]:
    """Kelejian-Prucha (1998) spatial two-stage least squares.

    ``Z = [Z_exog, W y]``, ``H = [Z_exog, W X_ex, ..., W^q X_ex]``,
    ``delta = (Z' P_H Z)^-1 Z' P_H y``. Because ``P_H`` is idempotent,
    ``Z' P_H Z = Zh' Zh`` with ``Zh = P_H Z``, which is what the covariance
    uses.
    """
    n = y.shape[0]
    Wy = weights.W @ y
    Z = np.column_stack([Z_exog, Wy])
    H = _build_instruments(Z_exog, X_ex, weights, gmm_lags, caller=caller)
    if H.shape[1] < Z.shape[1]:
        raise ValueError(
            f"{caller}: only {H.shape[1]} usable instruments for {Z.shape[1]} regressors. "
            "Raise gmm_lags, or add an exogenous regressor."
        )
    HtH_inv = inv_xtx(H, name=f"{caller} (instruments)")
    Zh = H @ (HtH_inv @ (H.T @ Z))
    ZhZh_inv = inv_xtx(Zh, name=f"{caller} (2SLS)")
    delta = ZhZh_inv @ (Zh.T @ y)
    resid = y - Z @ delta
    sigma2 = float(resid @ resid) / n
    if vcov == "robust":
        meat = (Zh.T * (resid ** 2)) @ Zh
        cov = ZhZh_inv @ meat @ ZhZh_inv
    else:
        cov = sigma2 * ZhZh_inv
    return {
        "delta": delta,
        "resid": resid,
        "sigma2": sigma2,
        "vcov": cov,
        "n_instruments": int(H.shape[1]),
    }


def _kp_moments(u: np.ndarray, weights: SpatialWeights, trWtW: float) -> tuple[np.ndarray, np.ndarray]:
    """The Kelejian-Prucha (1999) three-moment system ``(G, g)``."""
    n = u.shape[0]
    ub = weights.W @ u
    ubb = weights.W @ ub
    G = np.array([
        [2.0 * float(u @ ub), -float(ub @ ub), float(n)],
        [2.0 * float(ub @ ubb), -float(ubb @ ubb), trWtW],
        [float(u @ ubb) + float(ub @ ub), -float(ub @ ubb), 0.0],
    ]) / n
    g = np.array([float(u @ u), float(ub @ ub), float(u @ ub)]) / n
    return G, g


def _kp_lambda(u: np.ndarray, weights: SpatialWeights, trWtW: float,
               bounds: tuple[float, float]) -> tuple[float, float]:
    """Solve the KP three-moment system under ``alpha_2 = alpha_1^2``.

    Bounded L-BFGS-B rather than an unbounded Nelder-Mead: the surface is a
    two-dimensional nonlinear least-squares problem and an unbounded simplex
    can walk ``lambda`` outside the admissible interval.
    """
    G, g = _kp_moments(u, weights, trWtW)
    n = u.shape[0]
    start = np.array([0.0, float(u @ u) / n])

    def crit(p: np.ndarray) -> float:
        lam, s2 = float(p[0]), float(p[1])
        resid = g - G @ np.array([lam, lam * lam, s2])
        return float(resid @ resid)

    res = optimize.minimize(
        crit, start, method="L-BFGS-B",
        bounds=[(bounds[0], bounds[1]), (1e-12, None)],
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 2000},
    )
    return float(res.x[0]), float(res.x[1])


def _error_gmm_core(
    y: np.ndarray,
    X: np.ndarray,
    weights: SpatialWeights,
    *,
    n_iter: int,
    vcov: str,
    bounds: tuple[float, float],
    xtol: float,
    caller: str,
) -> dict[str, Any]:
    """Kelejian-Prucha (1999) GM estimator followed by spatially filtered OLS."""
    n = y.shape[0]
    trWtW = float(weights.W.multiply(weights.W).sum())
    XtX_inv = inv_xtx(X, name=caller)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    lam = 0.0
    converged = True
    iters = 0
    WX = weights.W @ X
    Wy = weights.W @ y
    for it in range(int(n_iter)):
        iters = it + 1
        lam_new, _ = _kp_lambda(u, weights, trWtW, bounds)
        Xs = X - lam_new * WX
        ys = y - lam_new * Wy
        XsXs_inv = inv_xtx(Xs, name=f"{caller} (filtered)")
        beta = XsXs_inv @ (Xs.T @ ys)
        u = y - X @ beta
        done = abs(lam_new - lam) < xtol
        lam = lam_new
        if done:
            break
    else:
        converged = int(n_iter) <= 1
    if int(n_iter) > 1 and not converged:
        warnings.warn(
            f"{caller}: the lambda iteration had not converged to {xtol:g} after "
            f"{n_iter} passes (last change kept); converged=False.",
            RuntimeWarning, stacklevel=3,
        )
    Xs = X - lam * WX
    ys = y - lam * Wy
    XsXs_inv = inv_xtx(Xs, name=f"{caller} (filtered)")
    beta = XsXs_inv @ (Xs.T @ ys)
    resid = ys - Xs @ beta
    sigma2 = float(resid @ resid) / n
    if vcov == "robust":
        meat = (Xs.T * (resid ** 2)) @ Xs
        cov_b = XsXs_inv @ meat @ XsXs_inv
    else:
        cov_b = sigma2 * XsXs_inv
    k = X.shape[1]
    cov = np.zeros((k + 1, k + 1))
    cov[:k, :k] = cov_b
    cov[k, k] = np.nan
    return {
        "beta": beta,
        "lam": lam,
        "resid": resid,
        "sigma2": sigma2,
        "vcov": cov,
        "converged": converged,
        "n_iter": iters,
        "Xs": Xs,
    }


# ---------------------------------------------------------------------------
# LeSage-Pace impact scalars
# ---------------------------------------------------------------------------
def _row_power_sums(weights: SpatialWeights, order: int) -> np.ndarray:
    """``m_k = 1' W^k 1`` for ``k = 0 .. order`` by ``order`` sparse matvecs.

    Exact and cheap: ``s(rho) = 1'(I - rho W)^-1 1 = sum_k rho^k m_k`` then
    costs nothing per simulation draw, where a sparse solve would cost one LU
    factorisation each (and a sparse solve is ``O(fill-in)``, not ``O(nnz)`` —
    for a 2-D lattice about ``O(n^1.5)``).
    """
    v = np.ones(weights.n)
    out = [float(v.sum())]
    for _ in range(order):
        v = weights.W @ v
        out.append(float(v.sum()))
    return np.asarray(out)


_MAX_SERIES_ORDER = 5000


def _series_order(radius: float, rho_max: float, tol: float = 1e-16) -> int | None:
    """Truncation order for ``sum rho^k m_k``, or ``None`` to use a sparse solve.

    ``|m_k| <= n * radius^k``, so truncating after ``K`` terms leaves at most
    ``n r^{K+1}/(1 - r)`` with ``r = |rho| radius``. ``None`` is returned when
    the series diverges (``r >= 1``) or would need more than
    ``_MAX_SERIES_ORDER`` terms, in which case the caller falls back to one
    sparse solve per evaluation.
    """
    r = abs(rho_max) * radius
    if r >= 1.0:
        return None
    if r <= 0.0:
        return 2
    order = int(np.ceil(np.log(tol * (1.0 - r)) / np.log(r)))
    if order > _MAX_SERIES_ORDER:
        return None
    return int(max(order, 2))


def _scalar_terms(
    weights: SpatialWeights,
    rho: float,
    *,
    spectrum: np.ndarray,
    m: np.ndarray | None,
) -> tuple[float, float, float, float]:
    """``(tau, t_W, s, s_W)`` — the four scalars the whole decomposition needs.

    ``tau = tr A^-1 = sum_i Re[1/(1 - rho lam_i)]`` and
    ``t_W = tr(A^-1 W) = sum_i Re[lam_i/(1 - rho lam_i)]`` come straight from
    the spectrum; ``s = 1'A^-1 1`` and ``s_W = 1'A^-1 W 1`` from the power
    series when it converges and from one sparse solve otherwise. Writing
    ``t_W`` directly rather than as ``(tau - n)/rho`` avoids the catastrophic
    cancellation at small ``rho`` and removes the need for a ``rho -> 0``
    special case: at ``rho = 0`` the formulas give ``tau = n``, ``t_W = tr W = 0``,
    ``s = n`` and ``s_W = S0`` exactly.
    """
    lam = np.asarray(spectrum)
    denom = 1.0 - rho * lam
    tau = float(np.real(np.sum(1.0 / denom)))
    t_w = float(np.real(np.sum(lam / denom)))
    if m is not None:
        powers = rho ** np.arange(m.shape[0])
        s = float(np.dot(powers[: m.shape[0] - 1], m[:-1]))
        s_w = float(np.dot(powers[: m.shape[0] - 1], m[1:]))
    else:
        n = weights.n
        A = (sp.identity(n, format="csc") - rho * weights.W.tocsc()).tocsc()
        col = spsolve(A, np.ones(n))
        s = float(np.sum(col))
        s_w = float(np.sum(weights.W @ col))
    return tau, t_w, s, s_w


def _resolve_n_sim(n_sim: int | None, n_draws: int | None, caller: str) -> int:
    """One draw count from the two accepted spellings; both given raises.

    ``n_sim`` is canonical here and ``n_draws`` is the alias that matches
    :func:`puremacro.spatial.spatial_panel`. Neither given means ``0``.
    """
    if n_sim is not None and n_draws is not None:
        raise ValueError(
            f"{caller}: pass n_sim= or n_draws=, not both — they are the same argument "
            "(n_sim is the canonical spelling here; n_draws is the spatial_panel alias)."
        )
    if n_draws is not None:
        return int(n_draws)
    return 0 if n_sim is None else int(n_sim)


def _effects_from_params(
    beta: np.ndarray,
    theta: np.ndarray,
    n: int,
    tau: float,
    t_w: float,
    s: float,
    s_w: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(direct, indirect, total)`` from the four scalars, exactly."""
    direct = (beta * tau + theta * t_w) / n
    total = (beta * s + theta * s_w) / n
    return direct, total - direct, total


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SpatialModelResult:
    """Result of :func:`sar`, :func:`sem`, :func:`sdm` or :func:`slx`.

    Attributes
    ----------
    model : str
        ``'sar'``, ``'sem'``, ``'sdm'`` or ``'slx'``.
    method : str
        ``'ml'``, ``'gmm'`` or ``'ols'`` (SLX).
    params, bse, z_values, p_values : pandas.Series
        Coefficients, asymptotic standard errors, ``params / bse`` and
        two-sided standard-normal p-values, indexed by parameter name: the
        exogenous block (``const`` first when added), then the Durbin block as
        ``W_<name>``, then ``rho`` and/or ``lambda``. Every standard error in
        this class is asymptotic, so the statistic is a ``z``, never a ``t``
        (the one exception in the module is :class:`SpatialOLSResult`).
        ``bse['lambda']`` is ``NaN`` for ``method='gmm'``.
    conf_int : pandas.DataFrame
        Columns ``['lower', 'upper']`` at level ``1 - alpha``.
    vcov : ndarray
        ``(p, p)`` covariance in the order given by ``vcov_names``.
    vcov_names : tuple of str
        Row/column labels of ``vcov``, so the ordering never has to be guessed.
        Includes a trailing ``'sigma2'`` for ``method='ml'``.
    beta : ndarray
        ``(k,)`` exogenous coefficients, including the constant.
    theta : ndarray or None
        ``(k_d,)`` Durbin coefficients (SDM and SLX only).
    durbin_names : tuple of str
        The *unlagged* names of the Durbin block, in order.
    rho, lam : float or None
        Spatial lag and spatial error parameters.
    sigma2 : float
        ML: ``e'e/n`` with **no** degrees-of-freedom correction. GMM: ``e'e/n``
        of the filtered residuals. SLX: ``e'e/(n - k)``.
    loglik, aic, bic : float
        Maximised concentrated log-likelihood and the two criteria,
        ``-2 loglik + 2 p`` and ``-2 loglik + p ln n`` with
        ``p = n_params``. ``NaN`` for ``method='gmm'``.
    n_params : int
        Parameters counted by ``aic`` / ``bic``: the mean parameters, plus one
        for ``rho``/``lambda`` where estimated, plus one for ``sigma2``.
    loglik_at_zero : float
        The concentrated log-likelihood evaluated at ``rho = 0`` (or
        ``lambda = 0``) through the *same* objective the optimiser used. Equal
        to the Gaussian OLS log-likelihood ``-(n/2)(ln 2pi + 1) - (n/2) ln(e'e/n)``
        to machine precision — the identity is pinned in the tests. It is not
        read off ``profile_loglik``, whose grid spacing makes interpolation
        accurate only to ``O(1e-3)``.
    pseudo_r2 : float
        Squared correlation between ``y`` and ``fitted``.
    resid : ndarray
        Structural residual — ``(I - rho W) y - Z b`` for the lag models,
        ``(I - lam W)(y - X b)`` for SEM, ``y - Z b`` for SLX. This is the
        vector ``sigma2`` is built from.
    resid_reduced : ndarray
        ``y - fitted``. For a lag model this differs from ``resid``: the
        structural residual and the reduced-form residual are different objects
        and the class reports both rather than letting users subtract and be
        surprised.
    fitted : ndarray
        Reduced-form prediction ``(I - rho W)^-1 Z b`` for the lag models,
        ``X b`` for SEM and ``Z b`` for SLX.
    n, k : int
        Observations and columns of the design actually fitted (including the
        Durbin block).
    exog_names : tuple of str
        Names of the fitted design's columns, in order.
    const_index : int or None
        Position of the constant in ``exog_names``.
    weights_ids : tuple
        ``SpatialWeights.ids``, so ``resid`` / ``fitted`` can be put back onto
        units.
    rho_bounds : tuple of float
        The interval actually searched: the non-singularity interval
        intersected with ``(-1/max|lam_i|, 1/max|lam_i|)`` over the full
        complex spectrum.
    rho_bounds_singularity : tuple of float
        The wider interval on which ``I - rho W`` is merely non-singular. For a
        row-standardised ``W`` the two coincide; for a k-nearest-neighbour or
        economic-flow ``W`` the second can be many units wide, which is why the
        optimiser is never given it.
    logdet_method : str
        ``'eig'``, ``'lu'`` or ``'chebyshev'`` — never ``'auto'``.
    vcov_type : str
        The covariance actually used. One of ``'analytic'`` or ``'hessian'``
        (``method='ml'``), ``'classical'`` or ``'robust'`` (``method='gmm'``),
        or — for :func:`slx`, which is plain OLS and takes ``cov_type`` rather
        than ``vcov`` — the OLS name ``'nonrobust'`` or ``'hc1'``. Downstream
        code must branch over all six; ``'nonrobust'``/``'hc1'`` occur if and
        only if ``model == 'slx'``. The names are deliberately not collapsed
        onto ``'classical'``/``'robust'``: HC1's finite-sample scaling is part
        of what was computed and dropping it from the label would lose it.
    vcov_traces_exact : bool
        Always ``True``. Present so downstream code can assert it: this module
        has no randomised trace estimator.
    converged, n_iter : bool, int
    at_bound : bool
        The optimum sits within ``10 * xtol`` of an endpoint.
    rho_admissible : bool
        Whether the reported spatial parameter lies inside :attr:`rho_bounds`.
        Always ``True`` for ``method='ml'`` (the likelihood is maximised *over*
        that interval) and for ``model='slx'``. The Kelejian-Prucha 2SLS of
        ``method='gmm'`` is an unconstrained linear estimator, so it can — and
        on weakly identified designs does — return a ``rho`` outside it. When
        it does, a ``RuntimeWarning`` fires, this flag is ``False``,
        :meth:`summary` says so, and ``fitted`` / ``resid_reduced`` /
        ``pseudo_r2`` are ``NaN`` rather than computed through the explosive
        multiplier ``(I - rho W)^-1``. :meth:`effects` raises in that state.
    n_islands : int
        Units with no neighbours. They contribute to ``beta`` but carry no
        information about ``rho``.
    common_factor : tuple or None
        ``(Wald statistic, df, p-value)`` of ``H0: theta + rho beta_d = 0``
        (SDM only).
    score_root : float or None
        The ``brentq`` root of the analytic score, kept for the pinned identity.
    spectrum : ndarray or None
        Eigenvalues of ``W`` when they were computed; reused free by
        :meth:`effects`.
    profile_grid, profile_loglik : ndarray
        201 equispaced points across ``rho_bounds`` and the concentrated
        log-likelihood there — for :meth:`plot` only.
    weights : SpatialWeights
    obs_weights : ndarray or None
        The mean-one normalised observation weights actually used.
    alpha : float
    y_name : str
    """

    model: str
    method: str
    params: pd.Series
    bse: pd.Series
    z_values: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    vcov: np.ndarray
    vcov_names: tuple[str, ...]
    beta: np.ndarray
    theta: np.ndarray | None
    durbin_names: tuple[str, ...]
    rho: float | None
    lam: float | None
    sigma2: float
    loglik: float
    aic: float
    bic: float
    n_params: int
    loglik_at_zero: float
    pseudo_r2: float
    resid: np.ndarray
    resid_reduced: np.ndarray
    fitted: np.ndarray
    n: int
    k: int
    exog_names: tuple[str, ...]
    const_index: int | None
    weights_ids: tuple
    rho_bounds: tuple[float, float]
    rho_bounds_singularity: tuple[float, float]
    logdet_method: str
    vcov_type: str
    vcov_traces_exact: bool
    converged: bool
    n_iter: int
    at_bound: bool
    n_islands: int
    common_factor: tuple[float, int, float] | None
    score_root: float | None
    spectrum: np.ndarray | None
    profile_grid: np.ndarray
    profile_loglik: np.ndarray
    weights: SpatialWeights
    obs_weights: np.ndarray | None
    alpha: float
    y_name: str
    rho_admissible: bool = True

    # -- effects ------------------------------------------------------------
    def effects(
        self,
        *,
        n_sim: int | None = None,
        seed: int | None = 0,
        alpha: float | None = None,
        keep_draws: bool = False,
        n_draws: int | None = None,
    ) -> "SpatialEffectsResult":
        """LeSage-Pace direct / indirect / total impacts for this fit.

        Builds the alignment :func:`spatial_effects` needs: one shared name
        list (``exog_names``), ``beta`` over it, ``theta`` over it with zeros
        for the regressors that carry no Durbin term, and a
        ``(2k + 1, 2k + 1)`` covariance scattered from :attr:`vcov` by name.
        The constant is dropped from the reported table by position.

        Parameters
        ----------
        n_sim : int, default 0
            Parameter draws for the simulated standard errors. ``0`` returns
            point estimates with ``NaN`` standard errors.
        seed : int or None, default 0
        alpha : float, optional
            Defaults to the model's own ``alpha``.
        keep_draws : bool, default False
        n_draws : int, optional
            Accepted alias for ``n_sim`` (the spelling
            :func:`puremacro.spatial.spatial_panel` uses); ``n_sim`` is
            canonical here and passing both raises. See
            :func:`spatial_effects` for why the two functions' defaults
            legitimately differ.

        Returns
        -------
        SpatialEffectsResult

        Raises
        ------
        ValueError
            For ``model='sem'``: the spatial error model has no impact
            decomposition — ``dy_i/dx_jr`` is ``beta_r`` on the diagonal and
            zero off it, whatever ``lambda`` is. Also when both ``n_sim=`` and
            ``n_draws=`` are given.
        """
        n_sim = _resolve_n_sim(n_sim, n_draws, "SpatialModelResult.effects")
        if self.model == "sem":
            raise ValueError(
                "SpatialModelResult.effects: the spatial error model has no LeSage-Pace "
                "decomposition (the marginal effect of x_r is beta_r for every unit and "
                "zero across units, whatever lambda is). Fit sar(), sdm() or slx() for "
                "an impact decomposition."
            )
        names = list(self.exog_names)[: self.k - len(self.durbin_names)]
        k = len(names)
        beta = np.asarray(self.beta, dtype=float)
        theta = np.zeros(k)
        pos = {nm: i for i, nm in enumerate(names)}
        if self.theta is not None:
            for j, nm in enumerate(self.durbin_names):
                theta[pos[nm]] = float(self.theta[j])
        big = np.zeros((2 * k + 1, 2 * k + 1))
        vnames = list(self.vcov_names)
        idx: dict[int, int] = {}
        for src, nm in enumerate(vnames):
            if nm in pos:
                idx[src] = pos[nm]
            elif nm.startswith("W_") and nm[2:] in pos:
                idx[src] = k + pos[nm[2:]]
            elif nm == "rho":
                idx[src] = 2 * k
        for a, ia in idx.items():
            for b, ib in idx.items():
                big[ia, ib] = self.vcov[a, b]
        return spatial_effects(
            self.weights,
            beta,
            0.0 if self.rho is None else float(self.rho),
            theta=theta,
            vcov=big,
            names=names,
            const_index=self.const_index,
            n_sim=n_sim,
            seed=seed,
            alpha=self.alpha if alpha is None else alpha,
            keep_draws=keep_draws,
            spectrum=self.spectrum,
        )

    # -- presentation -------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Coefficient table with a plain ``RangeIndex``.

        Columns ``['parameter', 'coefficient', 'std_error', 'z', 'p_value',
        'ci_lower', 'ci_upper']``; ``sigma2`` is appended as a final row with
        ``NaN`` in the inference columns.
        """
        df = pd.DataFrame({
            "parameter": list(self.params.index),
            "coefficient": self.params.to_numpy(dtype=float),
            "std_error": self.bse.to_numpy(dtype=float),
            "z": self.z_values.to_numpy(dtype=float),
            "p_value": self.p_values.to_numpy(dtype=float),
            "ci_lower": self.conf_int["lower"].to_numpy(dtype=float),
            "ci_upper": self.conf_int["upper"].to_numpy(dtype=float),
        })
        tail = pd.DataFrame({
            "parameter": ["sigma2"], "coefficient": [self.sigma2],
            "std_error": [np.nan], "z": [np.nan], "p_value": [np.nan],
            "ci_lower": [np.nan], "ci_upper": [np.nan],
        })
        return pd.concat([df, tail], ignore_index=True)

    def summary(self) -> str:
        """Human-readable report; ``print(result.summary())``."""
        label = {"sar": "Spatial lag (SAR)", "sem": "Spatial error (SEM)",
                 "sdm": "Spatial Durbin (SDM)", "slx": "Spatially lagged X (SLX)"}[self.model]
        est = {"ml": "Gaussian ML (concentrated)", "gmm": "Kelejian-Prucha GMM",
               "ols": "OLS"}[self.method]
        lines = [
            f"{label} — {est}",
            f"  n = {self.n}   k = {self.k}   log-determinant route: {self.logdet_method}",
            "",
            f"{'parameter':<18}{'coef':>12}{'std err':>11}{'z':>9}{'P>|z|':>9}"
            f"{'[' + format(1 - self.alpha, '.0%'):>12}{'CI]':>11}",
        ]
        for nm in self.params.index:
            lines.append(
                f"{str(nm):<18}{self.params[nm]:>12.4f}{self.bse[nm]:>11.4f}"
                f"{self.z_values[nm]:>9.3f}{self.p_values[nm]:>9.4f}"
                f"{self.conf_int.loc[nm, 'lower']:>12.4f}{self.conf_int.loc[nm, 'upper']:>11.4f}"
            )
        lines.append("")
        lines.append(f"  sigma^2 = {self.sigma2:.6f}   pseudo-R^2 = {self.pseudo_r2:.4f}")
        if np.isfinite(self.loglik):
            lines.append(
                f"  log-likelihood = {self.loglik:.4f}   AIC = {self.aic:.3f}   "
                f"BIC = {self.bic:.3f}   (p = {self.n_params})"
            )
        sp_name, sp_val = ("rho", self.rho) if self.rho is not None else ("lambda", self.lam)
        if sp_val is not None:
            if self.method == "gmm" and self.model in ("sar", "sdm"):
                # 2SLS never searched an interval, so do not say it did, and do
                # not assert non-singularity on a region the estimate may be
                # outside of (build rule G5).
                verdict = "inside" if self.rho_admissible else "OUTSIDE"
                lines.append(
                    f"  {sp_name} = {sp_val:+.6f} (Kelejian-Prucha 2SLS: unconstrained, no "
                    f"interval search); {verdict} the admissible interval "
                    f"({self.rho_bounds[0]:+.4f}, {self.rho_bounds[1]:+.4f})"
                )
                if not self.rho_admissible:
                    lines.append(
                        "  WARNING: rho is outside the admissible interval, so "
                        f"(I - {sp_name} W)^-1 has no convergent spatial-multiplier "
                        "expansion. fitted / resid_reduced / pseudo-R^2 are NaN and "
                        "effects() will raise. Refit with method='ml'."
                    )
            else:
                lines.append(
                    f"  {sp_name} = {sp_val:+.6f} searched over "
                    f"({self.rho_bounds[0]:+.4f}, {self.rho_bounds[1]:+.4f}); "
                    f"I - {sp_name} W is non-singular on "
                    f"({self.rho_bounds_singularity[0]:+.4f}, "
                    f"{self.rho_bounds_singularity[1]:+.4f})"
                )
            if self.at_bound:
                lines.append("  WARNING: the optimum sits on a bound; inference on it is not valid.")
        if self.common_factor is not None:
            stat, df_, pv = self.common_factor
            verdict = "SEM is not rejected" if pv > 0.05 else "SEM is rejected"
            lines.append(
                f"  Common factor (H0: theta + rho*beta = 0): W = {stat:.4f}, df = {df_}, "
                f"p = {pv:.4f} -> {verdict}"
            )
            if self.theta is not None and self.rho is not None:
                bd = np.array([self.beta[list(self.exog_names).index(nm)] for nm in self.durbin_names])
                if np.any(np.sign(self.theta) == np.sign(self.rho * bd)):
                    lines.append(
                        "    (caveat: the restriction is only meaningful where "
                        "sign(theta_r) = -sign(rho*beta_r); it does not hold for every r here)"
                    )
        if self.n_islands:
            lines.append(
                f"  {self.n_islands} island(s) with no neighbours: they enter beta but carry "
                "no information about the spatial parameter, and the closed-form total "
                "effect beta/(1 - rho) does not apply."
            )
        lines.append("")
        if self.method == "ml":
            lines.append(
                "  Standard errors: analytic information matrix. Asymptotically valid under\n"
                "  correct specification with Gaussian homoskedastic errors. Under non-normality\n"
                "  the beta block stays valid but the rho and sigma^2 blocks need the Lee (2004)\n"
                "  QMLE corrections; under heteroskedasticity of unknown form the ML estimator of\n"
                "  rho is itself inconsistent (Lin & Lee 2010) — refit with method='gmm',\n"
                "  vcov='robust'."
                if self.vcov_type == "analytic" else
                "  Standard errors: observed information (numerical Hessian of the full\n"
                "  log-likelihood, differentiated in ln sigma^2). Same asymptotic object as the\n"
                "  analytic matrix, differing by O_p(n^-1/2)."
            )
        elif self.method == "gmm":
            lines.append(
                f"  Standard errors: Kelejian-Prucha two-stage least squares, {self.vcov_type}.\n"
                "  The heteroskedasticity-robust GMM of Kelejian & Prucha (2010) is not\n"
                "  implemented; vcov='robust' is a White sandwich around the homoskedastic\n"
                "  moment conditions."
                + ("\n  No standard error is reported for lambda: it needs the KP (2010) Psi matrices."
                   if self.model == "sem" else "")
            )
        else:
            lines.append(
                "  SLX is ordinary least squares on [X, W X_d]: sigma^2 is dof-corrected and the\n"
                "  reported statistic is still an asymptotic z. Use ols_spatial() for exact t\n"
                "  inference on a design with no spatial lag."
            )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, axes=None, figsize: tuple[float, float] = (11.0, 4.2)):
        """Profile log-likelihood and a coefficient forest plot.

        The left panel is the single most informative diagnostic for a spatial
        model: a flat or boundary-peaked profile is exactly the failure mode
        users hit. The 95% likelihood-ratio interval is the region above
        ``max - 1.92``.
        """
        import matplotlib.pyplot as plt

        fig = None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax0, ax1 = axes[0], axes[1]
        sp_val = self.rho if self.rho is not None else self.lam
        if self.profile_grid.size and np.any(np.isfinite(self.profile_loglik)):
            ax0.plot(self.profile_grid, self.profile_loglik, color="steelblue")
            top = float(np.nanmax(self.profile_loglik))
            ax0.axhline(top - 1.92, color="grey", linestyle=":", linewidth=1.0,
                        label="max - 1.92 (95% LR)")
            if sp_val is not None:
                ax0.axvline(sp_val, color="firebrick", linewidth=1.3,
                            label=f"estimate = {sp_val:+.4f}")
            ax0.axvspan(self.rho_bounds[0], self.rho_bounds[1], color="lightgrey", alpha=0.25)
            ax0.legend(frameon=False, fontsize=8)
        ax0.set_xlabel("rho" if self.model != "sem" else "lambda")
        ax0.set_ylabel("concentrated log-likelihood")
        ax0.set_title("Profile log-likelihood")
        idx = list(self.params.index)
        ypos = np.arange(len(idx))[::-1]
        colours = ["firebrick" if nm in ("rho", "lambda") else "steelblue" for nm in idx]
        for yv, nm, col in zip(ypos, idx, colours):
            ax1.plot([self.conf_int.loc[nm, "lower"], self.conf_int.loc[nm, "upper"]],
                     [yv, yv], color=col, linewidth=1.4)
            ax1.plot([self.params[nm]], [yv], "o", color=col, markersize=5)
        ax1.axvline(0.0, color="grey", linewidth=0.8)
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(idx, fontsize=8)
        ax1.set_xlabel("coefficient")
        ax1.set_title(f"{self.model.upper()} coefficients ({1 - self.alpha:.0%} CI)")
        if fig is not None:
            fig.tight_layout()
            return fig
        return ax0.get_figure()


@dataclass(frozen=True, eq=False)
class SpatialEffectsResult:
    """Result of :func:`spatial_effects` — the LeSage-Pace impact decomposition.

    Attributes
    ----------
    variables : tuple of str
        The regressors whose impacts are reported. The constant is excluded:
        its "effect" is not a marginal effect.
    direct, indirect, total : ndarray
        ``(k_x,)`` point estimates. ``direct_r = (beta_r tau + theta_r t_W)/n``,
        ``total_r = (beta_r s + theta_r s_W)/n``, ``indirect = total - direct``.
    direct_se, indirect_se, total_se : ndarray
        Simulation standard deviations; all ``NaN`` when ``n_sim == 0`` or no
        covariance was supplied.
    direct_ci, indirect_ci, total_ci : ndarray
        ``(k_x, 2)`` empirical simulation quantiles at ``alpha/2`` and
        ``1 - alpha/2`` (``NaN`` without a simulation).
    direct_p, indirect_p, total_p : ndarray
        **Simulation** two-sided p-values,
        ``2 min(share of draws > 0, share < 0)``, floored at ``1/n_kept``.
        Deliberately not the normal approximation ``2 Phi(-|effect|/se)``: with
        ``rho`` near a bound the draw distribution is skewed and a normal
        p-value can contradict the quantile interval printed next to it.
    rho : float
    has_durbin : bool
    tau, t_w, s, s_w : float
        ``tr A^-1``, ``tr(A^-1 W)``, ``1'A^-1 1`` and ``1'A^-1 W 1``.
    s_solve_check : float
        ``|s - 1' spsolve(A, 1)|`` — the power series cross-checked once
        against a sparse solve.
    trace_method : str
        ``'spectrum+series'`` or ``'spectrum+solve'`` (the latter when the
        Neumann series does not converge at this ``rho``).
    n, n_sim, n_kept, n_dropped : int
    seed : int or None
    alpha : float
    draws : ndarray or None
        ``(n_kept, k_x, 3)`` simulated ``(direct, indirect, total)`` when
        ``keep_draws=True``.
    """

    variables: tuple[str, ...]
    direct: np.ndarray
    indirect: np.ndarray
    total: np.ndarray
    direct_se: np.ndarray
    indirect_se: np.ndarray
    total_se: np.ndarray
    direct_ci: np.ndarray
    indirect_ci: np.ndarray
    total_ci: np.ndarray
    direct_p: np.ndarray
    indirect_p: np.ndarray
    total_p: np.ndarray
    rho: float
    has_durbin: bool
    tau: float
    t_w: float
    s: float
    s_w: float
    s_solve_check: float
    trace_method: str
    n: int
    n_sim: int
    n_kept: int
    n_dropped: int
    seed: int | None
    alpha: float
    draws: np.ndarray | None

    def to_frame(self) -> pd.DataFrame:
        """One row per variable, sixteen columns, plain ``RangeIndex``."""
        return pd.DataFrame({
            "variable": list(self.variables),
            "direct": self.direct, "direct_se": self.direct_se, "direct_p": self.direct_p,
            "direct_lo": self.direct_ci[:, 0], "direct_hi": self.direct_ci[:, 1],
            "indirect": self.indirect, "indirect_se": self.indirect_se,
            "indirect_p": self.indirect_p,
            "indirect_lo": self.indirect_ci[:, 0], "indirect_hi": self.indirect_ci[:, 1],
            "total": self.total, "total_se": self.total_se, "total_p": self.total_p,
            "total_lo": self.total_ci[:, 0], "total_hi": self.total_ci[:, 1],
        })

    def summary(self) -> str:
        sim = (f"{self.n_sim} simulations, {self.n_kept} kept, {self.n_dropped} discarded"
               if self.n_sim else "point estimates only (n_sim = 0; standard errors are NaN)")
        lines = [
            f"LeSage-Pace impact decomposition (rho = {self.rho:+.6f}, n = {self.n}, {sim})",
            "",
        ]
        for i, v in enumerate(self.variables):
            lines.append(
                f"  {str(v):<16} direct {self.direct[i]:+.4f} ({self.direct_se[i]:.4f})"
                f"   indirect {self.indirect[i]:+.4f} ({self.indirect_se[i]:.4f})"
                f"   total {self.total[i]:+.4f} ({self.total_se[i]:.4f})"
            )
        lines += [
            "",
            f"  tr(A^-1)/n = {self.tau / self.n:.6f}   1'A^-1 1/n = {self.s / self.n:.6f}"
            f"   ({self.trace_method}, series-vs-solve gap {self.s_solve_check:.2e})",
            "",
            "  An indirect effect is the SUM of all n(n-1) off-diagonal cross-partials",
            "  divided by n — the average total spillover a unit sends (equivalently,",
            "  receives) — not the effect on any one neighbour, and not the mean",
            "  individual cross-partial (which is smaller by a factor of n-1).",
            "  The standard errors are a normal approximation to the *parameter*",
            "  distribution propagated through the impact formula — not a bootstrap of",
            "  the data — and the p-values are the simulation quantile p-values,",
            "  consistent with the intervals printed beside them.",
        ]
        if self.has_durbin:
            lines.append(
                "  NOTE: the Durbin term theta_r tr(A^-1 W)/n is kept in the DIRECT effect.\n"
                "  spreg's spat_impacts='full' assigns every SLX effect to the indirect column\n"
                "  instead, so the direct/indirect split of an SDM differs from spreg's by that\n"
                "  term (the total is the same). This module follows LeSage & Pace (2009)."
            )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, ax=None, figsize: tuple[float, float] = (7.5, 4.5)):
        """Grouped horizontal bars, one group per variable.

        Error bars are the asymmetric simulation quantiles, not
        ``+/- 1.96 se``; with ``n_sim == 0`` they are omitted rather than drawn
        at zero.
        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        k = len(self.variables)
        base = np.arange(k)
        height = 0.26
        specs = [("direct", self.direct, self.direct_ci, "steelblue"),
                 ("indirect", self.indirect, self.indirect_ci, "darkorange"),
                 ("total", self.total, self.total_ci, "seagreen")]
        for off, (label, vals, ci, colour) in zip((height, 0.0, -height), specs):
            err = None
            if self.n_sim and np.all(np.isfinite(ci)):
                err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
                err = np.clip(err, 0.0, None)
            ax.barh(base + off, vals, height=height * 0.9, color=colour, label=label,
                    xerr=err, error_kw={"linewidth": 1.0, "ecolor": "black"})
        ax.axvline(0.0, color="grey", linewidth=0.8)
        ax.set_yticks(base)
        ax.set_yticklabels(list(self.variables), fontsize=9)
        ax.set_xlabel("impact")
        ax.set_title(f"Direct / indirect / total impacts (rho = {self.rho:+.3f})")
        ax.legend(frameon=False, fontsize=8)
        if fig is not None:
            fig.tight_layout()
            return fig
        return ax.get_figure()


@dataclass(frozen=True, eq=False)
class SpatialLMResult:
    """Result of :func:`lm_spatial_tests` — the ABFY (1996) specification battery.

    Attributes
    ----------
    lm_lag, p_lag : float
        LM test for an omitted spatial lag, ``chi2(1)``.
    lm_error, p_error : float
        LM test for a spatial error, ``chi2(1)``.
    rlm_lag, p_rlm_lag : float
        Robust to a locally misspecified error process.
    rlm_error, p_rlm_error : float
        Robust to a locally misspecified lag.
    lm_sarma, p_sarma : float
        Joint test, ``chi2(2)``.
    moran_i, moran_expected, moran_variance, moran_z, moran_p : float
        Cliff-Ord (1972) residual Moran's I with the ``X``-dependent moments.
        These differ from :func:`puremacro.spatial.morans_i`, which uses the
        raw-variable moments — OLS residuals are not the raw variable.
    recommendation : str
    n, k : int
    sigma2 : float
        ``e'e/n``. The LM statistics use this ML variance, as the Gaussian
        likelihood derivation requires. Substituting the dof-corrected
        ``e'e/(n - k)`` would rescale RLM-lag by ``(n - k)/n`` and LM-error by
        ``((n - k)/n)^2``, and would rescale LM-lag, RLM-error and LM-SARMA by
        no constant at all, because ``nJ = [(WXb)'M(WXb) + T sigma^2]/sigma^2``
        is *affine* in ``1/sigma^2``, not linear.
    trace_T : float
        ``tr(W W + W' W)``.
    info_lag : float
        The ``nJ`` scalar. Reported because a near-zero ``nJ - T`` is exactly
        what breaks the robust LM-lag.
    moran_traces_exact : bool
        Always ``True``: the traces are formed from the dense ``M W M``.
    resid : ndarray
        The OLS residuals the battery was computed on.
    """

    lm_lag: float
    p_lag: float
    lm_error: float
    p_error: float
    rlm_lag: float
    p_rlm_lag: float
    rlm_error: float
    p_rlm_error: float
    lm_sarma: float
    p_sarma: float
    moran_i: float
    moran_expected: float
    moran_variance: float
    moran_z: float
    moran_p: float
    recommendation: str
    n: int
    k: int
    sigma2: float
    trace_T: float
    info_lag: float
    moran_traces_exact: bool
    resid: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("LM-lag", self.lm_lag, 1.0, self.p_lag),
            ("LM-error", self.lm_error, 1.0, self.p_error),
            ("robust LM-lag", self.rlm_lag, 1.0, self.p_rlm_lag),
            ("robust LM-error", self.rlm_error, 1.0, self.p_rlm_error),
            ("LM-SARMA", self.lm_sarma, 2.0, self.p_sarma),
            ("Moran's I (residuals)", self.moran_i, float("nan"), self.moran_p),
        ]
        return pd.DataFrame({
            "test": [r[0] for r in rows],
            "statistic": [r[1] for r in rows],
            "df": [r[2] for r in rows],
            "p_value": [r[3] for r in rows],
        })

    def summary(self) -> str:
        lines = [
            f"Spatial specification tests on OLS residuals (n = {self.n}, k = {self.k})",
            "",
            f"{'test':<24}{'statistic':>12}{'df':>5}{'p-value':>10}",
        ]
        for _, r in self.to_frame().iterrows():
            df_ = "" if not np.isfinite(r["df"]) else f"{int(r['df'])}"
            lines.append(f"{r['test']:<24}{r['statistic']:>12.6f}{df_:>5}{r['p_value']:>10.4f}")
        lines += [
            "",
            f"  Moran's I (residuals) = {self.moran_i:.4f}, E[I] = {self.moran_expected:+.4f}, "
            f"z = {self.moran_z:+.3f}, p = {self.moran_p:.4f}",
            f"  tr(WW + W'W) = {self.trace_T:.4f}   lag information nJ = {self.info_lag:.4f}",
            "",
            f"  Decision: {self.recommendation}",
            "",
            "  The LM battery is a LOCAL test computed entirely under the null of no spatial",
            "  dependence: it says which alternative the OLS residuals point at, not that the",
            "  indicated model is correct. A rejection of BOTH robust statistics points at a",
            "  model with two spatial processes (SDM or SARAR), not at either simple one.",
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, ax=None, figsize: tuple[float, float] = (6.5, 3.8)):
        """Bar chart of the five LM statistics against their critical values."""
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        labels = ["LM-lag", "LM-error", "robust LM-lag", "robust LM-error", "LM-SARMA"]
        stats_ = [self.lm_lag, self.lm_error, self.rlm_lag, self.rlm_error, self.lm_sarma]
        crit = [3.841458820694124] * 4 + [5.991464547107979]
        ypos = np.arange(len(labels))[::-1]
        for yv, s, c in zip(ypos, stats_, crit):
            filled = s > c
            ax.barh(yv, s, height=0.6, color="steelblue" if filled else "white",
                    edgecolor="steelblue")
        ax.axvline(3.841458820694124, color="firebrick", linewidth=1.0,
                   label="chi2(1) 95% = 3.841")
        ax.axvline(5.991464547107979, color="firebrick", linestyle="--", linewidth=1.0,
                   label="chi2(2) 95% = 5.991")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("statistic")
        ax.set_title("Spatial LM battery")
        ax.legend(frameon=False, fontsize=8)
        if fig is not None:
            fig.tight_layout()
            return fig
        return ax.get_figure()


@dataclass(frozen=True, eq=False)
class SpatialOLSResult:
    """Result of :func:`ols_spatial` — OLS plus the full spatial battery.

    This is the one result object in the module whose inference is *not*
    asymptotic-only: OLS with Gaussian errors gives exact ``t`` statistics on
    ``n - k`` degrees of freedom, and ``sigma2`` is dof-corrected. The spatial
    models report ``z``.

    Attributes
    ----------
    params, bse, t_values, p_values : pandas.Series
    conf_int : pandas.DataFrame
        ``['lower', 'upper']`` from the Student-``t`` quantile.
    vcov : ndarray
    sigma2 : float
        ``e'e/(n - k)``.
    resid, fitted : ndarray
    r2, r2_adj, loglik, aic, bic, f_stat, f_pvalue : float
        ``loglik`` uses the ML variance ``e'e/n``.
    n, k : int
    exog_names : tuple of str
    weights_ids : tuple
    cov_type : str
        ``'nonrobust'`` or ``'hc1'``.
    lm : SpatialLMResult
        The battery on this fit.
    weights : SpatialWeights
    alpha : float
    """

    params: pd.Series
    bse: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    vcov: np.ndarray
    sigma2: float
    resid: np.ndarray
    fitted: np.ndarray
    r2: float
    r2_adj: float
    loglik: float
    aic: float
    bic: float
    f_stat: float
    f_pvalue: float
    n: int
    k: int
    exog_names: tuple[str, ...]
    weights_ids: tuple
    cov_type: str
    lm: SpatialLMResult
    weights: SpatialWeights
    alpha: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": list(self.params.index),
            "coefficient": self.params.to_numpy(dtype=float),
            "std_error": self.bse.to_numpy(dtype=float),
            "t": self.t_values.to_numpy(dtype=float),
            "p_value": self.p_values.to_numpy(dtype=float),
            "ci_lower": self.conf_int["lower"].to_numpy(dtype=float),
            "ci_upper": self.conf_int["upper"].to_numpy(dtype=float),
        })

    def to_frame_lm(self) -> pd.DataFrame:
        """The LM battery table, for exporting that instead."""
        return self.lm.to_frame()

    def summary(self) -> str:
        lines = [
            f"OLS with spatial diagnostics — n = {self.n}, k = {self.k}, "
            f"cov_type = {self.cov_type}",
            "",
            f"{'parameter':<18}{'coef':>12}{'std err':>11}{'t':>9}{'P>|t|':>9}"
            f"{'[' + format(1 - self.alpha, '.0%'):>12}{'CI]':>11}",
        ]
        for nm in self.params.index:
            lines.append(
                f"{str(nm):<18}{self.params[nm]:>12.4f}{self.bse[nm]:>11.4f}"
                f"{self.t_values[nm]:>9.3f}{self.p_values[nm]:>9.4f}"
                f"{self.conf_int.loc[nm, 'lower']:>12.4f}{self.conf_int.loc[nm, 'upper']:>11.4f}"
            )
        lines += [
            "",
            f"  R^2 = {self.r2:.4f}   adj. R^2 = {self.r2_adj:.4f}   "
            f"sigma^2 = {self.sigma2:.6f}  (dof-corrected)",
            f"  log-likelihood = {self.loglik:.4f}   AIC = {self.aic:.3f}   BIC = {self.bic:.3f}",
            f"  F({self.k - 1}, {self.n - self.k}) = {self.f_stat:.4f}, p = {self.f_pvalue:.4f}",
            "",
        ]
        lines.extend("  " + ln for ln in self.lm.summary().split("\n"))
        nxt = {
            "ols": "no spatial model is indicated; ols_spatial() is the fit",
            "sar": "sar(y, X, W)",
            "sem": "sem(y, X, W)",
            "sdm": "sdm(y, X, W)  (or slx(y, X, W) if only the lagged X matter)",
        }.get(self.lm.recommendation.split()[0], "inspect the battery above by hand")
        lines.append(f"  Indicated next step: {nxt}")
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, axes=None, figsize: tuple[float, float] = (11.0, 4.2)):
        """Residual Moran scatterplot and the LM bar chart."""
        import matplotlib.pyplot as plt

        fig = None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax0, ax1 = axes[0], axes[1]
        e = self.resid
        le = self.weights.W @ e
        ax0.scatter(e, le, s=18, color="steelblue", alpha=0.8)
        denom = float(e @ e)
        slope = float(e @ le) / denom if denom > 0 else 0.0
        grid = np.linspace(float(e.min()), float(e.max()), 50)
        ax0.plot(grid, slope * grid, color="firebrick", linewidth=1.4,
                 label=f"slope = {slope:+.3f}")
        ax0.axhline(0.0, color="grey", linewidth=0.8)
        ax0.axvline(0.0, color="grey", linewidth=0.8)
        ax0.set_xlabel("OLS residual")
        ax0.set_ylabel("spatial lag of the residual")
        ax0.set_title(f"Residual Moran scatter, I = {self.lm.moran_i:+.3f}")
        ax0.legend(frameon=False, fontsize=8)
        self.lm.plot(ax=ax1)
        if fig is not None:
            fig.tight_layout()
            return fig
        return ax0.get_figure()


# ---------------------------------------------------------------------------
# The specification battery
# ---------------------------------------------------------------------------
def _projector(X: np.ndarray, caller: str) -> np.ndarray:
    n = X.shape[0]
    if n > 3000:
        warnings.warn(
            f"{caller}: forming the dense {n} x {n} OLS projector M for the residual Moran "
            "traces (O(n^2) memory). This is exact by design; there is no probe-based "
            "alternative in this module.",
            RuntimeWarning, stacklevel=3,
        )
    XtX_inv = inv_xtx(X, name=caller)
    return np.eye(n) - X @ XtX_inv @ X.T


def lm_spatial_tests(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
) -> SpatialLMResult:
    """Anselin-Bera-Florax-Yoon (1996) LM battery on the OLS residuals.

    Five Lagrange-multiplier statistics — LM-lag, LM-error, their robust
    versions and the joint LM-SARMA — plus the Cliff-Ord (1972) residual
    Moran's I. This is the standard way to choose between SAR and SEM *before*
    fitting either.

    Everything is computed under the null of no spatial dependence, from an OLS
    fit with the **ML** variance ``sigma2 = e'e/n``, as the Gaussian likelihood
    derivation requires. Substituting the dof-corrected ``e'e/(n - k)`` would
    rescale RLM-lag by ``(n - k)/n`` and LM-error by ``((n - k)/n)^2``, and
    would rescale LM-lag, RLM-error and LM-SARMA by no constant at all, because
    ``nJ = [(WXb)'M(WXb) + T sigma^2]/sigma^2`` is *affine* in ``1/sigma^2``,
    not linear.

    Parameters
    ----------
    y : array_like, pandas.Series or str
        Dependent variable, or a column name when ``data`` is given.
    X : array_like, pandas.DataFrame or sequence of str
        Regressors, or column names when ``data`` is given.
    weights : SpatialWeights
    data : pandas.DataFrame, optional
        When given, ``y`` and ``X`` must be column names; the frame is aligned
        onto ``weights.ids`` by label.
    add_constant : bool, default True
        Prepend a column of ones named ``const``. Raises if ``X`` already
        carries a constant.
    x_names : sequence of str, optional
        Override the regressor names.

    Returns
    -------
    SpatialLMResult

    Raises
    ------
    TypeError
        ``weights`` is not a :class:`SpatialWeights`.
    ValueError
        The weights have no neighbours (``S0 == 0``); ``y``/``X`` contain NaN
        or inf; ``n <= k + 2``; ``y`` is constant or is fitted exactly by ``X``,
        leaving no residual variation; ``X`` already has a constant with
        ``add_constant=True``; the robust LM-lag denominator ``nJ - T``
        vanishes because ``W X b`` lies in the column space of ``X`` (e.g. a
        single regressor that is an eigenvector of ``W``) — the message says so
        and points at the non-robust statistics.
    numpy.linalg.LinAlgError
        ``X'X`` is singular; the message names the collinear columns.

    Notes
    -----
    With ``d_rho = e'Wy/sigma2``, ``d_lam = e'We/sigma2``,
    ``T = tr(WW + W'W)``, ``M = I - X(X'X)^-1 X'`` and
    ``nJ = [(W X b)' M (W X b) + T sigma2]/sigma2``:

    ``LM_lag = d_rho^2 / nJ``, ``LM_err = d_lam^2 / T``,
    ``RLM_lag = (d_rho - d_lam)^2/(nJ - T)``,
    ``RLM_err = (d_lam - T d_rho/nJ)^2 / [T(1 - T/nJ)]``,
    ``LM_SARMA = LM_lag + RLM_err``.

    The residual Moran's I uses the Cliff-Ord moments, which need
    ``G = M W M`` (not ``M W``): ``tr(GG) = tr(MWMW)`` and
    ``tr(GG') = tr(MWMW')``, and ``tr(MW MW') != tr(MW W'M)``.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import lm_spatial_tests
    >>> nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < 25] for i in range(25)}
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(25, 2))
    >>> y = 1.0 + X @ [0.5, -0.3] + rng.normal(scale=0.5, size=25)
    >>> res = lm_spatial_tests(y, X, W)
    >>> res.recommendation
    'ols — neither LM statistic rejects at 5%'

    References
    ----------
    Anselin, Bera, Florax & Yoon (1996); Cliff & Ord (1972).
    """
    caller = "lm_spatial_tests"
    W = _check_weights(weights, caller)
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant,
                          x_names=x_names, caller=caller)
    if W.W.nnz == 0:
        raise ValueError(
            f"{caller}: the weights have no neighbours (every unit is an island), so no "
            "spatial alternative is identified."
        )
    yv, Xm = des.y, des.X
    n, k = des.n, des.k
    XtX_inv = inv_xtx(Xm, name=caller)
    b = XtX_inv @ (Xm.T @ yv)
    e = yv - Xm @ b
    sigma2 = float(e @ e) / n
    Wm = W.W
    Wy = Wm @ yv
    We = Wm @ e
    d_rho = float(e @ Wy) / sigma2
    d_lam = float(e @ We) / sigma2
    T = float(Wm.multiply(Wm).sum() + Wm.multiply(Wm.T).sum())
    WXb = Wm @ (Xm @ b)
    proj = Xm @ (XtX_inv @ (Xm.T @ WXb))
    quad = float(WXb @ WXb) - float(proj @ proj)
    nJ = (quad + T * sigma2) / sigma2
    if nJ - T <= 1e-12 * max(nJ, 1.0):
        raise ValueError(
            f"{caller}: the robust LM-lag denominator nJ - T = {nJ - T:.3e} has vanished, "
            "which happens when W X b lies in the column space of X (for instance a single "
            "regressor that is an eigenvector of W). The robust statistics are undefined "
            "here; use the non-robust LM-lag and LM-error, or add a regressor."
        )
    lm_lag = d_rho ** 2 / nJ
    lm_err = d_lam ** 2 / T
    rlm_lag = (d_rho - d_lam) ** 2 / (nJ - T)
    rlm_err = (d_lam - T * d_rho / nJ) ** 2 / (T * (1.0 - T / nJ))
    lm_sarma = lm_lag + rlm_err
    p = [float(stats.chi2.sf(s, 1)) for s in (lm_lag, lm_err, rlm_lag, rlm_err)]
    p_sarma = float(stats.chi2.sf(lm_sarma, 2))

    # -- Cliff-Ord residual Moran's I ---------------------------------------
    S0 = W.s0
    M = _projector(Xm, caller)
    Wd = Wm.toarray()
    G = M @ Wd @ M
    tr_mw = float(np.trace(G))
    tr_gg = float(np.sum(G * G.T))
    tr_ggt = float(np.sum(G * G))
    I = (n / S0) * float(e @ We) / float(e @ e)
    EI = (n / S0) * tr_mw / (n - k)
    VI = ((n / S0) ** 2) * (tr_ggt + tr_gg + tr_mw ** 2) / ((n - k) * (n - k + 2.0)) - EI ** 2
    zI = (I - EI) / np.sqrt(VI) if VI > 0 else float("nan")
    pI = float(2.0 * stats.norm.sf(abs(zI))) if np.isfinite(zI) else float("nan")

    sig_lag, sig_err = p[0] < 0.05, p[1] < 0.05
    if not sig_lag and not sig_err:
        rec = "ols — neither LM statistic rejects at 5%"
    elif sig_lag and not sig_err:
        rec = "sar — only the LM-lag rejects"
    elif sig_err and not sig_lag:
        rec = "sem — only the LM-error rejects"
    else:
        rl, re_ = p[2] < 0.05, p[3] < 0.05
        if rl and not re_:
            rec = "sar — both LM statistics reject; only the robust LM-lag does"
        elif re_ and not rl:
            rec = "sem — both LM statistics reject; only the robust LM-error does"
        elif rl and re_:
            rec = "sdm or sarar (both robust LM reject)"
        else:
            rec = "inconclusive (neither robust LM rejects)"
    return SpatialLMResult(
        lm_lag=float(lm_lag), p_lag=p[0], lm_error=float(lm_err), p_error=p[1],
        rlm_lag=float(rlm_lag), p_rlm_lag=p[2], rlm_error=float(rlm_err), p_rlm_error=p[3],
        lm_sarma=float(lm_sarma), p_sarma=p_sarma,
        moran_i=float(I), moran_expected=float(EI), moran_variance=float(VI),
        moran_z=float(zI), moran_p=pI, recommendation=rec,
        n=n, k=k, sigma2=sigma2, trace_T=T, info_lag=float(nJ),
        moran_traces_exact=True, resid=e,
    )


def ols_spatial(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    cov_type: str = "nonrobust",
    alpha: float = 0.05,
) -> SpatialOLSResult:
    """OLS with the spatial specification battery attached.

    The entry point to call **first**, before choosing between
    :func:`sar`, :func:`sem`, :func:`sdm` and :func:`slx`: the LM battery needs
    the OLS fit, the projector ``M`` and the raw ``y``, and assembling those by
    hand is the difference between a toolkit and a pile of formulas.

    Parameters
    ----------
    y, X, weights, data, add_constant, x_names
        As in :func:`lm_spatial_tests`.
    cov_type : {'nonrobust', 'hc1'}, default 'nonrobust'
        ``'nonrobust'`` is ``sigma2 (X'X)^-1`` with ``sigma2 = e'e/(n - k)``;
        ``'hc1'`` is the ``n/(n - k)``-scaled White sandwich. A Conley
        spatial-HAC covariance is deliberately not offered here — see the
        module's "Deliberately omitted" section.
    alpha : float, default 0.05

    Returns
    -------
    SpatialOLSResult

    Raises
    ------
    ValueError
        ``cov_type`` outside its enumeration, plus everything
        :func:`lm_spatial_tests` raises.

    Notes
    -----
    ``sigma2`` is dof-corrected (``e'e/(n - k)``) for the reported standard
    errors, while ``loglik``, ``aic`` and ``bic`` use the ML variance
    ``e'e/n`` — and so does the LM battery. The two scalings live side by side
    on purpose; the field docstrings say which is which.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import ols_spatial
    >>> nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < 30] for i in range(30)}
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(30, 2))
    >>> y = 2.0 + X @ [1.0, -0.5] + rng.normal(scale=0.4, size=30)
    >>> res = ols_spatial(y, X, W)
    >>> round(float(res.params['const']), 2)
    1.95
    >>> res.lm.recommendation
    'ols — neither LM statistic rejects at 5%'
    """
    caller = "ols_spatial"
    W = _check_weights(weights, caller)
    if cov_type not in ("nonrobust", "hc1"):
        raise ValueError(
            f"{caller}: cov_type must be 'nonrobust' or 'hc1', got {cov_type!r}. For a "
            "Conley spatial-HAC covariance call puremacro.spatial.conley_cov directly; it is "
            "deliberately not wired into the LM battery, which assumes homoskedastic "
            "residuals."
        )
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant,
                          x_names=x_names, caller=caller)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}")
    yv, Xm, n, k = des.y, des.X, des.n, des.k
    XtX_inv = inv_xtx(Xm, name=caller)
    b = XtX_inv @ (Xm.T @ yv)
    fitted = Xm @ b
    e = yv - fitted
    ess = float(e @ e)
    sigma2 = ess / (n - k)
    if cov_type == "hc1":
        meat = (Xm.T * (e ** 2)) @ Xm
        cov = (n / (n - k)) * (XtX_inv @ meat @ XtX_inv)
    else:
        cov = sigma2 * XtX_inv
    bse = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = np.where(bse > 0, b / bse, np.nan)
    pv = 2.0 * stats.t.sf(np.abs(tv), n - k)
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - k))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ess / tss if tss > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    loglik = -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(ess / n))
    n_params = k + 1
    if k > 1:
        f_stat = (r2 / (k - 1)) / ((1.0 - r2) / (n - k))
        f_p = float(stats.f.sf(f_stat, k - 1, n - k))
    else:
        f_stat, f_p = float("nan"), float("nan")
    names = des.names
    lm = lm_spatial_tests(pd.Series(yv, index=W.ids), pd.DataFrame(Xm, index=W.ids, columns=names),
                          W, add_constant=False)
    return SpatialOLSResult(
        params=pd.Series(b, index=names), bse=pd.Series(bse, index=names),
        t_values=pd.Series(tv, index=names), p_values=pd.Series(pv, index=names),
        conf_int=pd.DataFrame({"lower": b - crit * bse, "upper": b + crit * bse}, index=names),
        vcov=cov, sigma2=sigma2, resid=e, fitted=fitted,
        r2=float(r2), r2_adj=float(r2_adj), loglik=float(loglik),
        aic=float(-2 * loglik + 2 * n_params), bic=float(-2 * loglik + n_params * np.log(n)),
        f_stat=float(f_stat), f_pvalue=float(f_p),
        n=n, k=k, exog_names=names, weights_ids=W.ids, cov_type=cov_type,
        lm=lm, weights=W, alpha=float(alpha),
    )


# ---------------------------------------------------------------------------
# Shared plumbing for the four model entry points
# ---------------------------------------------------------------------------
def _resolve_vcov(vcov: str | None, method: str, caller: str) -> str:
    """``None`` -> ``'analytic'`` for ML and ``'classical'`` for GMM."""
    ml_ok = ("analytic", "hessian")
    gmm_ok = ("classical", "robust")
    if vcov is None:
        return "analytic" if method == "ml" else "classical"
    if method == "ml":
        if vcov == "robust":
            raise ValueError(
                f"{caller}: vcov='robust' is not available with method='ml'. Under "
                "heteroskedasticity of unknown form the Gaussian ML estimator of the spatial "
                "parameter is itself inconsistent (Lin & Lee 2010), so a robust sandwich "
                "would be a standard error for the wrong number. Use method='gmm' with "
                "vcov='robust', or pass obs_weights= if the relative precisions are known."
            )
        if vcov not in ml_ok:
            raise ValueError(f"{caller}: with method='ml', vcov must be one of {ml_ok} "
                             f"or None, got {vcov!r}")
        return vcov
    if vcov not in gmm_ok:
        raise ValueError(
            f"{caller}: with method='gmm', vcov must be one of {gmm_ok} or None, got "
            f"{vcov!r} (there is no likelihood, so 'analytic'/'hessian' are undefined)"
        )
    return vcov


def _reject_ml_only(caller: str, method: str, **kw: Any) -> None:
    """Raise when an ML-only keyword is passed with ``method='gmm'`` (and back)."""
    if method != "gmm":
        return
    for name, val in kw.items():
        if val is not None:
            raise ValueError(
                f"{caller}: `{name}` is meaningless with method='gmm' (there is no "
                f"likelihood and no log-determinant to compute). Got {name}={val!r}."
            )


def _resolve_spatial_setup(
    W: SpatialWeights,
    *,
    logdet: str,
    logdet_order: int | None,
    bounds_arg: Any,
    caller: str,
    bounds_name: str,
    need_logdet: bool = True,
) -> tuple[_LogDet, tuple[float, float], tuple[float, float], np.ndarray | None]:
    """Spectrum, admissible interval and the log-determinant callable.

    The dense spectrum is computed only when it is actually needed — for
    ``bounds_arg='auto'`` or for ``logdet='eig'``. With an explicit interval and
    ``logdet='lu'`` no eigendecomposition happens at all, which is the whole
    point of the LU route at large ``n``; ``rho_bounds_singularity`` is then
    ``(nan, nan)`` and the score-root refinement is skipped.

    ``need_logdet=False`` says the caller will never evaluate the returned
    ``_LogDet`` — the GMM routes have no likelihood and no log-determinant. The
    ``logdet='eig'`` trigger is then dropped, so ``method='gmm'`` with an
    explicit interval does no eigendecomposition at all, and with
    ``bounds_arg='auto'`` the warning names the interval as the only escape
    (``logdet='lu'`` is rejected outright under ``method='gmm'``).
    """
    if logdet not in _LOGDET_METHODS:
        raise ValueError(f"{caller}: logdet must be one of {_LOGDET_METHODS}, got {logdet!r}")
    if logdet_order is not None and logdet != "chebyshev":
        raise ValueError(
            f"{caller}: logdet_order only applies to logdet='chebyshev' (got "
            f"logdet={logdet!r}); the exact routes have no order."
        )
    order = 5 if logdet_order is None else int(logdet_order)
    resolved = logdet
    if logdet == "auto":
        resolved = "eig" if W.n <= 1000 else "lu"
    # The dense spectrum is needed for the automatic interval and for the 'eig'
    # route, and for nothing else. Skipping it when neither applies is what
    # makes logdet='lu' with an explicit interval genuinely cheap at large n.
    for_bounds = isinstance(bounds_arg, str)
    for_logdet = need_logdet and resolved == "eig"
    want_spectrum = for_bounds or for_logdet
    spectrum: np.ndarray | None = None
    sing: tuple[float, float] = (float("nan"), float("nan"))
    auto_bounds: tuple[float, float] | None = None
    if want_spectrum:
        escapes = []
        if for_bounds:
            escapes.append(f"an explicit {bounds_name}=(lo, hi) pair")
        if for_logdet:
            escapes.append("logdet='lu'")
        escape = f"Pass {' and '.join(escapes)} to avoid it"
        # Only the lag models have impacts, and effects() rebuilds the spectrum
        # for itself, so say so there and nowhere else.
        escape += (" (effects() would still recompute it if you ask for impacts)."
                   if bounds_name == "rho_bounds" else ".")
        spectrum, _ = _spectrum(W, escape=escape)
        auto_bounds, sing = _admissible_bounds(spectrum)
    if isinstance(bounds_arg, str):
        if bounds_arg != "auto":
            raise ValueError(
                f"{caller}: {bounds_name} must be 'auto' or a (lo, hi) pair, got {bounds_arg!r}"
            )
        assert auto_bounds is not None
        bounds = auto_bounds
    else:
        lo, hi = (float(v) for v in bounds_arg)
        if not (lo < hi):
            raise ValueError(f"{caller}: {bounds_name} = ({lo}, {hi}) is not an increasing interval")
        if not (lo <= 0.0 <= hi):
            raise ValueError(
                f"{caller}: {bounds_name} = ({lo}, {hi}) does not contain 0, so the "
                "no-spatial-dependence null is outside the search region. The code never "
                "silently re-centres the interval."
            )
        bounds = (lo, hi)
    # With need_logdet=False the callable is never evaluated, so build the
    # cheap 'lu' one: constructing an 'eig' _LogDet would trigger exactly the
    # O(n^3) eigendecomposition this branch exists to avoid.
    ld = _LogDet(W, method="lu" if not need_logdet else logdet, order=order,
                 spectrum=spectrum, caller=caller)
    return ld, bounds, sing, spectrum


def _warn_islands(W: SpatialWeights, caller: str) -> int:
    n_isl = W.n_islands
    if n_isl:
        warnings.warn(
            f"{caller}: {n_isl} unit(s) have no neighbours ({list(W.islands)[:5]}). They "
            "contribute to beta but carry no information about the spatial parameter, and "
            "the closed-form total effect beta/(1 - rho) does not apply to this W.",
            RuntimeWarning, stacklevel=3,
        )
    return n_isl


def _durbin_block(
    des: _Design, W: SpatialWeights, durbin: Any, caller: str
) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
    """``(W X_d, unlagged names, positions in X)`` for the Durbin/SLX block.

    The **constant is always excluded**. For a row-standardised island-free
    ``W`` the lagged constant is exactly collinear with the constant
    (``W 1 = 1``); with islands it is a zero/one island indicator, which is a
    different specification the user must construct explicitly.

    The fallback the empty-block errors point at depends on the caller: an SDM
    with no Durbin block is a SAR, but an **SLX** with no Durbin block is plain
    OLS, so ``slx`` is sent to :func:`ols_spatial` and never to ``sar()``, which
    would silently substitute a model carrying a ``rho``.
    """
    no_durbin = "ols_spatial()" if caller == "slx" else "sar()"
    names = list(des.names)
    non_const = [j for j in range(des.k) if j != des.const_index]
    if durbin is None:
        picks = non_const
    else:
        if isinstance(durbin, (str, int, np.integer)):
            durbin = [durbin]
        picks = []
        for item in durbin:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if not (0 <= idx < des.k):
                    raise ValueError(
                        f"{caller}: durbin index {idx} is outside [0, {des.k}); negative and "
                        "out-of-range integers are rejected rather than wrapped."
                    )
            else:
                if str(item) not in names:
                    raise KeyError(
                        f"{caller}: durbin entry {item!r} is not a regressor "
                        f"(available: {names})"
                    )
                idx = names.index(str(item))
            if idx == des.const_index:
                raise ValueError(
                    f"{caller}: the constant cannot enter the Durbin block — for a "
                    "row-standardised W it is exactly collinear with itself lagged."
                )
            if idx in picks:
                raise ValueError(f"{caller}: durbin lists {names[idx]!r} twice")
            picks.append(idx)
        if not picks:
            raise ValueError(
                f"{caller}: durbin is empty. Pass durbin=None for every non-constant "
                f"column, or call {no_durbin} for no Durbin block."
            )
    if not picks:
        raise ValueError(
            f"{caller}: there is no non-constant regressor to lag, so the Durbin block is "
            f"empty. Use {no_durbin} instead."
        )
    lagged = W.W @ des.X[:, picks]
    return lagged, tuple(names[j] for j in picks), picks


def _common_factor_test(
    vcov: np.ndarray,
    vnames: Sequence[str],
    beta: np.ndarray,
    theta: np.ndarray,
    rho: float,
    durbin_names: Sequence[str],
    exog_names: Sequence[str],
) -> tuple[float, int, float]:
    """Wald test of ``H0: theta + rho beta_d = 0`` (Anselin 1988, pp. 226-229)."""
    kd = len(durbin_names)
    vn = list(vnames)
    base_idx = [vn.index(nm) for nm in durbin_names]
    lag_idx = [vn.index("W_" + nm) for nm in durbin_names]
    rho_idx = vn.index("rho")
    sel = base_idx + lag_idx + [rho_idx]
    V = vcov[np.ix_(sel, sel)]
    beta_d = np.array([beta[list(exog_names).index(nm)] for nm in durbin_names])
    g = theta + rho * beta_d
    Gj = np.zeros((2 * kd + 1, kd))
    Gj[:kd, :] = rho * np.eye(kd)
    Gj[kd:2 * kd, :] = np.eye(kd)
    Gj[2 * kd, :] = beta_d
    mid = Gj.T @ V @ Gj
    stat = float(g @ np.linalg.solve(mid, g))
    return stat, kd, float(stats.chi2.sf(stat, kd))


def _reduced_form(W: SpatialWeights, rho: float, Zb: np.ndarray) -> np.ndarray:
    """``(I - rho W)^-1 Z b`` by one sparse solve."""
    n = W.n
    A = (sp.identity(n, format="csc") - rho * W.W.tocsc()).tocsc()
    return np.asarray(spsolve(A, Zb), dtype=float)


def _lag_result(
    des: _Design,
    W: SpatialWeights,
    Z: np.ndarray,
    znames: tuple[str, ...],
    *,
    model: str,
    durbin_names: tuple[str, ...],
    n_durbin: int,
    method: str,
    logdet: str,
    logdet_order: int | None,
    rho_bounds: Any,
    vcov: str | None,
    gmm_lags: int | None,
    xtol: float,
    alpha: float,
    common_factor: bool,
    caller: str,
    instrument_extra: int,
) -> SpatialModelResult:
    """Estimate ``y = rho W y + Z delta + eps`` and package the result."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}")
    if method not in ("ml", "gmm"):
        raise ValueError(f"{caller}: method must be 'ml' or 'gmm', got {method!r}")
    vcov_type = _resolve_vcov(vcov, method, caller)
    if W.W.nnz == 0:
        raise ValueError(
            f"{caller}: the weights have no neighbours, so rho is not identified. Use "
            "puremacro.spatial.ols_spatial for a non-spatial fit."
        )
    n_isl = _warn_islands(W, caller)
    n, kz = Z.shape
    yv = des.y
    rho_admissible = True
    ld, bounds, sing, spectrum = _resolve_spatial_setup(
        W, logdet=logdet, logdet_order=logdet_order, bounds_arg=rho_bounds,
        caller=caller, bounds_name="rho_bounds", need_logdet=method == "ml",
    )

    if method == "ml":
        core = _lag_ml_core(yv, Z, W, obs_w=des.obs_w, logdet=ld, bounds=bounds,
                            xtol=xtol, caller=caller)
        rho = core["rho"]
        beta = core["beta"]
        sigma2 = core["sigma2"]
        resid = core["resid"]
        if vcov_type == "analytic":
            info = _lag_information(Z, W, beta=beta, rho=rho, sigma2=sigma2,
                                    obs_w=des.obs_w, spectrum=spectrum, caller=caller)
            cov_full = _psd_inverse(info, caller=caller)
        else:
            Wy = W.W @ yv
            w_obs = des.obs_w if des.obs_w is not None else np.ones(n)
            logw = 0.5 * float(np.sum(np.log(w_obs)))
            const = -0.5 * n * np.log(2.0 * np.pi)

            def neg_ll(psi: np.ndarray) -> float:
                b = psi[:kz]
                r = float(psi[kz])
                s2 = float(np.exp(psi[kz + 1]))
                e = (yv - r * Wy) - Z @ b
                return -(const - 0.5 * n * np.log(s2) + ld(r)
                         - 0.5 * float(np.sum(w_obs * e * e)) / s2 + logw)

            psi0 = np.concatenate([beta, [rho, np.log(sigma2)]])
            H = _numerical_hessian_vcov(neg_ll, psi0)
            cov_psi = _psd_inverse(H, caller=caller)
            D = np.eye(kz + 2)
            D[kz + 1, kz + 1] = sigma2
            cov_full = D @ cov_psi @ D
        vnames = tuple([*znames, "rho", "sigma2"])
        params = np.concatenate([beta, [rho]])
        pnames = tuple([*znames, "rho"])
        cov_params = cov_full[: kz + 1, : kz + 1]
        loglik = core["loglik"]
        loglik0 = core["loglik_at_zero"]
        n_params = kz + 2
        converged, n_iter, at_bound, score_root = (
            core["converged"], core["n_iter"], core["at_bound"], core["score_root"])
        grid = np.linspace(bounds[0], bounds[1], 201)
        prof = np.array([core["objective"](float(r)) for r in grid])
    else:
        if des.obs_w is not None:
            raise ValueError(
                f"{caller}: obs_weights is a maximum-likelihood feature (it enters the "
                "Gaussian likelihood as a known relative precision); it is not defined for "
                "method='gmm'."
            )
        q = 2 if gmm_lags is None else int(gmm_lags)
        if q < 1:
            raise ValueError(f"{caller}: gmm_lags must be at least 1, got {q}")
        X_ex = des.X[:, [j for j in range(des.k) if j != des.const_index]]
        core = _lag_gmm_core(yv, Z, X_ex, W, gmm_lags=q + instrument_extra,
                             vcov=vcov_type, caller=caller)
        delta = core["delta"]
        beta = delta[:kz]
        rho = float(delta[kz])
        sigma2 = core["sigma2"]
        resid = core["resid"]
        cov_params = core["vcov"]
        cov_full = cov_params
        vnames = tuple([*znames, "rho"])
        params = delta
        pnames = vnames
        loglik = float("nan")
        loglik0 = float("nan")
        n_params = kz + 1
        converged, n_iter, at_bound, score_root = True, 1, False, None
        grid = np.asarray([], dtype=float)
        prof = np.asarray([], dtype=float)
        # Kelejian-Prucha 2SLS is unconstrained: nothing in the estimator keeps
        # rho inside the admissible interval. Saying so is build rule G5 — the
        # alternative is a result object that reports a rho, an interval it is
        # not in, and a summary line asserting non-singularity there.
        if not (bounds[0] < rho < bounds[1]):
            rho_admissible = False
            warnings.warn(
                f"{caller}: the Kelejian-Prucha 2SLS estimate rho = {rho:.6f} lies OUTSIDE "
                f"the admissible interval ({bounds[0]:.6f}, {bounds[1]:.6f}) on which "
                "(I - rho W)^-1 has a convergent spatial-multiplier expansion. The 2SLS "
                "estimator is unconstrained, so this is a symptom of a weakly identified or "
                "misspecified spatial lag, not a bug. fitted / resid_reduced / pseudo_r2 are NaN "
                "(they would run through an explosive multiplier) and effects() will raise. "
                "Refit with method='ml', which maximises over the interval.",
                RuntimeWarning, stacklevel=3,
            )

    bse = np.sqrt(np.clip(np.diag(cov_params), 0.0, None))
    z, p, lo, hi = _zstats(params, bse, alpha)
    if rho_admissible:
        fitted = _reduced_form(W, rho, Z @ beta)
    else:
        fitted = np.full(n, np.nan)
    theta = beta[kz - n_durbin:] if n_durbin else None
    cf = None
    if common_factor and n_durbin:
        cf = _common_factor_test(cov_full, vnames, beta[: kz - n_durbin], theta, rho,
                                 durbin_names, znames[: kz - n_durbin])
    corr = np.corrcoef(yv, fitted)[0, 1] if rho_admissible else float("nan")
    return SpatialModelResult(
        model=model, method=method,
        params=pd.Series(params, index=pnames), bse=pd.Series(bse, index=pnames),
        z_values=pd.Series(z, index=pnames), p_values=pd.Series(p, index=pnames),
        conf_int=pd.DataFrame({"lower": lo, "upper": hi}, index=pnames),
        vcov=cov_full, vcov_names=vnames,
        beta=beta[: kz - n_durbin] if n_durbin else beta,
        theta=theta, durbin_names=durbin_names,
        rho=rho, lam=None, sigma2=sigma2,
        loglik=loglik,
        aic=float(-2 * loglik + 2 * n_params), bic=float(-2 * loglik + n_params * np.log(n)),
        n_params=n_params, loglik_at_zero=loglik0,
        pseudo_r2=float(corr ** 2), resid=resid, resid_reduced=yv - fitted, fitted=fitted,
        n=n, k=kz, exog_names=znames, const_index=des.const_index, weights_ids=W.ids,
        rho_bounds=bounds, rho_bounds_singularity=sing,
        logdet_method=ld.method if method == "ml" else "none",
        vcov_type=vcov_type, vcov_traces_exact=True,
        converged=converged, n_iter=n_iter, at_bound=at_bound, n_islands=n_isl,
        common_factor=cf, score_root=score_root, spectrum=spectrum,
        profile_grid=grid, profile_loglik=prof, weights=W,
        obs_weights=des.obs_w, alpha=float(alpha), y_name=des.y_name,
        rho_admissible=rho_admissible,
    )


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------
def sar(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    method: str = "ml",
    obs_weights: Any = None,
    logdet: str = "auto",
    logdet_order: int | None = None,
    rho_bounds: Any = "auto",
    vcov: str | None = None,
    gmm_lags: int | None = None,
    xtol: float = 1e-8,
    alpha: float = 0.05,
) -> SpatialModelResult:
    """Spatial autoregressive (lag) model ``y = rho W y + X beta + eps``.

    Parameters
    ----------
    y : array_like, pandas.Series or str
        Dependent variable, or a column name when ``data`` is given.
    X : array_like, pandas.DataFrame or sequence of str
        Regressors, or column names when ``data`` is given.
    weights : SpatialWeights
    data : pandas.DataFrame, optional
    add_constant : bool, default True
    x_names : sequence of str, optional
    method : {'ml', 'gmm'}, default 'ml'
        ``'ml'`` is the concentrated Gaussian likelihood; ``'gmm'`` is the
        Kelejian-Prucha (1998) spatial two-stage least squares.
    obs_weights : array_like, optional
        Known relative precisions ``w_i`` (``Var(eps_i) = sigma^2 / w_i``),
        normalised to mean one. ML only. The spatial lag is formed **first**
        and weighted second, and the Jacobian ``ln|A|`` is unchanged (it is the
        Jacobian of ``y -> eps`` and does not involve ``Omega``); the
        likelihood gains ``+ (1/2) sum ln w_i``.
    logdet : {'auto', 'eig', 'lu', 'chebyshev'}, default 'auto'
        ``'auto'`` is ``'eig'`` for ``n <= 1000`` and ``'lu'`` above. ``'eig'``
        and ``'lu'`` are both exact and agree to ~1e-12. ``'chebyshev'`` is an
        **approximation** (Pace & LeSage 2004): on a 20x20 rook lattice at
        order 5 its absolute error is ~2e-4 at ``rho = 0.3``, ~5e-3 at
        ``rho = 0.5`` and ~0.66 at ``rho = 0.9``. It requires a symmetric or
        symmetrisable ``W`` whose eigenvalues lie in ``[-1, 1]``.
    logdet_order : int, optional
        Chebyshev degree; defaults to 5 and is capped at 8 (the matrix powers
        fill in: on a 30x30 rook lattice ``W`` is 0.4% dense and ``W^8`` about
        7%). Passing it with any other ``logdet`` raises.
    rho_bounds : {'auto'} or (float, float), default 'auto'
        ``'auto'`` is the non-singularity interval over the **real**
        eigenvalues intersected with ``(-1/max|lam_i|, 1/max|lam_i|)`` over the
        full complex spectrum. Both are reported (``rho_bounds`` and
        ``rho_bounds_singularity``). An explicit interval must contain 0 — and
        passing one together with ``logdet='lu'`` is what makes the LU route
        genuinely cheap at large ``n``, because the automatic interval is the
        only other thing that needs the dense eigendecomposition. In that case
        ``rho_bounds_singularity`` is ``(nan, nan)``, ``spectrum`` is ``None``
        and the score-root refinement is skipped.

        It applies to ``method='gmm'`` too, and is the *only* ML-adjacent
        keyword that does. Kelejian-Prucha 2SLS never searches the interval —
        it is unconstrained — but the interval is what the returned ``rho`` is
        certified against (:attr:`SpatialModelResult.rho_admissible`), what
        ``rho_bounds`` reports, and what :meth:`SpatialModelResult.effects`
        uses. Passing it explicitly is therefore the way to make a GMM fit skip
        the dense eigendecomposition entirely, since ``logdet='lu'`` is
        rejected under ``method='gmm'`` and would be no escape anyway.
    vcov : {None, 'analytic', 'hessian', 'classical', 'robust'}, optional
        ``None`` resolves to ``'analytic'`` for ML and ``'classical'`` for GMM.
        ``'robust'`` with ``method='ml'`` raises (see Notes).
    gmm_lags : int, optional
        Instrument order ``q`` in ``H = [X, W X_ex, ..., W^q X_ex]``; defaults
        to 2. GMM only.
    xtol : float, default 1e-8
        Absolute tolerance of the bounded scalar optimiser. This is a floor,
        not a target: a Brent minimiser cannot localise the argmax of a smooth
        function better than about ``sqrt(eps) ~ 1.5e-8``, so tightening it
        further buys nothing. The optimum is refined by ``brentq`` on the
        analytic score whenever the spectrum is available.
    alpha : float, default 0.05

    Returns
    -------
    SpatialModelResult

    Raises
    ------
    TypeError
        ``weights`` is not a :class:`SpatialWeights`, or an unknown keyword is
        passed (there is no ``**kwargs`` sink anywhere in this module).
    ValueError
        ``len(y) != weights.n``; NaN or inf in ``y``/``X`` (the column is
        named); ``n <= k + 2``; ``y`` constant, or fitted exactly by ``X`` so
        that no residual variation is left and the spatial parameter is not
        identified; ``W`` has no non-zero weight; a constant column
        already in ``X`` with ``add_constant=True``; ``obs_weights`` non-finite,
        non-positive or of the wrong length; ``method``/``vcov``/``logdet``
        outside their enumerations; ``vcov='robust'`` with ``method='ml'``;
        ``vcov='analytic'``/``'hessian'`` with ``method='gmm'``; ``logdet``,
        ``logdet_order``, ``rho_bounds`` or ``obs_weights`` with
        ``method='gmm'``; ``gmm_lags`` with ``method='ml'``;
        ``logdet='chebyshev'`` on a non-symmetrisable ``W``; ``rho_bounds``
        that does not contain 0.
    KeyError
        A named column is absent from ``data``, or a unit of ``weights.ids`` is
        missing from a pandas index.
    numpy.linalg.LinAlgError
        ``X'X`` is singular; the message names the collinear columns.

    Warns
    -----
    RuntimeWarning
        Islands present; the optimum within ``10 * xtol`` of a bound; the score
        root and the bounded optimum disagreeing by more than
        ``max(100 * xtol, 1e-6)`` (searched over a bracket 100 times that wide,
        so the check can actually fire); an *interior* optimum across which the
        analytic score does not change sign, so ``score_root`` stays ``None``;
        an inadmissible ``rho`` from ``method='gmm'``; a dense ``n x n`` object
        formed above ``n = 2000``; instrument columns dropped for collinearity.

    Notes
    -----
    Under heteroskedasticity of unknown form the Gaussian SAR ML estimator of
    ``rho`` is **inconsistent** (Lin & Lee 2010) — no sandwich rescues it, which
    is why ``vcov='robust'`` raises here instead of quietly computing one. Lee
    (2004)'s QML robustness is to non-normality only, and even then the ``rho``
    and ``sigma^2`` blocks need his corrections.

    A ``rho`` estimated on a binary ``W`` is not comparable with one estimated
    on the row-standardised version: the normalisation changes the scale of the
    parameter (on a 6x6 rook lattice a binary ``W`` restricts ``rho`` to about
    ``(-0.277, 0.277)``).

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import sar
    >>> side = 7
    >>> nb = {}
    >>> for i in range(side):
    ...     for j in range(side):
    ...         u = i * side + j
    ...         nb[u] = [v for v in ((i - 1) * side + j, (i + 1) * side + j,
    ...                              u - 1 if j > 0 else -1, u + 1 if j < side - 1 else -1)
    ...                  if 0 <= v < side * side and v != u]
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(20240906)
    >>> n = side * side
    >>> X = rng.normal(size=(n, 2))
    >>> A = np.linalg.inv(np.eye(n) - 0.5 * W.to_dense())
    >>> y = A @ (1.0 + X @ np.array([1.0, -0.5]) + rng.normal(size=n))
    >>> res = sar(y, X, W)
    >>> round(float(res.rho), 4)
    0.4325
    >>> round(float(res.params['x1']), 4)
    0.8775
    """
    caller = "sar"
    W = _check_weights(weights, caller)
    # `rho_bounds` is NOT ml-only: under method='gmm' it is the admissible
    # interval the unconstrained 2SLS estimate is certified against, the one
    # reported in `rho_bounds`, and the one effects() uses -- and passing it
    # explicitly is what lets a GMM fit skip the dense eigendecomposition.
    _reject_ml_only(caller, method, logdet=None if logdet == "auto" else logdet,
                    logdet_order=logdet_order)
    if method == "ml" and gmm_lags is not None:
        raise ValueError(f"{caller}: gmm_lags is meaningless with method='ml'.")
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant, x_names=x_names,
                          obs_weights=obs_weights, caller=caller)
    return _lag_result(
        des, W, des.X, des.names, model="sar", durbin_names=(), n_durbin=0,
        method=method, logdet=logdet, logdet_order=logdet_order, rho_bounds=rho_bounds,
        vcov=vcov, gmm_lags=gmm_lags, xtol=xtol, alpha=alpha, common_factor=False,
        caller=caller, instrument_extra=0,
    )


def sdm(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    durbin: Sequence[str] | Sequence[int] | None = None,
    method: str = "ml",
    obs_weights: Any = None,
    logdet: str = "auto",
    logdet_order: int | None = None,
    rho_bounds: Any = "auto",
    vcov: str | None = None,
    gmm_lags: int | None = None,
    common_factor: bool = True,
    xtol: float = 1e-8,
    alpha: float = 0.05,
) -> SpatialModelResult:
    """Spatial Durbin model ``y = rho W y + X beta + W X_d theta + eps``.

    A SAR whose regressor block is augmented with the spatial lags of the
    (non-constant) covariates, plus the common-factor Wald test of the SEM
    nesting. ``sdm`` is *literally* :func:`sar` on ``Z = [X, W X_d]`` — the same
    concentration, the same log-determinant machinery, the same information
    matrix with ``X`` replaced by ``Z`` — and the identity
    ``sdm(y, X, W) == sar(y, [X, W X_ex], W)`` holds to machine precision and is
    pinned in the tests.

    Parameters
    ----------
    durbin : sequence of str or int, optional
        Which regressors to lag. Default: every non-constant column. **The
        constant is always excluded**: for a row-standardised island-free ``W``,
        ``W 1 = 1`` makes the lagged constant exactly collinear with the
        constant, and with islands it is an island indicator — a different
        specification the user must build explicitly.
    common_factor : bool, default True
        Report the Wald test of ``H0: theta = -rho beta_d``, whose non-rejection
        says the SDM collapses to a SEM. Only defined when **every**
        non-constant regressor is lagged: with a partial Durbin block the
        restriction does not nest SEM at all, so ``common_factor=True`` with a
        strict subset in ``durbin`` raises rather than printing a claim about a
        model that is not in the parameter space being tested.
    y, X, weights, data, add_constant, x_names, method, obs_weights, logdet, logdet_order, rho_bounds, vcov, gmm_lags, xtol, alpha
        As in :func:`sar`.

    Returns
    -------
    SpatialModelResult
        ``theta`` and ``durbin_names`` are populated; the lagged columns are
        named ``W_<name>``.

    Raises
    ------
    ValueError
        Everything :func:`sar` raises, plus: ``durbin`` empty, listing the
        constant, listing a column twice, or carrying an integer outside
        ``[0, k)``; ``common_factor=True`` with a partial Durbin block.
    KeyError
        ``durbin`` names a column that is not in the design.
    numpy.linalg.LinAlgError
        ``[X, W X_d]`` is rank-deficient — the usual cause is a group dummy
        that ``W`` maps into a multiple of itself. The message names the
        columns.

    Notes
    -----
    The common-factor restriction is only *interpretable* where
    ``sign(theta_r) = -sign(rho beta_r)`` for every ``r``. When that fails, the
    statistic is still reported but :meth:`SpatialModelResult.summary` appends
    the caveat rather than suppressing it.

    For the GMM path the instrument set gains one extra lag order,
    ``H = [X, W X_ex, ..., W^{q+1} X_ex]``, because ``W X_ex`` is now an
    included regressor and cannot instrument for ``W y``.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import sdm
    >>> nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < 40] for i in range(40)}
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(40, 2))
    >>> A = np.linalg.inv(np.eye(40) - 0.3 * W.to_dense())
    >>> y = A @ (X @ [1.0, -0.5] + W.lag(X) @ [0.4, 0.0] + rng.normal(scale=0.5, size=40))
    >>> res = sdm(y, X, W)
    >>> res.durbin_names
    ('x1', 'x2')
    >>> bool(res.common_factor[2] < 0.05)
    True
    """
    caller = "sdm"
    W = _check_weights(weights, caller)
    # `rho_bounds` is NOT ml-only: under method='gmm' it is the admissible
    # interval the unconstrained 2SLS estimate is certified against, the one
    # reported in `rho_bounds`, and the one effects() uses -- and passing it
    # explicitly is what lets a GMM fit skip the dense eigendecomposition.
    _reject_ml_only(caller, method, logdet=None if logdet == "auto" else logdet,
                    logdet_order=logdet_order)
    if method == "ml" and gmm_lags is not None:
        raise ValueError(f"{caller}: gmm_lags is meaningless with method='ml'.")
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant, x_names=x_names,
                          obs_weights=obs_weights, caller=caller)
    lagged, dnames, picks = _durbin_block(des, W, durbin, caller)
    n_non_const = des.k - (1 if des.const_index is not None else 0)
    if common_factor and len(dnames) != n_non_const:
        raise ValueError(
            f"{caller}: common_factor=True needs the FULL Durbin block ({n_non_const} "
            f"non-constant regressors), but durbin selects {len(dnames)}. With a partial "
            "block the restriction theta = -rho*beta does not nest the spatial error "
            "model, so reporting it as 'SEM is not rejected' would be false. Pass "
            "common_factor=False, or durbin=None."
        )
    Z = np.column_stack([des.X, lagged])
    znames = tuple([*des.names, *(f"W_{nm}" for nm in dnames)])
    return _lag_result(
        des, W, Z, znames, model="sdm", durbin_names=dnames, n_durbin=len(dnames),
        method=method, logdet=logdet, logdet_order=logdet_order, rho_bounds=rho_bounds,
        vcov=vcov, gmm_lags=gmm_lags, xtol=xtol, alpha=alpha, common_factor=common_factor,
        caller=caller, instrument_extra=1,
    )


def sem(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    method: str = "ml",
    obs_weights: Any = None,
    logdet: str = "auto",
    logdet_order: int | None = None,
    lambda_bounds: Any = "auto",
    vcov: str | None = None,
    n_iter: int | None = None,
    xtol: float = 1e-8,
    alpha: float = 0.05,
) -> SpatialModelResult:
    """Spatial error model ``y = X beta + u``, ``u = lam W u + eps``.

    Parameters
    ----------
    lambda_bounds : {'auto'} or (float, float), default 'auto'
        As ``rho_bounds`` in :func:`sar`.
    n_iter : int, optional
        GMM only: how many times to alternate the Kelejian-Prucha (1999)
        three-moment step with the spatially filtered least-squares step.
        Defaults to 1 (the classic two-step estimator). Values above 1 iterate
        until ``|lam_new - lam_old| < xtol``.
    y, X, weights, data, add_constant, x_names, method, obs_weights, logdet, logdet_order, vcov, xtol, alpha
        As in :func:`sar`.

    Returns
    -------
    SpatialModelResult
        ``lam`` is populated and ``rho`` is ``None``. ``bse['lambda']`` is
        ``NaN`` for ``method='gmm'``.

    Raises
    ------
    ValueError
        Everything :func:`sar` raises, with ``lambda_bounds`` in place of
        ``rho_bounds``; plus ``n_iter < 1``, and ``n_iter`` passed with
        ``method='ml'``.

    Warns
    -----
    RuntimeWarning
        As :func:`sar`, plus a non-converged ``lambda`` iteration when
        ``n_iter > 1``.

    Notes
    -----
    Unlike the lag model there is no scalar profile shortcut: ``beta(lam)`` is
    the spatially filtered GLS estimator and is recomputed at every ``lam``.
    The ten Gram matrices of ``[X, WX, y, Wy]`` are formed once, so each
    evaluation is ``O(k^2)`` with no pass over the data.

    The information matrix (Anselin 1988, eq. 6.28) is **block diagonal**
    between ``beta`` and ``(lambda, sigma^2)``, which is why
    ``SE(beta) = sqrt(diag(sigma^2 (Xs'Xs)^-1))`` with ``Xs = (I - lam W) X``
    is exact here and not a plug-in.

    The GMM path solves the three-moment system by bounded L-BFGS-B rather than
    an unbounded simplex, so ``lambda`` cannot leave the admissible interval;
    agreement with ``spreg.GM_Error`` is to about six significant figures, not
    eight, because the two optimisers stop in different places on the same flat
    surface. The reported ``sigma2`` is that of the **filtered** residuals, not
    the GMM ``sigma2`` from the moment system; the two genuinely differ.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import sem
    >>> side = 7
    >>> nb = {}
    >>> for i in range(side):
    ...     for j in range(side):
    ...         u = i * side + j
    ...         nb[u] = [v for v in ((i - 1) * side + j, (i + 1) * side + j,
    ...                              u - 1 if j > 0 else -1, u + 1 if j < side - 1 else -1)
    ...                  if 0 <= v < side * side and v != u]
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(11)
    >>> n = side * side
    >>> X = rng.normal(size=(n, 2))
    >>> B = np.linalg.inv(np.eye(n) - 0.6 * W.to_dense())
    >>> y = 1.0 + X @ np.array([1.0, -0.5]) + B @ rng.normal(size=n)
    >>> res = sem(y, X, W)
    >>> round(float(res.lam), 4)   # true value 0.6; n = 49 is a small sample
    0.401
    >>> res.rho is None
    True
    """
    caller = "sem"
    W = _check_weights(weights, caller)
    # `lambda_bounds` is NOT ml-only: under method='gmm' it bounds the
    # Kelejian-Prucha moment minimisation, and passing it explicitly lets the
    # fit skip the dense eigendecomposition.
    _reject_ml_only(caller, method, logdet=None if logdet == "auto" else logdet,
                    logdet_order=logdet_order)
    if method == "ml" and n_iter is not None:
        raise ValueError(
            f"{caller}: n_iter is meaningless with method='ml' (the concentrated likelihood "
            "is maximised once over lambda)."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}")
    if method not in ("ml", "gmm"):
        raise ValueError(f"{caller}: method must be 'ml' or 'gmm', got {method!r}")
    vcov_type = _resolve_vcov(vcov, method, caller)
    if W.W.nnz == 0:
        raise ValueError(
            f"{caller}: the weights have no neighbours, so lambda is not identified. Use "
            "puremacro.spatial.ols_spatial for a non-spatial fit."
        )
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant, x_names=x_names,
                          obs_weights=obs_weights, caller=caller)
    n_isl = _warn_islands(W, caller)
    yv, Xm, n, k = des.y, des.X, des.n, des.k
    ld, bounds, sing, spectrum = _resolve_spatial_setup(
        W, logdet=logdet, logdet_order=logdet_order, bounds_arg=lambda_bounds,
        caller=caller, bounds_name="lambda_bounds", need_logdet=method == "ml",
    )
    names = des.names
    if method == "ml":
        core = _error_ml_core(yv, Xm, W, obs_w=des.obs_w, logdet=ld, bounds=bounds,
                              xtol=xtol, caller=caller)
        lam = core["lam"]
        beta = core["beta"]
        sigma2 = core["sigma2"]
        resid = core["resid"]
        Xs = Xm - lam * (W.W @ Xm)
        if vcov_type == "analytic":
            info = _error_information(Xs, W, lam=lam, sigma2=sigma2, obs_w=des.obs_w,
                                      spectrum=spectrum, caller=caller)
            cov_full = _psd_inverse(info, caller=caller)
        else:
            w_obs = des.obs_w if des.obs_w is not None else np.ones(n)
            logw = 0.5 * float(np.sum(np.log(w_obs)))
            const = -0.5 * n * np.log(2.0 * np.pi)
            WX = W.W @ Xm
            Wy = W.W @ yv

            def neg_ll(psi: np.ndarray) -> float:
                b = psi[:k]
                l_ = float(psi[k])
                s2 = float(np.exp(psi[k + 1]))
                e = (yv - l_ * Wy) - (Xm - l_ * WX) @ b
                return -(const - 0.5 * n * np.log(s2) + ld(l_)
                         - 0.5 * float(np.sum(w_obs * e * e)) / s2 + logw)

            H = _numerical_hessian_vcov(neg_ll, np.concatenate([beta, [lam, np.log(sigma2)]]))
            cov_psi = _psd_inverse(H, caller=caller)
            D = np.eye(k + 2)
            D[k + 1, k + 1] = sigma2
            cov_full = D @ cov_psi @ D
        vnames = tuple([*names, "lambda", "sigma2"])
        cov_params = cov_full[: k + 1, : k + 1]
        loglik = core["loglik"]
        loglik0 = core["loglik_at_zero"]
        n_params = k + 2
        converged, n_iter_out, at_bound = core["converged"], core["n_iter"], core["at_bound"]
        grid = np.linspace(bounds[0], bounds[1], 201)
        prof = np.array([core["objective"](float(v)) for v in grid])
    else:
        if des.obs_w is not None:
            raise ValueError(
                f"{caller}: obs_weights is a maximum-likelihood feature; it is not defined "
                "for method='gmm'."
            )
        iters = 1 if n_iter is None else int(n_iter)
        if iters < 1:
            raise ValueError(f"{caller}: n_iter must be at least 1, got {iters}")
        core = _error_gmm_core(yv, Xm, W, n_iter=iters, vcov=vcov_type, bounds=bounds,
                               xtol=xtol, caller=caller)
        lam = core["lam"]
        beta = core["beta"]
        sigma2 = core["sigma2"]
        resid = core["resid"]
        cov_full = core["vcov"]
        cov_params = cov_full
        vnames = tuple([*names, "lambda"])
        loglik = float("nan")
        loglik0 = float("nan")
        n_params = k + 1
        converged, n_iter_out, at_bound = core["converged"], core["n_iter"], False
        grid = np.asarray([], dtype=float)
        prof = np.asarray([], dtype=float)

    params = np.concatenate([beta, [lam]])
    pnames = tuple([*names, "lambda"])
    diag = np.diag(cov_params).copy()
    bse = np.where(np.isfinite(diag) & (diag >= 0), np.sqrt(np.clip(diag, 0.0, None)), np.nan)
    z, p, lo, hi = _zstats(params, bse, alpha)
    fitted = Xm @ beta
    corr = np.corrcoef(yv, fitted)[0, 1]
    return SpatialModelResult(
        model="sem", method=method,
        params=pd.Series(params, index=pnames), bse=pd.Series(bse, index=pnames),
        z_values=pd.Series(z, index=pnames), p_values=pd.Series(p, index=pnames),
        conf_int=pd.DataFrame({"lower": lo, "upper": hi}, index=pnames),
        vcov=cov_full, vcov_names=vnames,
        beta=beta, theta=None, durbin_names=(),
        rho=None, lam=float(lam), sigma2=sigma2,
        loglik=loglik,
        aic=float(-2 * loglik + 2 * n_params), bic=float(-2 * loglik + n_params * np.log(n)),
        n_params=n_params, loglik_at_zero=loglik0,
        pseudo_r2=float(corr ** 2), resid=resid, resid_reduced=yv - fitted, fitted=fitted,
        n=n, k=k, exog_names=names, const_index=des.const_index, weights_ids=W.ids,
        rho_bounds=bounds, rho_bounds_singularity=sing,
        logdet_method=ld.method if method == "ml" else "none",
        vcov_type=vcov_type, vcov_traces_exact=True,
        converged=converged, n_iter=n_iter_out, at_bound=at_bound, n_islands=n_isl,
        common_factor=None, score_root=None, spectrum=spectrum,
        profile_grid=grid, profile_loglik=prof, weights=W,
        obs_weights=des.obs_w, alpha=float(alpha), y_name=des.y_name,
    )


def slx(
    y: Any,
    X: Any,
    weights: SpatialWeights,
    *,
    data: pd.DataFrame | None = None,
    add_constant: bool = True,
    x_names: Sequence[str] | None = None,
    durbin: Sequence[str] | Sequence[int] | None = None,
    cov_type: str = "nonrobust",
    alpha: float = 0.05,
) -> SpatialModelResult:
    """Spatially lagged X model ``y = X beta + W X_d theta + eps`` — theta without rho.

    The one member of the family that needs no spatial parameter and no
    likelihood: it is ordinary least squares on ``Z = [X, W X_d]``. It is worth
    a first-class entry point because it is the specification the LM battery
    never points at (the battery tests for ``rho`` and ``lambda``, not for
    ``theta``) and because its impacts are exact and local — ``direct = beta``,
    ``indirect = theta * S0/n`` — with no spatial multiplier at all.

    Parameters
    ----------
    durbin : sequence of str or int, optional
        Which regressors to lag; default every non-constant column. The
        constant is always excluded, as in :func:`sdm`.
    cov_type : {'nonrobust', 'hc1'}, default 'nonrobust'
    y, X, weights, data, add_constant, x_names, alpha
        As in :func:`sar`.

    Returns
    -------
    SpatialModelResult
        ``model='slx'``, ``method='ols'``, ``rho`` and ``lam`` are ``None``.
        ``sigma2`` is dof-corrected (``e'e/(n - k)``) and the reported statistic
        is still an asymptotic ``z``; use :func:`ols_spatial` when exact ``t``
        inference matters.

    Raises
    ------
    ValueError
        As :func:`sdm`'s ``durbin`` validation, plus ``cov_type`` outside its
        enumeration.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import slx
    >>> nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < 40] for i in range(40)}
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(40, 2))
    >>> y = 1.0 + X @ [1.0, -0.5] + W.lag(X) @ [0.8, 0.0] + rng.normal(scale=0.3, size=40)
    >>> res = slx(y, X, W)
    >>> round(float(res.params['W_x1']), 2)
    0.82
    >>> res.rho is None
    True
    """
    caller = "slx"
    W = _check_weights(weights, caller)
    if cov_type not in ("nonrobust", "hc1"):
        raise ValueError(f"{caller}: cov_type must be 'nonrobust' or 'hc1', got {cov_type!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}")
    des = _prepare_design(y, X, W, data=data, add_constant=add_constant, x_names=x_names,
                          caller=caller)
    lagged, dnames, _ = _durbin_block(des, W, durbin, caller)
    n_isl = _warn_islands(W, caller) if W.n_islands else 0
    Z = np.column_stack([des.X, lagged])
    znames = tuple([*des.names, *(f"W_{nm}" for nm in dnames)])
    yv, n = des.y, des.n
    kz = Z.shape[1]
    ZtZ_inv = inv_xtx(Z, name=caller)
    delta = ZtZ_inv @ (Z.T @ yv)
    fitted = Z @ delta
    e = yv - fitted
    ess = float(e @ e)
    sigma2 = ess / (n - kz)
    if cov_type == "hc1":
        meat = (Z.T * (e ** 2)) @ Z
        cov = (n / (n - kz)) * (ZtZ_inv @ meat @ ZtZ_inv)
    else:
        cov = sigma2 * ZtZ_inv
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z, p, lo, hi = _zstats(delta, bse, alpha)
    loglik = -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(ess / n))
    n_params = kz + 1
    corr = np.corrcoef(yv, fitted)[0, 1]
    return SpatialModelResult(
        model="slx", method="ols",
        params=pd.Series(delta, index=znames), bse=pd.Series(bse, index=znames),
        z_values=pd.Series(z, index=znames), p_values=pd.Series(p, index=znames),
        conf_int=pd.DataFrame({"lower": lo, "upper": hi}, index=znames),
        vcov=cov, vcov_names=znames,
        beta=delta[: kz - len(dnames)], theta=delta[kz - len(dnames):], durbin_names=dnames,
        rho=None, lam=None, sigma2=sigma2,
        loglik=float(loglik), aic=float(-2 * loglik + 2 * n_params),
        bic=float(-2 * loglik + n_params * np.log(n)),
        n_params=n_params, loglik_at_zero=float(loglik),
        pseudo_r2=float(corr ** 2), resid=e, resid_reduced=e, fitted=fitted,
        n=n, k=kz, exog_names=znames, const_index=des.const_index, weights_ids=W.ids,
        rho_bounds=(0.0, 0.0), rho_bounds_singularity=(float("-inf"), float("inf")),
        logdet_method="none", vcov_type=cov_type, vcov_traces_exact=True,
        converged=True, n_iter=1, at_bound=False, n_islands=n_isl,
        common_factor=None, score_root=None, spectrum=None,
        profile_grid=np.asarray([], dtype=float), profile_loglik=np.asarray([], dtype=float),
        weights=W, obs_weights=None, alpha=float(alpha), y_name=des.y_name,
    )


def _draw_parameters(
    b: np.ndarray,
    th: np.ndarray,
    rho: float,
    V: np.ndarray,
    bounds: tuple[float, float],
    n_sim: int,
    seed: int | None,
    caller: str,
) -> tuple[np.ndarray, int, int]:
    """Draw ``[beta, theta, rho]`` from ``N(psi_hat, V)`` and drop inadmissible rho.

    Draws whose ``rho`` falls outside the admissible interval are **discarded**,
    not clipped, so the sampled impacts stay finite and the reported quantiles
    are quantiles of a genuine distribution rather than of a pile-up at the
    boundary.
    """
    rng = np.random.default_rng(seed)
    mean = np.concatenate([b, th, [rho]])
    Vs = 0.5 * (V + V.T)
    try:
        L = np.linalg.cholesky(Vs + 1e-12 * np.eye(Vs.shape[0]))
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(Vs)
        if float(evals.min()) < -1e-10 * max(1.0, float(evals.max())):
            warnings.warn(
                f"{caller}: the supplied covariance is not positive semi-definite "
                f"(smallest eigenvalue {evals.min():.3e}); negative eigenvalues were "
                "clipped to zero before drawing.",
                RuntimeWarning, stacklevel=3,
            )
        L = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))
    raw = mean + (rng.standard_normal((n_sim, mean.shape[0])) @ L.T)
    ok = (raw[:, -1] > bounds[0]) & (raw[:, -1] < bounds[1])
    n_dropped = int(np.sum(~ok))
    n_kept = int(np.sum(ok))
    frac = n_dropped / float(n_sim)
    if frac > 0.5:
        raise ValueError(
            f"{caller}: {frac:.0%} of the {n_sim} draws fell outside the admissible "
            f"interval ({bounds[0]:.4f}, {bounds[1]:.4f}); the estimated rho is too close "
            "to the boundary for the simulation to be meaningful."
        )
    if frac > 0.05:
        warnings.warn(
            f"{caller}: {frac:.1%} of the draws fell outside the admissible interval and "
            "were discarded (not clipped).",
            RuntimeWarning, stacklevel=3,
        )
    return raw[ok], n_kept, n_dropped


def spatial_effects(
    weights: SpatialWeights,
    beta: Any,
    rho: float,
    *,
    theta: Any = None,
    vcov: Any = None,
    names: Sequence[str] | None = None,
    const_index: int | None = None,
    n_sim: int | None = None,
    seed: int | None = 0,
    alpha: float = 0.05,
    keep_draws: bool = False,
    spectrum: Any = None,
    n_draws: int | None = None,
) -> SpatialEffectsResult:
    """LeSage-Pace (2009, sec. 2.7) direct / indirect / total impact decomposition.

    A spatial-lag coefficient is not a marginal effect. The ``r``-th
    regressor's impact matrix is ``S_r(W) = A^-1 (I beta_r + W theta_r)`` with
    ``A = I - rho W``, and the three summaries are ``direct_r = tr(S_r)/n``,
    ``total_r = 1'S_r 1/n`` and ``indirect_r = total_r - direct_r``.

    ``indirect_r = (1'S_r 1 - tr S_r)/n`` is the **sum** of all ``n(n-1)``
    off-diagonal cross-partials divided by ``n`` — the average total spillover a
    unit sends (equivalently, receives) — not the effect on any one neighbour,
    and not the mean individual cross-partial, which is smaller by a factor of
    ``n - 1``.

    Parameters
    ----------
    weights : SpatialWeights
    beta : array_like
        ``(k,)`` coefficients over one shared name list.
    rho : float
        The spatial-lag parameter. Use ``0.0`` for an SLX model.
    theta : array_like, optional
        ``(k,)`` Durbin coefficients over the **same** name list, with
        ``theta_r = 0`` for the regressors that carry no Durbin term. ``beta``
        and ``theta`` must have the same length: they are aligned *by variable*,
        not by block. ``None`` means all zeros.
    vcov : array_like, optional
        Covariance of ``[beta, theta, rho]`` — shape ``(2k + 1, 2k + 1)`` — or,
        when ``theta`` is absent, of ``[beta, rho]`` (``(k + 1, k + 1)``).
        Structurally-zero ``theta`` entries take zero rows and columns.
        ``None`` gives point estimates with ``NaN`` standard errors; it is
        **never** an error.
    names : sequence of str, optional
        Variable labels, length ``k``. Defaults to ``x1 ... xk``.
    const_index : int, optional
        Position of the constant, dropped from the reported table (its "effect"
        is not a marginal effect). When ``names`` contains exactly one entry
        called ``'const'`` and ``const_index`` is not given, that one is used.
    n_sim : int, default 0
        Parameter draws for the simulated standard errors. ``0`` (the default)
        returns point estimates only, so the plainest call
        ``spatial_effects(W, beta, rho)`` always works.
    seed : int or None, default 0
    alpha : float, default 0.05
    keep_draws : bool, default False
    spectrum : array_like, optional
        Pre-computed eigenvalues of ``W`` (a fitted model passes its own),
        saving the dense eigendecomposition.
    n_draws : int, optional
        Accepted alias for ``n_sim``, because
        :func:`puremacro.spatial.spatial_panel` spells the same operation
        ``n_draws``. ``n_sim`` is the canonical spelling *here*; passing both
        raises. The two functions' **defaults** differ on purpose and neither
        should be aligned to the other: ``spatial_effects`` takes an optional
        ``vcov=`` and so cannot default to simulating (with no covariance there
        is nothing to draw from), whereas ``spatial_panel`` always has a fitted
        covariance in hand and therefore defaults to ``n_draws=1000``.

    Returns
    -------
    SpatialEffectsResult

    Raises
    ------
    TypeError
        ``weights`` is not a :class:`SpatialWeights`.
    ValueError
        ``beta`` and ``theta`` of different lengths; ``rho`` not finite or
        outside the interval implied by ``W`` (the message reports it);
        ``n_sim < 0``; ``alpha`` outside ``(0, 1)``; ``vcov`` not square or of
        the wrong dimension; ``n_sim > 0`` with ``vcov=None``; more than 50% of
        the draws falling outside the admissible interval; both ``n_sim=`` and
        ``n_draws=`` given.

    Warns
    -----
    RuntimeWarning
        More than 5% of the draws discarded; a non-positive-definite ``vcov``
        whose negative eigenvalues had to be clipped.

    Notes
    -----
    **Divergence from spreg, deliberate.** ``spreg``'s ``spat_impacts='full'``
    computes ``direct_r = beta_r tr(A^-1)/n`` and assigns every SLX effect to
    the indirect column ("Assumes all SLX effects are indirect effects"),
    dropping the ``theta_r tr(A^-1 W)/n`` term from the direct effect. This
    module follows the textbook and keeps it. The **total** is identical either
    way, so for a SAR (``theta = 0``) the two agree exactly.

    ``spreg``'s ``spmultiplier`` also hard-codes the average total impact as
    ``1/(1 - rho)``. That equals ``1'A^-1 1/n`` **only** for a row-standardised,
    island-free ``W``; with a binary ``W`` or a single island it is simply
    wrong (on a 5x5 lattice with one island at ``rho = 0.6`` the true value is
    2.4423 against 2.5). This function always computes ``1'A^-1 1`` and never
    the shortcut.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.models import spatial_effects
    >>> nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < 20] for i in range(20)}
    >>> W = contiguity_weights(nb)
    >>> eff = spatial_effects(W, np.array([1.0, -0.5]), 0.0, names=['x1', 'x2'])
    >>> np.allclose(eff.direct, [1.0, -0.5]) and np.allclose(eff.indirect, 0.0)
    True
    >>> eff = spatial_effects(W, np.array([1.0]), 0.5, names=['x1'])
    >>> round(float(eff.total[0]), 6)
    2.0
    """
    caller = "spatial_effects"
    n_sim = _resolve_n_sim(n_sim, n_draws, caller)
    W = _check_weights(weights, caller)
    b = np.asarray(beta, dtype=float).ravel()
    k = b.shape[0]
    if theta is None:
        th = np.zeros(k)
        has_durbin = False
    else:
        th = np.asarray(theta, dtype=float).ravel()
        if th.shape[0] != k:
            raise ValueError(
                f"{caller}: beta has {k} entries and theta has {th.shape[0]}. They are "
                "aligned BY VARIABLE over one shared name list — pass theta_r = 0 for the "
                "regressors that carry no Durbin term (this is the normal case for a "
                "partial Durbin block, and for the constant)."
            )
        has_durbin = bool(np.any(th != 0.0))
    rho = float(rho)
    if not np.isfinite(rho):
        raise ValueError(f"{caller}: rho must be finite, got {rho}")
    if n_sim < 0:
        raise ValueError(f"{caller}: n_sim must be non-negative, got {n_sim}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}")
    if names is None:
        vnames = tuple(f"x{i + 1}" for i in range(k))
    else:
        vnames = tuple(str(s) for s in names)
        if len(vnames) != k:
            raise ValueError(f"{caller}: names has {len(vnames)} entries for {k} coefficients")
    if const_index is None and list(vnames).count("const") == 1:
        const_index = list(vnames).index("const")

    spec = np.asarray(spectrum) if spectrum is not None else _spectrum(W)[0]
    bounds, _ = _admissible_bounds(spec)
    if not (bounds[0] <= rho <= bounds[1]):
        raise ValueError(
            f"{caller}: rho = {rho} lies outside the interval ({bounds[0]:.6f}, "
            f"{bounds[1]:.6f}) on which (I - rho W) is invertible with a convergent "
            "spatial multiplier."
        )
    n = W.n
    radius = float(np.abs(spec).max()) if spec.size else 0.0

    if vcov is None:
        V: np.ndarray | None = None
    else:
        V = np.asarray(vcov, dtype=float)
        if V.ndim != 2 or V.shape[0] != V.shape[1]:
            raise ValueError(f"{caller}: vcov must be square, got shape {V.shape}")
        if V.shape[0] == k + 1:
            big = np.zeros((2 * k + 1, 2 * k + 1))
            idx = list(range(k)) + [2 * k]
            big[np.ix_(idx, idx)] = V
            V = big
        elif V.shape[0] != 2 * k + 1:
            raise ValueError(
                f"{caller}: vcov must be ({2 * k + 1}, {2 * k + 1}) over [beta, theta, rho] "
                f"or ({k + 1}, {k + 1}) over [beta, rho], got {V.shape}"
            )
    if n_sim > 0 and V is None:
        raise ValueError(
            f"{caller}: n_sim = {n_sim} needs a covariance. Pass vcov=, or leave n_sim=0 "
            "for point estimates with NaN standard errors."
        )

    # Draw first, so the power series is sized to the rho values actually used
    # rather than to the (much wider) admissible interval.
    kept: np.ndarray | None = None
    n_kept = 0
    n_dropped = 0
    if n_sim > 0 and V is not None:
        kept, n_kept, n_dropped = _draw_parameters(
            b, th, rho, V, bounds, int(n_sim), seed, caller
        )
    rho_max = abs(rho) if kept is None or n_kept == 0 else max(
        abs(rho), float(np.abs(kept[:, -1]).max()))
    order = _series_order(radius, rho_max)
    m = _row_power_sums(W, order + 1) if order is not None else None
    trace_method = "spectrum+series" if m is not None else "spectrum+solve"

    tau, t_w, s, s_w = _scalar_terms(W, rho, spectrum=spec, m=m)
    A = (sp.identity(n, format="csc") - rho * W.W.tocsc()).tocsc()
    s_exact = float(np.sum(spsolve(A, np.ones(n))))
    s_check = abs(s - s_exact)
    direct, indirect, total = _effects_from_params(b, th, n, tau, t_w, s, s_w)

    keep = [j for j in range(k) if j != const_index]
    sl = np.asarray(keep, dtype=int)
    nan_se = np.full(len(keep), np.nan)
    nan_ci = np.full((len(keep), 2), np.nan)
    draws_out = None
    se = [nan_se.copy() for _ in range(3)]
    ci = [nan_ci.copy() for _ in range(3)]
    pv = [nan_se.copy() for _ in range(3)]

    if kept is not None and n_kept > 0:
        sims = np.empty((n_kept, len(keep), 3))
        for i in range(n_kept):
            bi = kept[i, :k]
            ti = kept[i, k:2 * k]
            ri = float(kept[i, -1])
            t1, t2, s1, s2 = _scalar_terms(W, ri, spectrum=spec, m=m)
            d_, ind_, tot_ = _effects_from_params(bi, ti, n, t1, t2, s1, s2)
            sims[i, :, 0] = d_[sl]
            sims[i, :, 1] = ind_[sl]
            sims[i, :, 2] = tot_[sl]
        qlo, qhi = alpha / 2.0, 1.0 - alpha / 2.0
        for c in range(3):
            se[c] = sims[:, :, c].std(axis=0, ddof=1) if n_kept > 1 else nan_se.copy()
            ci[c] = np.column_stack([
                np.quantile(sims[:, :, c], qlo, axis=0),
                np.quantile(sims[:, :, c], qhi, axis=0),
            ]) if n_kept > 1 else nan_ci.copy()
            above = np.mean(sims[:, :, c] > 0.0, axis=0)
            below = np.mean(sims[:, :, c] < 0.0, axis=0)
            pv[c] = np.clip(2.0 * np.minimum(above, below), 1.0 / max(n_kept, 1), 1.0)
        if keep_draws:
            draws_out = sims

    return SpatialEffectsResult(
        variables=tuple(vnames[j] for j in keep),
        direct=direct[sl], indirect=indirect[sl], total=total[sl],
        direct_se=se[0], indirect_se=se[1], total_se=se[2],
        direct_ci=ci[0], indirect_ci=ci[1], total_ci=ci[2],
        direct_p=pv[0], indirect_p=pv[1], total_p=pv[2],
        rho=rho, has_durbin=has_durbin,
        tau=tau, t_w=t_w, s=s, s_w=s_w, s_solve_check=float(s_check),
        trace_method=trace_method,
        n=n, n_sim=int(n_sim), n_kept=n_kept, n_dropped=n_dropped,
        seed=seed, alpha=float(alpha), draws=draws_out,
    )
