"""Validation cases for the ``garch`` subsystem.

Scope: ``puremacro.garch.garch11_fit`` (Gaussian MLE of a GARCH(1,1)). Two
PACKAGE cross-checks against the independent ``arch`` library (params and the
conditional-volatility path on the same simulated series) plus four INTERNAL
checks: the GARCH(1,1) variance recursion identity, the ``persistence == α+β``
bookkeeping identity, a simulate-then-recover consistency check, and a
log-likelihood self-consistency check.

Pure deps only — the PACKAGE references are the frozen goldens in
``goldens/garch.json`` (live ``arch`` is touched only by the ``reference``-marked
drift-guard in ``tests/validation/test_reference_drift_garch.py``). The
``arch`` package resolves to ``arch.univariate.mean`` — a wholly independent
source tree from ``puremacro.garch.fit`` (verified with ``inspect.getsourcefile``),
so the cross-check is non-circular.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase

# Burn-in dropped before comparing conditional-volatility *paths* to ``arch``.
# puremacro seeds σ²₀ with the sample variance while ``arch`` backcasts it with
# an EWMA of squared residuals; the two recursions coincide once that one-off
# initialisation transient has decayed (well under this many observations for a
# persistence of ≈0.98). The fitted *parameters* are unaffected and compared on
# the full series.
_BURN = 200


def garch_demo_data() -> dict:
    """A long, stable GARCH(1,1) sample with known parameters.

    Seeded → byte-identical every call, so the puremacro fit and its frozen
    ``arch`` reference are built from the same series. ``T`` is deliberately
    large (5000): a high-information sample sharply peaks the Gaussian
    likelihood, so puremacro's and ``arch``'s independent L-BFGS-B runs converge
    to the same MLE rather than to distinct local optima.
    """
    rng = np.random.default_rng(20260531)
    T = 5000
    omega, alpha, beta = 0.05, 0.08, 0.90
    eps = np.empty(T)
    sigma2 = np.empty(T)
    sigma2[0] = omega / (1.0 - alpha - beta)  # unconditional variance
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return {
        "y": eps,
        "true_params": {"omega": omega, "alpha": alpha, "beta": beta},
        "burn": _BURN,
    }


# --- compute callables (puremacro estimator imported lazily, pyodide-light) ---


def _fit_params() -> dict:
    from puremacro.garch import garch11_fit

    r = garch11_fit(garch_demo_data()["y"], mean="zero")
    return {"omega": r.omega, "alpha": r.alpha, "beta": r.beta}


def _fit_cond_vol_tail() -> dict:
    from puremacro.garch import garch11_fit

    d = garch_demo_data()
    r = garch11_fit(d["y"], mean="zero")
    return {"sigma_tail": np.asarray(r.sigma.values[d["burn"]:], dtype=float)}


def _recursion_residual() -> dict:
    """σ²_t reported by the fit must satisfy the GARCH(1,1) recursion exactly."""
    from puremacro.garch import garch11_fit

    y = garch_demo_data()["y"]
    r = garch11_fit(y, mean="zero")
    sig2 = np.asarray(r.sigma.values, dtype=float) ** 2
    rhs = r.omega + r.alpha * y[:-1] ** 2 + r.beta * sig2[:-1]
    return {"max_abs_resid": float(np.max(np.abs(sig2[1:] - rhs)))}


def _persistence_identity() -> dict:
    from puremacro.garch import garch11_fit

    r = garch11_fit(garch_demo_data()["y"], mean="zero")
    return {"persistence_gap": float(abs(r.persistence - (r.alpha + r.beta)))}


def _simulate_then_recover() -> dict:
    from puremacro.garch import garch11_fit

    d = garch_demo_data()
    r = garch11_fit(d["y"], mean="zero")
    return {"omega": r.omega, "alpha": r.alpha, "beta": r.beta}


def _loglik_self_consistency() -> dict:
    """Reported log-likelihood equals the Gaussian GARCH(1,1) log-likelihood
    re-evaluated independently at the fitted parameters (σ²₀ = sample variance,
    matching the estimator's initialisation)."""
    from puremacro.garch import garch11_fit

    y = garch_demo_data()["y"]
    r = garch11_fit(y, mean="zero")
    T = len(y)
    sigma2 = np.empty(T)
    sigma2[0] = float(np.var(y))
    for t in range(1, T):
        sigma2[t] = r.omega + r.alpha * y[t - 1] ** 2 + r.beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-10)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + y ** 2 / sigma2)
    return {"loglik": float(r.loglik), "loglik_recomputed": float(ll)}


def _garch_midas_components_identity() -> dict:
    from puremacro.midas import garch_midas

    rng = np.random.default_rng(20260815)
    returns = rng.normal(0, 0.02, 600)
    res = garch_midas(returns, x_lf=None, K=20, L=6)
    tau = np.asarray(res.tau)
    g = np.asarray(res.g)
    sigma = np.asarray(res.sigma)
    max_gap = float(np.max(np.abs(sigma**2 - tau * g)))
    w_sum_gap = float(abs(np.sum(res.weights) - 1.0))
    return {"max_gap": max_gap, "w_sum_gap": w_sum_gap}


_BOLLERSLEV = (
    "Bollerslev, T. (1986). Generalized autoregressive conditional "
    "heteroskedasticity. Journal of Econometrics 31(3), 307-327."
)


CASES: list[ValidationCase] = [
    ValidationCase(
        id="garch.params_vs_arch",
        subsystem="garch",
        title="GARCH(1,1) MLE params (ω, α, β) match the arch package",
        title_es="Los parámetros EMV (ω, α, β) del GARCH(1,1) coinciden con el paquete arch",
        mechanism=Mechanism.PACKAGE,
        compute=_fit_params,
        reference="garch:params_vs_arch",  # frozen golden
        tol=Tol.NUMERIC,  # distinct L-BFGS-B runs → params agree to ~1e-3
        citation=(
            "arch 8.0.0 arch_model(y, mean='Zero', vol='GARCH', p=1, q=1, "
            "dist='normal', rescale=False).fit() — independent Gaussian MLE."
        ),
        notes="Same simulated series; reference is the arch ML estimate of (ω, α, β).",
    ),
    ValidationCase(
        id="garch.cond_vol_vs_arch",
        subsystem="garch",
        title="Conditional volatility path matches the arch package (post burn-in)",
        title_es="La trayectoria de volatilidad condicional coincide con el paquete arch (tras el calentamiento)",
        mechanism=Mechanism.PACKAGE,
        compute=_fit_cond_vol_tail,
        reference="garch:cond_vol_vs_arch",  # frozen golden
        tol=Tol.NUMERIC,
        citation=(
            "arch 8.0.0 arch_model(...).fit().conditional_volatility — same "
            "GARCH(1,1) recursion, independent implementation."
        ),
        notes=(
            "First 200 observations dropped: puremacro seeds σ²₀ with the sample "
            "variance, arch backcasts it, so the paths coincide only after that "
            "initialisation transient decays."
        ),
    ),
    ValidationCase(
        id="garch.recursion_identity",
        subsystem="garch",
        title="Fitted σ²_t satisfies the GARCH(1,1) variance recursion",
        title_es="La σ²_t ajustada satisface la recursión de varianza GARCH(1,1)",
        mechanism=Mechanism.INTERNAL,
        compute=_recursion_residual,
        reference=lambda: {"max_abs_resid": 0.0},
        tol=Tol.TIGHT,
        citation=(
            "GARCH(1,1) conditional-variance recursion σ²_t = ω + α u²_{t-1} + "
            "β σ²_{t-1} (" + _BOLLERSLEV + ")."
        ),
    ),
    ValidationCase(
        id="garch.persistence_identity",
        subsystem="garch",
        title="Reported persistence equals α + β",
        title_es="La persistencia reportada es igual a α + β",
        mechanism=Mechanism.INTERNAL,
        compute=_persistence_identity,
        reference=lambda: {"persistence_gap": 0.0},
        tol=Tol.EXACT,
        citation=(
            "GARCH(1,1) persistence is the sum of the ARCH and GARCH "
            "coefficients α + β (" + _BOLLERSLEV + ")."
        ),
    ),
    ValidationCase(
        id="garch.simulate_then_recover",
        subsystem="garch",
        title="MLE recovers the data-generating GARCH(1,1) parameters",
        title_es="La EMV recupera los parámetros generadores del GARCH(1,1)",
        mechanism=Mechanism.INTERNAL,
        compute=_simulate_then_recover,
        reference=lambda: garch_demo_data()["true_params"],
        tol=Tol.COARSE,  # finite-sample MLE sampling error
        citation=(
            "Consistency of the Gaussian quasi-MLE for GARCH(1,1) "
            "(Bollerslev & Wooldridge 1992, Econometric Reviews 11(2), 143-172)."
        ),
        notes="True DGP (ω, α, β) = (0.05, 0.08, 0.90); recovered on T=5000.",
    ),
    ValidationCase(
        id="garch.loglik_self_consistency",
        subsystem="garch",
        title="Reported log-likelihood matches an independent re-evaluation",
        title_es="La log-verosimilitud reportada coincide con una reevaluación independiente",
        mechanism=Mechanism.INTERNAL,
        compute=_loglik_self_consistency,
        reference=lambda: {
            "loglik": _loglik_self_consistency()["loglik_recomputed"],
            "loglik_recomputed": _loglik_self_consistency()["loglik_recomputed"],
        },
        tol=Tol.TIGHT,
        citation=(
            "Gaussian GARCH(1,1) log-likelihood "
            "ℓ = -½ Σ_t [log 2π + log σ²_t + u²_t / σ²_t] (" + _BOLLERSLEV + ")."
        ),
    ),
    ValidationCase(
        id="garch.midas_variance_decomposition_identity",
        subsystem="garch",
        title="GARCH-MIDAS total conditional variance equals product of long-run and short-run components",
        title_es="La varianza condicional total de GARCH-MIDAS equivale al producto de las componentes de largo y corto plazo",
        mechanism=Mechanism.ANALYTICAL,
        compute=_garch_midas_components_identity,
        reference=lambda: {"max_gap": 0.0, "w_sum_gap": 0.0},
        tol=Tol.EXACT,
        citation=(
            "Engle, Ghysels & Sohn (2013, RESTAT 95(3), 776-798): conditional variance "
            "decomposes as sigma_{i,t}^2 = tau_t * g_{i,t} with Beta weights summing to 1."
        ),
    ),
]
