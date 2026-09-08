"""Econometric estimators (pyodide-clean, pure numpy).

Besides the panel local projection that has always lived here, this package
holds the statsmodels-parity replacements added in 2.6.0: ``ols`` (with the
robust/HAC/cluster covariance family, ``t_test`` and ``f_test``), ``logit``
and ``poisson``, and ``quantreg``. All of them import numpy / scipy / pandas
and nothing else, so a notebook that swaps statsmodels for these keeps
running under Pyodide.
"""
# ``ols`` names two things: this package's ``ols`` submodule and the estimator
# inside it. Bind the submodule FIRST so it is already in ``sys.modules`` when
# a later ``import puremacro.regress.ols`` runs — the import machinery only
# setattr's the parent-package attribute when it actually loads a module, so
# after this the callable below keeps the name. Same tactic, same reason, as
# ``newey_west`` in ``puremacro.inference``. Either spelling still reaches the
# function: ``from puremacro.regress import ols`` and
# ``from puremacro.regress.ols import ols``, and
# ``importlib.import_module("puremacro.regress.ols")`` still returns the module
# (which is what the public-API and Pyodide sweeps use). The one spelling that
# does NOT give you the module is ``import puremacro.regress.ols as m``: since
# Python 3.7 that resolves by ``getattr`` on the parent package first, so ``m``
# is the function. Use ``from ... import`` for the names you want.
from . import ols as _ols_module  # noqa: F401

from .lp import lp_panel
from .ols import (
    add_constant,
    ols,
    OLSResult,
    TTestResult,
    FTestResult,
    PredictionResult,
)
from .discrete import (
    logit,
    poisson,
    DiscreteResult,
    PerfectSeparationError,
    ConvergenceWarning,
)
from .quantile import quantreg, QuantRegResult

__all__ = [
    # Panel local projection. Deprecated — it emits a FutureWarning and
    # points at puremacro.lp.panel.panel_lp / puremacro.lp.panel_dk.panel_lp_dk
    # — but kept exported, because retirement is deferred and callers exist.
    "lp_panel",
    # Linear regression.
    "add_constant",
    "ols",
    "OLSResult",
    "TTestResult",
    "FTestResult",
    "PredictionResult",
    # Limited-dependent-variable models.
    "logit",
    "poisson",
    "DiscreteResult",
    "PerfectSeparationError",
    "ConvergenceWarning",
    # Quantile regression.
    "quantreg",
    "QuantRegResult",
]
