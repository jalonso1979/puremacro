"""Compute budgets sized to the device.

A 2,000-draw wild bootstrap is thirty seconds on a workstation and a
dead notebook on an iPad — same code, same call, different machine. This
module turns "how much compute can I ask for here?" into a value you can
read, rather than a number you hard-code and later regret.

Two ways to use it:

**Explicitly** — ask for the clamped value and pass it on::

    n_boot = runtime.fit(n_boot=2000)["n_boot"]   # 400 on a tablet
    res = cholesky_svar(Y, p=2, n_boot=n_boot)

**By wrapping** — let the wrapper clamp the cost arguments it recognises::

    svar = runtime.budgeted(cholesky_svar)
    res = svar(Y, p=2, n_boot=2000)   # warns once, runs 400 on a tablet

Deliberately **not** done here: reaching into the estimators themselves.
Every ``cholesky_svar`` / ``lp_hac`` / ``VFIProblem`` keeps its exact
documented default, so a script that runs on your laptop produces
bit-identical output after this module lands. The clamp only ever
happens where a caller asks for it.

Only parameters that change *cost* are clamped, never ones that change
the *estimand*. ``n_boot`` narrows or widens confidence bands (a
precision knob); ``horizon`` changes what is being estimated, so it is
left alone even though it is just as expensive.
"""
from __future__ import annotations

import functools
import inspect
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

from puremacro.runtime._capabilities import capabilities

__all__ = [
    "Budget",
    "TIERS",
    "current",
    "fit",
    "budgeted",
    "override",
    "BudgetWarning",
]


class BudgetWarning(UserWarning):
    """Raised (as a warning) when a requested workload was clamped."""


@dataclass(frozen=True)
class Budget:
    """Ceilings on the four workload dimensions that dominate run time.

    Attributes
    ----------
    tier : str
        ``"workstation"``, ``"tablet"`` or ``"minimal"``.
    n_boot : int
        Bootstrap / simulation replications for confidence bands.
    n_draws : int
        Posterior draws (BVAR Gibbs, DSGE Metropolis-Hastings, sign
        restrictions).
    n_grid : int
        Points per state-space dimension in VFI / EGM solvers.
    n_sim : int
        Periods in a stochastic simulation.
    """

    tier: str
    n_boot: int
    n_draws: int
    n_grid: int
    n_sim: int

    def cap(self, field: str) -> int:
        """The ceiling for ``field``; raises ``KeyError`` if unknown."""
        if field not in _COST_FIELDS:
            raise KeyError(
                f"{field!r} is not a budget dimension; expected one of "
                f"{sorted(_COST_FIELDS)}"
            )
        return int(getattr(self, field))


_COST_FIELDS = frozenset({"n_boot", "n_draws", "n_grid", "n_sim"})

# Tier ceilings. "workstation" is set high enough to be a no-op for any
# call a person would plausibly write by hand — it exists so that the
# wrapper is safe to leave in code that also runs on a laptop.
TIERS: dict[str, Budget] = {
    "workstation": Budget(
        tier="workstation", n_boot=100_000, n_draws=1_000_000,
        n_grid=10_000, n_sim=1_000_000,
    ),
    "tablet": Budget(
        tier="tablet", n_boot=400, n_draws=20_000, n_grid=300, n_sim=5_000,
    ),
    "minimal": Budget(
        tier="minimal", n_boot=100, n_draws=5_000, n_grid=120, n_sim=1_000,
    ),
}

# Argument names estimators across the package use for each dimension.
# Kept explicit rather than pattern-matched: a false positive here would
# silently shrink someone's workload.
_ALIASES: dict[str, str] = {
    "n_boot": "n_boot",
    "nboot": "n_boot",
    "n_bootstrap": "n_boot",
    "n_rep": "n_boot",
    "n_reps": "n_boot",
    "n_draws": "n_draws",
    "ndraws": "n_draws",
    "n_iter": "n_draws",
    "n_keep": "n_draws",
    "n_grid": "n_grid",
    "grid_size": "n_grid",
    "n_a": "n_grid",
    "n_k": "n_grid",
    "n_sim": "n_sim",
    "n_periods": "n_sim",
    "T_sim": "n_sim",
}

_MEMORY_MINIMAL_MB = 2048  # below this, drop a tablet to the minimal tier

_FORCED: str | None = None
_WARNED: set[tuple[str, str, str]] = set()


def _tier_for_device() -> str:
    env = os.environ.get("PUREMACRO_BUDGET", "").strip().lower()
    if env:
        if env not in TIERS:
            raise ValueError(
                f"PUREMACRO_BUDGET={env!r} is not one of {sorted(TIERS)}"
            )
        return env
    caps = capabilities()
    if caps.device in ("tablet", "browser"):
        if caps.memory_mb is not None and caps.memory_mb < _MEMORY_MINIMAL_MB:
            return "minimal"
        return "tablet"
    return "workstation"


def current() -> Budget:
    """The budget in force right now.

    Derived from :func:`puremacro.runtime._capabilities`, unless pinned by
    ``PUREMACRO_BUDGET`` or by an active :func:`override` block.
    """
    if _FORCED is not None:
        return TIERS[_FORCED]
    return TIERS[_tier_for_device()]


@contextmanager
def override(tier: str):
    """Pin the budget tier inside a ``with`` block.

    Chiefly for rehearsing on a laptop what a notebook will do on the
    iPad::

        with runtime.override("tablet"):
            run_the_notebook()
    """
    global _FORCED
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r} is not one of {sorted(TIERS)}")
    previous, _FORCED = _FORCED, tier
    try:
        yield TIERS[tier]
    finally:
        _FORCED = previous


def _clamp(name: str, value, budget: Budget, *, where: str) -> tuple[object, bool]:
    """Clamp one keyword. Returns ``(value, was_clamped)``."""
    field = _ALIASES.get(name)
    if field is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return value, False
    cap = budget.cap(field)
    if value <= cap:
        return value, False
    key = (where, name, budget.tier)
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"{where}: {name}={value:g} exceeds the {budget.tier} budget "
            f"({field} cap {cap}); running with {name}={cap}. Raise it with "
            f"puremacro.runtime.override('workstation') or PUREMACRO_BUDGET.",
            BudgetWarning,
            stacklevel=3,
        )
    return type(value)(cap), True


def fit(**kwargs):
    """Clamp workload keywords to the current budget.

    >>> with override("tablet"):
    ...     fit(n_boot=2000, p=4)          # doctest: +SKIP
    {'n_boot': 400, 'p': 4}

    Unrecognised keywords pass through untouched, so this is safe to
    splat over a whole call's kwargs. Warns once per keyword per tier.
    """
    budget = current()
    return {
        name: _clamp(name, value, budget, where="runtime.fit")[0]
        for name, value in kwargs.items()
    }


def budgeted(func=None, *, tier: str | None = None):
    """Wrap ``func`` so its cost keywords are clamped to the budget.

    Works as a decorator or a one-shot wrapper::

        svar = runtime.budgeted(cholesky_svar)
        res = svar(Y, p=2, n_boot=5000)

    Positional arguments are clamped too — the signature is bound, so
    ``lp_hac(df, y, x, 500)`` is handled the same as the keyword form.
    Functions whose signature cannot be introspected (C builtins) fall
    back to keyword-only clamping.

    Parameters
    ----------
    tier : str, optional
        Pin a tier for this wrapper instead of detecting one.
    """
    def decorate(fn):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            sig = None
        where = getattr(fn, "__qualname__", repr(fn))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            budget = TIERS[tier] if tier is not None else current()
            if sig is None:
                clamped = {
                    k: _clamp(k, v, budget, where=where)[0]
                    for k, v in kwargs.items()
                }
                return fn(*args, **clamped)
            bound = sig.bind_partial(*args, **kwargs)
            for name, value in list(bound.arguments.items()):
                new, was = _clamp(name, value, budget, where=where)
                if was:
                    bound.arguments[name] = new
            return fn(*bound.args, **bound.kwargs)

        wrapper.__budget_wrapped__ = True  # type: ignore[attr-defined]
        return wrapper

    if func is None:
        return decorate
    return decorate(func)
