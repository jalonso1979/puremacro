"""Shared failure descriptions and cooperative run controls; no solver dispatch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Event
from time import monotonic
from typing import Any, Callable

import numpy as np

from .build import ModelError, SteadyStateError
from .klein import BlanchardKahnError


@dataclass(frozen=True)
class DSGEFailure:
    """A failure category, its original message and a suggested diagnostic."""

    category: str
    exception_type: str
    message: str
    next_step: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class _Cancelled(Exception):
    pass


class _BudgetExpired(Exception):
    pass


def describe_failure(error: Exception) -> DSGEFailure:
    """Classify known errors without guessing that every model error is BK."""
    name = type(error).__name__
    if isinstance(error, _Cancelled):
        category, action = "cancelled", "Inspect the retained checkpoint before starting a new run."
    elif isinstance(error, (_BudgetExpired, TimeoutError)):
        category, action = "budget", "Inspect progress and the checkpoint; increase the budget explicitly if appropriate."
    elif isinstance(error, BlanchardKahnError):
        category, action = "determinacy", "Inspect variable timing, the eigenvalues and the policy-rule parameters."
    elif isinstance(error, SteadyStateError):
        category, action = "steady_state", "Inspect named steady-state residuals, initial guesses and admissible values."
    elif name in {"ModCompatibilityError", "DynareFeatureError"}:
        category, action = "compatibility", "Inspect the source locations and unsupported commands before solving."
    elif name == "HANKContinuationAccuracyError":
        category, action = "refinement", "Inspect equation, horizon and branching checks separately in the retained result."
    elif name == "StackedNewtonError":
        category, action = "numerical_solver", "Inspect the last residual, line search and linear-solver history."
    elif isinstance(error, (FloatingPointError, OverflowError, ZeroDivisionError)):
        category, action = "domain", "Inspect equation inputs for nonfinite values and domain violations."
    elif isinstance(error, ModelError):
        category, action = "model", "Inspect the reported equation, declaration or derivative check."
    elif isinstance(error, (ValueError, TypeError, KeyError)):
        category, action = "input", "Check the named inputs, dimensions, units and supported options."
    else:
        category, action = "execution", "Inspect the original exception and the recorded inputs; no alternative solution was substituted."
    return DSGEFailure(category, name, str(error), action)


class DSGERunControl:
    """Cooperative cancellation and optional time budget for one run.

    Krylov solves check at accepted steps and existing inner progress hooks.
    Other operations check at stage boundaries. This is not a hard deadline
    or a background execution service. Use a fresh control for every run.
    """

    def __init__(self, *, max_seconds: float | None = None,
                 on_progress: Callable[[dict[str, Any]], None] | None = None):
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be callable")
        self.max_seconds = max_seconds
        self.on_progress = on_progress
        self._cancelled = Event()
        self._started: float | None = None
        self._events: list[dict[str, Any]] = []
        self._checkpoint: np.ndarray | None = None
        self._checkpoint_record: dict[str, Any] | None = None

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def events(self) -> list[dict[str, Any]]:
        from copy import deepcopy
        return deepcopy(self._events)

    def _start(self) -> None:
        if self._started is not None:
            raise ValueError("A DSGERunControl belongs to one run; create a fresh control")
        self._started = monotonic()
        self._check()

    def _check(self) -> None:
        if self._cancelled.is_set():
            raise _Cancelled("Cancellation requested; stopped at a cooperative checkpoint")
        if self._started is not None and self.max_seconds is not None:
            if monotonic() - self._started >= self.max_seconds:
                raise _BudgetExpired("Run time budget exhausted at a cooperative checkpoint")

    def _progress(self, record: dict[str, Any]) -> None:
        from copy import deepcopy
        event = deepcopy(record)
        event["elapsed_seconds"] = 0. if self._started is None else monotonic() - self._started
        self._events.append(event)
        if self.on_progress is not None:
            self.on_progress(deepcopy(event))
        # Accepted Newton steps immediately call _save_checkpoint next. Defer
        # cancellation there so the most recently accepted iterate survives.
        if event.get("phase") is not None:
            self._check()

    def _save_checkpoint(self, values: np.ndarray, record: dict[str, Any]) -> None:
        from copy import deepcopy
        self._checkpoint = np.array(values, dtype=float, copy=True)
        self._checkpoint_record = deepcopy(record)
        self._check()
