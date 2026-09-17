"""Serial-or-threaded evaluation of independent bootstrap replications.

Every bootstrap engine in the package —
:func:`~puremacro.inference.wild_bootstrap.wild_bootstrap`,
:func:`~puremacro.inference.wild_bootstrap.wild_bootstrap_var`,
:func:`~puremacro.inference.block_bootstrap.block_bootstrap`,
:func:`~puremacro.inference.lp_block_bootstrap.cum_irf_block_bootstrap` and
:func:`~puremacro.var.bootstrap.bootstrap_bands` — draws all of its random
material up front with the caller's generator and then maps a pure
per-replication function over it. Because the randomness is fixed before
any worker runs and ``Executor.map`` preserves input order, the thread
pool and the plain loop produce identical output; this module only
decides which of the two runs, and keeps the rules in one place:

* ``n_jobs == 0`` is rejected with ``ValueError`` before anything runs
  (3.3.0 raised the same error from ``ThreadPoolExecutor``); any negative
  ``n_jobs`` means every CPU core; ``n_jobs == 1`` is the plain loop.
* When :func:`puremacro.runtime.capabilities` reports no working OS
  threads — a Pyodide/WASM kernel, or ``PUREMACRO_THREADS=0`` — the loop
  runs and no executor is constructed.
* If the executor itself cannot be created (``RuntimeError``, ``OSError``
  or ``ValueError`` out of ``ThreadPoolExecutor``), a ``RuntimeWarning``
  says so and the loop runs once instead.
* An exception raised by the per-replication function is never caught
  here. It propagates unchanged from either path, so a failing
  ``refit_fn`` is reported rather than silently re-run serially.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Iterable, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def _can_use_threads() -> bool:
    """True when the runtime reports working OS threads.

    Goes through :func:`puremacro.runtime.capabilities`, so the
    ``PUREMACRO_THREADS`` override is honoured. Any failure to answer is
    read as "no threads", the safe direction.
    """
    try:
        from puremacro.runtime import capabilities

        return bool(capabilities().threads)
    except (ValueError, ArithmeticError, Exception):
        return False


def _resolve_workers(n_jobs: int) -> int:
    """Worker count for ``n_jobs``: negative means every CPU core, zero is an error."""
    if n_jobs == 0:
        raise ValueError(
            "n_jobs must be a positive integer, or -1 for all CPU cores; got 0"
        )
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def _map_draws(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    n_jobs: int,
    *,
    label: str,
) -> list[_R]:
    """``[fn(item) for item in items]``, serially or on a thread pool.

    Parameters
    ----------
    fn
        Pure per-replication function. Whatever it raises propagates.
    items
        Pre-drawn per-replication inputs: rows of a weight matrix, index
        arrays, or plain replication numbers.
    n_jobs
        ``1`` runs the loop, ``-1`` uses every CPU core, ``0`` raises
        ``ValueError``.
    label
        Name of the calling engine, for the fallback warning.

    Returns
    -------
    list
        Results in input order, identical whichever path ran.
    """
    workers = _resolve_workers(n_jobs)
    if workers == 1 or not _can_use_threads():
        return [fn(item) for item in items]

    import concurrent.futures

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    except (RuntimeError, OSError, ValueError) as exc:
        warnings.warn(
            f"{label}: could not start a thread pool ({exc}); evaluating the "
            f"bootstrap replications serially instead of with n_jobs={n_jobs}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return [fn(item) for item in items]
    with executor:
        return list(executor.map(fn, items))
