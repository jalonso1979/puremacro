"""Resumable, time-boxed estimation.

iPadOS suspends a backgrounded app. A 2,000-draw bootstrap that takes
four minutes does not survive someone answering a message halfway
through — and neither does the twenty minutes of a Krusell-Smith solve
or a Metropolis-Hastings chain. The tablet is not slow so much as
*interruptible*, and nothing in a normal estimator call is designed for
that: it is one opaque block of compute that either finishes or is lost.

A :class:`Job` breaks that block into chunks, persists after every one,
and can be told to work for thirty seconds and come back::

    job = longrun.chunked(one_draw, n_total=2000, checkpoint="svar.ckpt")
    job.run(seconds=30)      # 240/2000 · 12%
    job.run(seconds=30)      # 490/2000 · 24%   (after an app suspend, too)
    irfs = job.result()      # (2000, 21, 3, 3)

Determinism does not depend on how the work was sliced. Draw *i* always
uses ``np.random.default_rng([seed, i])``, so the same ``seed`` gives
bit-identical output whether it ran in one call or forty, on a laptop or
a tablet. That is what makes a resumed run publishable rather than
merely finished.

The checkpoint is a plain npz (numpy only — no pyarrow, no pickle), so
it reads back on any machine; a job started on the iPad can be finished
on the workstation.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CheckpointError",
    "Progress",
    "Job",
    "chunked",
    "bootstrap",
]


class CheckpointError(RuntimeError):
    """A checkpoint is unreadable, or does not match the job resuming it."""


@dataclass(frozen=True)
class Progress:
    """How far along a job is.

    Attributes
    ----------
    done, total : int
        Items completed and requested.
    elapsed : float
        Seconds of compute spent across every :meth:`Job.run` call,
        accumulated in the checkpoint (so it survives a restart).
    last_run : float
        Seconds spent in the most recent :meth:`Job.run`.
    rate : float
        Items per second over the whole job; 0.0 before any work.
    eta : float | None
        Seconds of compute remaining at the current rate, None when
        unknown or finished.
    """

    done: int
    total: int
    elapsed: float
    last_run: float
    rate: float
    eta: float | None

    @property
    def pct(self) -> float:
        """Percent complete."""
        return 100.0 * self.done / self.total if self.total else 100.0

    @property
    def finished(self) -> bool:
        return self.done >= self.total

    def __str__(self) -> str:
        head = f"{self.done}/{self.total} · {self.pct:.0f}%"
        if self.finished:
            return f"{head} · done in {self.elapsed:.1f}s"
        if self.eta is None:
            return head
        return f"{head} · ~{self.eta:.0f}s of compute left"


def _fingerprint(name: str, n_total: int, seed: int, fn) -> str:
    """Identity of a job, so a checkpoint cannot be resumed by the wrong one."""
    ident = json.dumps({
        "name": name,
        "n_total": int(n_total),
        "seed": int(seed),
        "fn": getattr(fn, "__qualname__", repr(fn)),
    }, sort_keys=True)
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16]


class Job:
    """A chunked, checkpointed computation over ``n_total`` independent items.

    Build one with :func:`chunked` rather than directly.

    Parameters
    ----------
    fn : callable
        ``fn(i, rng) -> scalar | ndarray``, computing item ``i``. Must be
        pure: it is called exactly once per item, but possibly in a
        different process run than its neighbours.
    n_total : int
        Number of items.
    chunk : int
        Items per checkpoint write. Smaller means less lost work on an
        interruption and more I/O; the default (25) is tuned for
        bootstrap draws of a few hundred milliseconds each.
    checkpoint : str | Path | None
        Where to persist. None keeps the job in memory only — useful in
        tests, useless against a suspend.
    seed : int
        Base seed. Item ``i`` draws from ``default_rng([seed, i])``.
    name : str
        Label, recorded in the checkpoint and used in messages.
    """

    def __init__(self, fn, n_total: int, *, chunk: int = 25,
                 checkpoint=None, seed: int = 0, name: str = "job"):
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")
        if chunk <= 0:
            raise ValueError(f"chunk must be positive, got {chunk}")
        self.fn = fn
        self.n_total = int(n_total)
        self.chunk = int(chunk)
        self.seed = int(seed)
        self.name = name
        self.path = Path(checkpoint) if checkpoint is not None else None
        self._fp = _fingerprint(name, n_total, seed, fn)

        self._results: np.ndarray | None = None
        self._done = np.zeros(self.n_total, dtype=bool)
        self._elapsed = 0.0
        self._last_run = 0.0

        if self.path is not None and self.path.exists():
            self._load(self.path)

    # -- persistence ---------------------------------------------------

    def _load(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as ckpt:
                meta = json.loads(bytes(ckpt["meta"]).decode("utf-8"))
                if meta.get("fingerprint") != self._fp:
                    raise CheckpointError(
                        f"{path} belongs to a different job "
                        f"({meta.get('name')!r}, n_total={meta.get('n_total')}, "
                        f"seed={meta.get('seed')}) than the one resuming it "
                        f"({self.name!r}, n_total={self.n_total}, "
                        f"seed={self.seed}). Delete the file or pass a "
                        f"different checkpoint= path."
                    )
                self._results = ckpt["results"]
                self._done = ckpt["done"].astype(bool)
                self._elapsed = float(meta.get("elapsed", 0.0))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                f"could not read checkpoint {path}: {exc}. Delete it to "
                f"start over."
            ) from exc

    def _save(self) -> None:
        if self.path is None or self._results is None:
            return
        meta = json.dumps({
            "fingerprint": self._fp,
            "name": self.name,
            "n_total": self.n_total,
            "seed": self.seed,
            "chunk": self.chunk,
            "elapsed": self._elapsed,
            "done": int(self._done.sum()),
        }).encode("utf-8")
        # Write beside the target and rename: a checkpoint half-written
        # when the OS suspends the app is worse than no checkpoint.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            np.savez(fh, results=self._results, done=self._done,
                     meta=np.frombuffer(meta, dtype=np.uint8))
        os.replace(tmp, self.path)

    # -- execution -----------------------------------------------------

    def _allocate(self, first) -> np.ndarray:
        sample = np.asarray(first, dtype=float)
        self._results = np.full((self.n_total, *sample.shape), np.nan)
        return self._results

    def _compute(self, i: int):
        return self.fn(i, np.random.default_rng([self.seed, i]))

    def run(self, *, seconds: float | None = None,
            chunks: int | None = None, progress: bool = False) -> Progress:
        """Do some of the work, persist, and return.

        With neither argument, runs to completion. The current chunk is
        always finished, so a ``seconds`` budget is a floor on the work
        done, not a ceiling on the time taken.

        Parameters
        ----------
        seconds : float, optional
            Stop starting new chunks once this much time has passed.
        chunks : int, optional
            Stop after this many chunks.
        progress : bool, default False
            Print a line per chunk.

        Returns
        -------
        Progress
        """
        started = time.monotonic()
        chunks_done = 0
        pending = np.flatnonzero(~self._done)

        for start in range(0, len(pending), self.chunk):
            if seconds is not None and time.monotonic() - started >= seconds:
                break
            if chunks is not None and chunks_done >= chunks:
                break
            chunk_started = time.monotonic()
            batch = pending[start:start + self.chunk]
            for i in batch:
                value = self._compute(int(i))
                results = self._results
                if results is None:
                    results = self._allocate(value)
                arr = np.asarray(value, dtype=float)
                if arr.shape != results.shape[1:]:
                    raise ValueError(
                        f"{self.name}: item {int(i)} returned shape "
                        f"{arr.shape}, but earlier items returned "
                        f"{results.shape[1:]}. Every item must have the "
                        f"same shape."
                    )
                results[int(i)] = arr
                self._done[int(i)] = True
            chunks_done += 1
            self._elapsed += time.monotonic() - chunk_started
            self._save()
            if progress:
                print(self.progress, flush=True)

        self._last_run = time.monotonic() - started
        return self.progress

    # -- inspection ----------------------------------------------------

    @property
    def progress(self) -> Progress:
        done = int(self._done.sum())
        rate = done / self._elapsed if self._elapsed > 0 else 0.0
        eta = None
        if rate > 0 and done < self.n_total:
            eta = (self.n_total - done) / rate
        return Progress(done=done, total=self.n_total, elapsed=self._elapsed,
                        last_run=self._last_run, rate=rate, eta=eta)

    @property
    def finished(self) -> bool:
        return bool(self._done.all())

    def result(self, *, allow_partial: bool = False) -> np.ndarray:
        """The stacked results, shape ``(n_total, *item_shape)``.

        Raises unless the job is finished, so a half-run bootstrap can
        never be mistaken for a full one.

        Parameters
        ----------
        allow_partial : bool, default False
            Return what exists, with NaN rows for the items not yet
            computed.
        """
        if self._results is None:
            raise CheckpointError(
                f"{self.name}: nothing computed yet — call run() first"
            )
        if not self.finished and not allow_partial:
            p = self.progress
            raise CheckpointError(
                f"{self.name} is {p.done}/{p.total} complete. Call run() "
                f"again, or result(allow_partial=True) to accept NaN rows."
            )
        return self._results

    def reset(self) -> None:
        """Discard all progress and delete the checkpoint file."""
        self._results = None
        self._done[:] = False
        self._elapsed = 0.0
        if self.path is not None and self.path.exists():
            self.path.unlink()

    def __repr__(self) -> str:
        return f"<Job {self.name!r} {self.progress}>"


def chunked(fn, n_total: int, *, chunk: int = 25, checkpoint=None,
            seed: int = 0, name: str = "job") -> Job:
    """Build a resumable :class:`Job` over ``n_total`` independent items.

    ``fn(i, rng)`` computes item ``i``; ``rng`` is seeded from
    ``[seed, i]``, so results do not depend on chunking or on how many
    sessions the job took.

    >>> job = chunked(lambda i, rng: rng.standard_normal(3), 100)
    >>> job.run().finished
    True
    >>> job.result().shape
    (100, 3)
    """
    return Job(fn, n_total, chunk=chunk, checkpoint=checkpoint, seed=seed,
               name=name)


def bootstrap(draw, n_boot: int, *, chunk: int = 25, checkpoint=None,
              seed: int = 0, name: str = "bootstrap") -> Job:
    """A resumable bootstrap: ``draw(rng)`` returns one replication.

    Thin wrapper over :func:`chunked` for the common case where the
    replication index is not needed::

        from puremacro.var.estimate import estimate_var
        from puremacro.var.irf import irf as var_irf

        def one(rng):
            Yb = Y[rng.integers(0, len(Y), len(Y))]
            fit = estimate_var(Yb, p=2)
            return var_irf(fit.A_list, np.linalg.cholesky(fit.Sigma), horizon=20)

        job = longrun.bootstrap(one, 2000, checkpoint="irf.ckpt")
        job.run(seconds=45)
        bands = np.percentile(job.result(), [5, 95], axis=0)
    """
    return chunked(lambda i, rng: draw(rng), n_boot, chunk=chunk,
                   checkpoint=checkpoint, seed=seed, name=name)
