"""Resumable estimation for machines that get interrupted.

Long-running work — bootstrap bands, posterior chains, value-function
iteration — normally has to finish in one go. On a tablet it often
can't: iPadOS suspends a backgrounded app, and the whole run is lost.

:func:`chunked` and :func:`bootstrap` split the work into checkpointed
chunks that can be run a bit at a time and resumed in a later session,
with results identical to an uninterrupted run.

    >>> from puremacro import longrun
    >>> job = longrun.bootstrap(one_draw, 2000,
    ...                            checkpoint="irf.ckpt")   # doctest: +SKIP
    >>> job.run(seconds=30)                                 # doctest: +SKIP
    240/2000 · 12% · ~220s of compute left
"""
from puremacro.longrun._job import (
    CheckpointError,
    Job,
    Progress,
    bootstrap,
    chunked,
)

__all__ = ["Job", "Progress", "CheckpointError", "chunked", "bootstrap"]
