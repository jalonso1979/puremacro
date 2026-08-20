"""Tests for puremacro.longrun — chunked, checkpointed, resumable jobs.

The property that matters is that *how* the work was sliced never shows
up in the answer: same seed, same result, whether it ran in one call, in
forty, or across two sessions with the process restarted in between.
"""
import time

import numpy as np
import pytest

from puremacro import longrun


def _draw(i, rng):
    return rng.standard_normal(3)


@pytest.mark.pyodide_smoke
def test_results_are_invariant_to_chunk_size():
    one_chunk = longrun.chunked(_draw, 60, chunk=60, name="j").run()
    many_chunks = longrun.chunked(_draw, 60, chunk=7, name="j")
    many_chunks.run()
    a = longrun.chunked(_draw, 60, chunk=60, name="j")
    a.run()
    np.testing.assert_array_equal(a.result(), many_chunks.result())
    assert one_chunk.finished


def test_resuming_in_a_new_session_gives_identical_results(tmp_path):
    ckpt = tmp_path / "job.ckpt"
    reference = longrun.chunked(_draw, 50, chunk=50, name="j")
    reference.run()

    first = longrun.chunked(_draw, 50, chunk=5, checkpoint=ckpt, name="j")
    first.run(chunks=3)
    assert first.progress.done == 15
    assert not first.finished

    # A fresh Job object, as a new process would build.
    second = longrun.chunked(_draw, 50, chunk=5, checkpoint=ckpt, name="j")
    assert second.progress.done == 15, "checkpoint was not picked up"
    second.run()
    np.testing.assert_array_equal(second.result(), reference.result())


def test_seed_changes_results_but_stays_reproducible():
    a = longrun.chunked(_draw, 20, seed=0, name="j"); a.run()
    b = longrun.chunked(_draw, 20, seed=1, name="j"); b.run()
    c = longrun.chunked(_draw, 20, seed=0, name="j"); c.run()
    assert not np.allclose(a.result(), b.result())
    np.testing.assert_array_equal(a.result(), c.result())


def test_partial_results_are_not_silently_returned():
    job = longrun.chunked(_draw, 20, chunk=5, name="j")
    job.run(chunks=1)
    with pytest.raises(longrun.CheckpointError, match="5/20"):
        job.result()
    partial = job.result(allow_partial=True)
    assert np.isnan(partial[5:]).all()
    assert not np.isnan(partial[:5]).any()


def test_result_before_any_work_is_an_error():
    job = longrun.chunked(_draw, 10, name="j")
    with pytest.raises(longrun.CheckpointError, match="nothing computed"):
        job.result()


def test_checkpoint_from_a_different_job_is_refused(tmp_path):
    ckpt = tmp_path / "job.ckpt"
    longrun.chunked(_draw, 50, chunk=5, checkpoint=ckpt, name="j").run(chunks=1)
    with pytest.raises(longrun.CheckpointError, match="different job"):
        longrun.chunked(_draw, 99, chunk=5, checkpoint=ckpt, name="j")
    with pytest.raises(longrun.CheckpointError, match="different job"):
        longrun.chunked(_draw, 50, chunk=5, checkpoint=ckpt, name="other")


def test_unreadable_checkpoint_says_how_to_recover(tmp_path):
    ckpt = tmp_path / "job.ckpt"
    ckpt.write_bytes(b"garbage")
    with pytest.raises(longrun.CheckpointError, match="Delete it"):
        longrun.chunked(_draw, 10, checkpoint=ckpt, name="j")


def test_time_box_stops_starting_new_chunks():
    def slow(i, rng):
        time.sleep(0.005)
        return rng.standard_normal()

    job = longrun.chunked(slow, 400, chunk=10, name="slow")
    progress = job.run(seconds=0.2)
    assert 0 < progress.done < 400
    assert progress.done % 10 == 0, "a chunk was left half-finished"


def test_chunks_argument_bounds_the_work():
    job = longrun.chunked(_draw, 100, chunk=8, name="j")
    job.run(chunks=2)
    assert job.progress.done == 16


def test_inconsistent_result_shape_is_rejected():
    job = longrun.chunked(
        lambda i, rng: np.zeros(3 if i < 5 else 4), 10, chunk=10, name="j",
    )
    with pytest.raises(ValueError, match="same shape"):
        job.run()


def test_scalar_results_are_supported():
    job = longrun.chunked(lambda i, rng: float(i), 10, name="j")
    job.run()
    np.testing.assert_array_equal(job.result(), np.arange(10.0))


def test_bootstrap_wrapper_ignores_the_index():
    calls = []

    def one(rng):
        value = rng.standard_normal()
        calls.append(value)
        return value

    job = longrun.bootstrap(one, 25, chunk=5, name="boot")
    job.run()
    assert job.result().shape == (25,)
    assert len(calls) == 25


def test_reset_clears_progress_and_file(tmp_path):
    ckpt = tmp_path / "job.ckpt"
    job = longrun.chunked(_draw, 20, chunk=5, checkpoint=ckpt, name="j")
    job.run(chunks=1)
    assert ckpt.exists()
    job.reset()
    assert not ckpt.exists()
    assert job.progress.done == 0


def test_progress_reporting():
    job = longrun.chunked(_draw, 40, chunk=10, name="j")
    job.run(chunks=1)
    p = job.progress
    assert p.done == 10 and p.total == 40 and p.pct == 25.0
    assert not p.finished
    assert p.eta is None or p.eta > 0
    assert "10/40" in str(p)
    job.run()
    assert job.progress.finished
    assert "done in" in str(job.progress)


def test_invalid_construction():
    with pytest.raises(ValueError, match="n_total"):
        longrun.chunked(_draw, 0)
    with pytest.raises(ValueError, match="chunk"):
        longrun.chunked(_draw, 10, chunk=0)


def test_checkpoint_is_written_atomically(tmp_path):
    """No .tmp file may survive a completed run."""
    ckpt = tmp_path / "job.ckpt"
    longrun.chunked(_draw, 20, chunk=5, checkpoint=ckpt, name="j").run()
    assert ckpt.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_loads_without_pickle(tmp_path):
    ckpt = tmp_path / "job.ckpt"
    longrun.chunked(_draw, 10, chunk=5, checkpoint=ckpt, name="j").run()
    with np.load(ckpt, allow_pickle=False) as archive:
        assert set(archive.files) == {"results", "done", "meta"}
