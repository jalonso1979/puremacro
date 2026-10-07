"""Every fenced ``python`` block on the front-door pages and the 2.3.0 feature
pages runs verbatim.

The 2.3.0 audit found that the first code block on four of the six new feature
pages, and three of the four README quickstart sections, raised on the first
statement (wrong keyword names, attributes that did not exist, a broken
import).  This test executes the blocks exactly as a reader would copy them:
in file order, sharing one namespace per page, from a scratch working
directory so that pages which write files do not touch the repository.

Blocks whose first line starts with ``# requires:`` (Google Colab, Pyodide, a
network connection, a local LLM engine) are skipped; a fenced block that is
not valid Python is a failure -- API listings belong in ```text fences.
"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]

PAGES = [
    "docs/sw07_estimator_experiment.md",
    "docs/es/sw07_estimator_experiment.md",
    "docs/distributional_trade_ge.md",
    "docs/es/distributional_trade_ge.md",
    "docs/sw07_finite_sample.md",
    "docs/es/sw07_finite_sample.md",
    "docs/empirical_research.md",
    "docs/es/empirical_research.md",
    "docs/research_workflows.md",
    "docs/research_benchmarks.md",
    "docs/structural_bridge.md",
    "docs/trade_distributional.md",
    "docs/es/research_workflows.md",
    "docs/es/research_benchmarks.md",
    "docs/es/structural_bridge.md",
    "docs/es/trade_distributional.md",
    "README.md",
    "docs/quickstart.md",
    "docs/narrative_sign_svar.md",
    "docs/honest_did.md",
    "docs/spatial.md",
    "docs/smooth_lp.md",
    "docs/hank_nonlinear.md",
    "docs/gertler_karadi.md",
    "docs/bvar_sv.md",
    "docs/gvar.md",
    "docs/dsge_estimation.md",
    "docs/did.md",
    "docs/es/narrative_sign_svar.md",
    "docs/es/honest_did.md",
    "docs/es/spatial.md",
    "docs/es/smooth_lp.md",
    "docs/es/hank_nonlinear.md",
    "docs/es/gertler_karadi.md",
    "docs/es/bvar_sv.md",
    "docs/es/gvar.md",
    "docs/es/dsge_estimation.md",
    "docs/es/did.md",
    # MRIO engines and the stacked Newton-Krylov (English and Spanish, identical code).
    "docs/trade_mrio.md",
    "docs/trade_condensed.md",
    "docs/trade_ces_newton.md",
    "docs/trade_household.md",
    "docs/trade_continuation.md",
    "docs/trade_stability.md",
    "docs/trade_dynamic.md",
    "docs/dsge_stacked_newton.md",
    "docs/es/trade_mrio.md",
    "docs/es/trade_condensed.md",
    "docs/es/trade_ces_newton.md",
    "docs/es/trade_household.md",
    "docs/es/trade_continuation.md",
    "docs/es/trade_stability.md",
    "docs/es/trade_dynamic.md",
    "docs/es/dsge_stacked_newton.md",
]

_FENCE = re.compile(r"```python\n(.*?)```", re.S)
_PER_BLOCK_SECONDS = 120


def _blocks(page: str) -> list[str]:
    return _FENCE.findall((REPO / page).read_text(encoding="utf-8"))


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # pragma: no cover - only fires on a hang
    raise _Timeout("code block exceeded the per-block time budget")


@contextlib.contextmanager
def _block_timeout(seconds: int):
    """SIGALRM watchdog where the platform has one; Windows has no SIGALRM, so
    there the block simply runs without a budget (CI's job timeout still applies)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("page", PAGES)
def test_page_code_blocks_run_verbatim(page, tmp_path, monkeypatch):
    blocks = _blocks(page)
    assert blocks, f"{page} has no python blocks"
    monkeypatch.chdir(tmp_path)          # pages that write files write here
    namespace: dict = {"__name__": "__doc__"}
    failures = []
    for i, block in enumerate(blocks):
        first = block.strip().splitlines()[0] if block.strip() else ""
        if first.startswith("# requires:"):
            continue
        try:
            code = compile(block, f"{page}:block{i}", "exec")
        except SyntaxError as exc:  # API listings must live in ```text fences
            failures.append(f"block {i}: not valid Python ({exc.msg} at line {exc.lineno})")
            continue
        try:
            with _block_timeout(_PER_BLOCK_SECONDS):
                exec(code, namespace)
        except _Timeout as exc:
            failures.append(f"block {i}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failures.append(f"block {i}: {type(exc).__name__}: {exc}")
        finally:
            # Close figures only when a block imported pyplot (pages without plots never do).
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is not None:
                pyplot.close("all")
    assert not failures, f"{page}:\n  " + "\n  ".join(failures)
