"""F1 Slice A — each connector's iter_<cb>_decision yields ≥1 tuple
when run against its golden fixture."""
from __future__ import annotations

import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbn", "cbe", "cbk"]

# Per the T2 path-fix: SINGLE puremacro prefix from .parent.parent.parent.
_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _decision_fixture_text(cb: str) -> str:
    for ext in ("html", "xml", "json"):
        p = _FIXTURE_DIR / f"{cb}_decision_v1.{ext}"
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no decision fixture for {cb!r}")


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_decision_fixture_yields_at_least_one(cb, monkeypatch):
    from puremacro.narrative.sources import _fallback
    mod = importlib.import_module(f"puremacro.narrative.sources.{cb}")
    text = _decision_fixture_text(cb)
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: text,
    )
    iter_fn = getattr(mod, f"iter_{cb}_decision")
    records = list(iter_fn())
    assert len(records) >= 1, (
        f"{cb}.iter_{cb}_decision yielded {len(records)} records "
        f"against fixture; expected ≥1."
    )
    for r in records:
        assert isinstance(r, tuple)
        assert len(r) in (3, 4)
