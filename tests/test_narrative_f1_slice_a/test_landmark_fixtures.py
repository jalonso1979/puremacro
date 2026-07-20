"""F1 Slice A — each connector's golden fixture passes its
landmark check. Regression guard for upstream layout drift."""
from __future__ import annotations

import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbe", "cbn", "cbk"]

# NOTE: per the path-fix observed in T2 (commit 279149f), the fixture
# directory resolves with a SINGLE puremacro prefix from .parent.parent.parent
# of the test file (which sits at puremacro/tests/test_narrative_f1_slice_a/).
_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _fixture_text(cb: str, kind: str) -> str:
    """Return the bytes of <cb>_<kind>_v1.html, .xml, or .json, whichever exists."""
    for ext in ("html", "xml", "json"):
        p = _FIXTURE_DIR / f"{cb}_{kind}_v1.{ext}"
        if p.exists():
            return p.read_text()
    return ""   # speeches fixture may be intentionally absent


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_decision_fixture_passes_landmark_check(cb):
    from puremacro.narrative.sources._schema_check import assert_landmarks
    mod = importlib.import_module(f"puremacro.narrative.sources.{cb}")
    text = _fixture_text(cb, "decision")
    assert text, f"missing decision fixture for {cb!r}"
    landmarks = mod._DECISION_LANDMARKS
    # Should not raise.
    assert_landmarks(
        text, source=cb,
        expected_version=mod.PARSER_SCHEMA_VERSION,
        landmarks=landmarks,
    )


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_speeches_fixture_passes_landmark_check_if_present(cb):
    """If iter_<cb>_speeches exists AND a fixture exists, the fixture
    passes the landmark check. Skipped if either is absent."""
    from puremacro.narrative.sources._schema_check import assert_landmarks
    mod = importlib.import_module(f"puremacro.narrative.sources.{cb}")
    iter_name = f"iter_{cb}_speeches"
    if not hasattr(mod, iter_name):
        pytest.skip(f"{cb}: no iter_{cb}_speeches (CB has no English speeches archive)")
    text = _fixture_text(cb, "speeches")
    if not text:
        pytest.skip(f"{cb}: no speeches fixture (function exists but fixture not generated)")
    landmarks = mod._SPEECHES_LANDMARKS
    assert_landmarks(
        text, source=cb,
        expected_version=mod.PARSER_SCHEMA_VERSION,
        landmarks=landmarks,
    )
