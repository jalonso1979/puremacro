# tests/test_local_llm_live.py
"""Opt-in live tests against real local engines. Each SKIPS if its engine
(or model) is unavailable, so CI without engines stays green.

Run locally after:  pip install -e ".[local-llm]"   (MLX needs Apple Silicon)
and, for the Ollama case, `ollama serve` + `ollama pull qwen2.5:3b`.
"""
import pytest

from puremacro.narrative import _local_engines as le

pytestmark = [pytest.mark.local_llm, pytest.mark.slow]

_CALM = "The committee kept policy unchanged; conditions were stable and as expected."
_UNCERTAIN = ("Officials warned of highly uncertain, unpredictable risks; the "
              "outlook is murky and could shift abruptly in either direction.")


def _engine_or_skip(name):
    eng = le._make_engine(name, base_url="http://localhost:11434", timeout=5)
    if not eng.available():
        pytest.skip(f"engine {name!r} not available")
    return eng


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp", "ollama"])
def test_local_provider_directional(engine_name):
    _engine_or_skip(engine_name)
    from puremacro.narrative.indices import LocalProvider
    try:
        p = LocalProvider("qwen2.5-3b-instruct", engine=engine_name)
    except le.LocalLLMUnavailable:
        pytest.skip(f"{engine_name} unusable")
    hi = p.score_paragraph(_UNCERTAIN, "economic uncertainty")
    lo = p.score_paragraph(_CALM, "economic uncertainty")
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi >= lo  # uncertain text scores at least as high as calm text


@pytest.mark.parametrize("engine_name", ["mlx", "llamacpp", "ollama"])
def test_local_backend_event_json_parses(engine_name):
    _engine_or_skip(engine_name)
    from puremacro.narrative.scoring import LocalBackend, score_llm
    try:
        be = LocalBackend("qwen2.5-3b-instruct", engine=engine_name)
    except le.LocalLLMUnavailable:
        pytest.skip(f"{engine_name} unusable")
    rec = ("2020-03-15",
           "The government announced a $500 billion infrastructure spending package.",
           "http://example.test")
    events = score_llm([rec], backend=be, kind="fiscal")
    # We assert structure, not exact extraction (small models vary):
    assert isinstance(events, list)
    for ev in events:
        assert ev.kind == "fiscal"
        assert ev.sign in (-1, 0, 1)
