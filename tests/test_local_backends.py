# tests/test_local_backends.py
"""LocalBackend/LocalProvider wiring at the two call sites, via a fake engine."""
import pytest

from puremacro.narrative import _local_engines as le
from puremacro.narrative.scoring import (
    LocalBackend, OllamaBackend, MockBackend, get_default_backend, score_llm,
)


class _FakeEngine:
    def __init__(self, name="fake", response="[]"):
        self.name = name
        self.response = response
        self.calls = []

    def available(self):
        return True

    def complete(self, model, prompt, *, max_tokens, temperature, json_mode):
        self.calls.append({"model": model, "prompt": prompt,
                           "max_tokens": max_tokens, "json_mode": json_mode})
        return self.response


def test_local_backend_calls_engine(monkeypatch):
    fake = _FakeEngine(name="mlx", response='[]')
    monkeypatch.setattr(le, "resolve_engine", lambda *a, **k: fake)
    be = LocalBackend("qwen2.5-3b-instruct", engine="auto")
    assert be.call("hello") == "[]"
    # canonical name resolved to the per-engine (mlx) id, json_mode on:
    assert fake.calls[0]["model"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert fake.calls[0]["json_mode"] is True


def test_mock_backend_returns_empty_array():
    assert MockBackend().call("anything") == "[]"


def test_get_default_backend_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(
        le, "resolve_engine",
        lambda *a, **k: (_ for _ in ()).throw(le.LocalLLMUnavailable("none")),
    )
    be = get_default_backend()
    assert isinstance(be, MockBackend)


def test_score_llm_runs_with_mock_backend():
    # MockBackend -> "[]" -> zero events, but the loop completes cleanly.
    out = score_llm([("2020-01-01", "fiscal stimulus announced", "http://x")],
                    backend=MockBackend(), kind="fiscal")
    assert out == []


class _DownBackend:
    model = "x"
    max_tokens = 8
    temperature = 0.0

    def call(self, prompt):
        raise le.LocalLLMUnavailable("server down")


class _GarbageBackend:
    model = "x"
    max_tokens = 8
    temperature = 0.0

    def call(self, prompt):
        return "not json at all"


def test_score_llm_propagates_backend_unavailable():
    with pytest.raises(le.BackendUnavailable):
        score_llm([("2020-01-01", "text", "http://x")],
                  backend=_DownBackend(), kind="fiscal")


def test_score_llm_still_drops_parse_errors():
    # Garbage output is dropped (not raised); loop returns [] cleanly.
    out = score_llm([("2020-01-01", "text", "http://x")],
                    backend=_GarbageBackend(), kind="fiscal")
    assert out == []


from puremacro.narrative.indices import (
    LocalProvider, OllamaProvider, get_default_provider, MockProvider,
)


def test_local_provider_parses_float(monkeypatch):
    fake = _FakeEngine(name="ollama", response="The score is 0.73.")
    monkeypatch.setattr(le, "resolve_engine", lambda *a, **k: fake)
    p = LocalProvider("qwen2.5-3b-instruct", engine="ollama")
    assert p.name == "ollama"                      # cache partitions per engine
    assert abs(p.score_paragraph("very uncertain outlook", "uncertainty")
               - 0.73) < 1e-9


def test_local_provider_clamps_and_defaults(monkeypatch):
    monkeypatch.setattr(le, "resolve_engine",
                        lambda *a, **k: _FakeEngine(response="no number here"))
    p = LocalProvider(engine="ollama")
    assert p.score_paragraph("x", "uncertainty") == 0.0


def test_local_provider_clamps_out_of_range(monkeypatch):
    # A reply above 1.0 is clamped to 1.0 (the min() guard).
    monkeypatch.setattr(le, "resolve_engine",
                        lambda *a, **k: _FakeEngine(response="1.8"))
    p = LocalProvider(engine="ollama")
    assert p.score_paragraph("x", "uncertainty") == 1.0


def test_get_default_provider_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(
        le, "resolve_engine",
        lambda *a, **k: (_ for _ in ()).throw(le.LocalLLMUnavailable("none")),
    )
    assert isinstance(get_default_provider(), MockProvider)
