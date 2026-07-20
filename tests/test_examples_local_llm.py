"""Smoke: the local-LLM example runs end-to-end on the Mock fallback.

The no-engine path is forced by monkeypatching ``resolve_engine`` (same
pattern as test_local_backends): on a machine with a real engine installed
(e.g. mlx-lm on Apple Silicon), the example would otherwise load — and on
first use download — a real model, which is a live-network action this
offline smoke test must never take.
"""
import importlib

import puremacro.narrative._local_engines as le


def test_example_main_runs(capsys, monkeypatch):
    monkeypatch.setattr(
        le, "resolve_engine",
        lambda *a, **k: (_ for _ in ()).throw(le.LocalLLMUnavailable("forced offline")),
    )
    mod = importlib.import_module("puremacro.examples.narrative_local_llm")
    mod.main()  # engine resolution raises -> get_default_* falls back to Mock
    out = capsys.readouterr().out
    assert "MockBackend" in out or "MockProvider" in out
