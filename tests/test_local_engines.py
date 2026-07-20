# tests/test_local_engines.py
"""Local inference engine layer: aliases, resolution, engines, errors."""
import http.server
import json
import sys
import threading
import types

import pytest

from puremacro.narrative import _local_engines as le


def test_backend_unavailable_is_base_of_local_unavailable():
    assert issubclass(le.LocalLLMUnavailable, le.BackendUnavailable)


def test_resolve_model_id_known_alias_per_engine():
    assert le.resolve_model_id("qwen2.5-3b-instruct", "ollama") == "qwen2.5:3b"
    assert le.resolve_model_id("qwen2.5-3b-instruct", "mlx") == \
        "mlx-community/Qwen2.5-3B-Instruct-4bit"
    repo, fname = le.resolve_model_id("qwen2.5-3b-instruct", "llamacpp")
    assert repo == "Qwen/Qwen2.5-3B-Instruct-GGUF" and fname.endswith(".gguf")


def test_resolve_model_id_unknown_passes_through():
    assert le.resolve_model_id("my-custom-model", "ollama") == "my-custom-model"
    assert le.resolve_model_id("my-custom-model", "mlx") == "my-custom-model"


class _OllamaStub(http.server.BaseHTTPRequestHandler):
    """Mimics Ollama /api/chat + /api/tags and an OpenAI /v1 server."""
    chat_content = "[]"          # over/written per test via class attr
    score_content = "0.7"

    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "qwen2.5:3b"}]})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "qwen2.5-3b"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant",
                                    "content": type(self).chat_content}})
        elif self.path == "/v1/chat/completions":
            self._send({"choices": [{"message":
                        {"content": type(self).score_content}}]})
        else:
            self.send_error(404)


@pytest.fixture
def stub_server():
    _OllamaStub.chat_content = "[]"
    _OllamaStub.score_content = "0.7"
    srv = http.server.HTTPServer(("127.0.0.1", 0), _OllamaStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_http_engine_ollama_available_and_complete(stub_server):
    eng = le.HTTPEngine(base_url=stub_server, api="ollama", timeout=5)
    assert eng.name == "ollama"
    assert eng.available() is True
    _OllamaStub.chat_content = '[{"x": 1}]'
    out = eng.complete("qwen2.5:3b", "hi", max_tokens=16,
                       temperature=0.0, json_mode=True)
    assert out == '[{"x": 1}]'


def test_http_engine_openai_complete(stub_server):
    eng = le.HTTPEngine(base_url=stub_server, api="openai", timeout=5)
    assert eng.name == "openai"
    _OllamaStub.score_content = "0.42"
    out = eng.complete("qwen2.5-3b", "hi", max_tokens=8,
                       temperature=0.0, json_mode=False)
    assert out == "0.42"


def test_http_engine_unavailable_when_no_server():
    # Port 1 is never an Ollama server -> available() False, complete() raises.
    eng = le.HTTPEngine(base_url="http://127.0.0.1:1", api="ollama", timeout=1)
    assert eng.available() is False
    with pytest.raises(le.LocalLLMUnavailable):
        eng.complete("qwen2.5:3b", "hi", max_tokens=8,
                     temperature=0.0, json_mode=False)


def test_mlx_engine_complete_with_fake_module(monkeypatch):
    captured = {}

    class _Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            captured["messages"] = messages
            return "PROMPT:" + messages[-1]["content"]

    fake = types.ModuleType("mlx_lm")
    fake.load = lambda model_id: ("MODEL", _Tok())
    fake.generate = lambda model, tok, prompt, max_tokens=256, verbose=False: \
        f"gen({prompt}|{max_tokens})"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setattr(le.MLXEngine, "available", lambda self: True)

    eng = le.MLXEngine()
    assert eng.name == "mlx"
    out = eng.complete("mlx-community/Qwen2.5-3B-Instruct-4bit", "hello",
                       max_tokens=32, temperature=0.0, json_mode=True)
    assert out == "gen(PROMPT:hello|32)"
    assert captured["messages"][-1]["content"] == "hello"


def test_llamacpp_engine_complete_with_fake_module(monkeypatch):
    class _Llama:
        def __init__(self, **kw):
            self.kw = kw

        @classmethod
        def from_pretrained(cls, **kw):
            return cls(**kw)

        def create_chat_completion(self, messages, temperature=0.0,
                                   max_tokens=256, **kw):
            txt = messages[-1]["content"]
            return {"choices": [{"message": {"content": f"cc({txt})"}}]}

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setattr(le.LlamaCppEngine, "available", lambda self: True)

    eng = le.LlamaCppEngine()
    assert eng.name == "llamacpp"
    out = eng.complete(("Qwen/Qwen2.5-3B-Instruct-GGUF", "*Q4_K_M.gguf"),
                       "hi", max_tokens=16, temperature=0.0, json_mode=True)
    assert out == "cc(hi)"


def _force_available(monkeypatch, available_names):
    """Make each engine's available() report membership in available_names."""
    monkeypatch.setattr(le.MLXEngine, "available",
                        lambda self: "mlx" in available_names)
    monkeypatch.setattr(le.LlamaCppEngine, "available",
                        lambda self: "llamacpp" in available_names)
    monkeypatch.setattr(le.HTTPEngine, "available",
                        lambda self: "ollama" in available_names)


def test_resolve_auto_prefers_mlx_on_darwin(monkeypatch):
    monkeypatch.setattr(le.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(le.platform, "machine", lambda: "arm64")
    _force_available(monkeypatch, {"mlx", "llamacpp", "ollama"})
    assert le.resolve_engine("auto").name == "mlx"


def test_resolve_auto_falls_through_to_http(monkeypatch):
    monkeypatch.setattr(le.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(le.platform, "machine", lambda: "arm64")
    _force_available(monkeypatch, {"ollama"})
    assert le.resolve_engine("auto").name == "ollama"


def test_resolve_auto_none_raises(monkeypatch):
    monkeypatch.setattr(le.platform, "system", lambda: "Linux")
    monkeypatch.setattr(le.platform, "machine", lambda: "x86_64")
    _force_available(monkeypatch, set())
    with pytest.raises(le.LocalLLMUnavailable):
        le.resolve_engine("auto")


def test_resolve_explicit_unavailable_raises(monkeypatch):
    _force_available(monkeypatch, set())
    with pytest.raises(le.LocalLLMUnavailable):
        le.resolve_engine("ollama")
