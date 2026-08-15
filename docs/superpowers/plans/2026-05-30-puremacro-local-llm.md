# puremacro local-LLM backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let puremacro's two paid LLM features (narrative event extraction + index scoring) run free on a user's own machine (Windows/macOS) via local inference engines, with the same call signatures as the paid backends.

**Architecture:** One shared, lazily-imported **engine layer** (`narrative/_local_engines.py`) with three engines — `MLXEngine` (Apple GPU), `LlamaCppEngine` (cross-platform GGUF), `HTTPEngine` (Ollama / OpenAI-compatible, pure `urllib`) — selected by `engine="auto"`. Two thin wrappers expose it at the existing call sites: `LocalBackend`/`OllamaBackend` (for `score_llm`) and `LocalProvider`/`OllamaProvider` (for `llm_prob_kernel`). `get_default_*` factories fall back to a Mock so notebooks/CI/playground stay green. No new core deps (HTTP path is `urllib`); engines live in a new `[local-llm]` extra.

**Tech Stack:** Python 3.11+, stdlib `urllib`/`http.server`, `pytest`; optional `mlx-lm` (darwin) + `llama-cpp-python`; reuses `puremacro._http`, `puremacro.narrative.scoring.llm`, `puremacro.narrative.indices._llm_kernel`.

**Spec:** `docs/superpowers/specs/2026-05-30-puremacro-local-llm-design.md`

**Working directory for all commands:** the puremacro package root (the dir containing `pyproject.toml`, `puremacro/`, `tests/`, `notebooks/`, `tools/`). All paths below are relative to it.

**Conventions to follow (verified in this repo):**
- Backend interface for `score_llm`: a subclass of the `@dataclass _BackendBase` (`model`, `max_tokens=1024`, `temperature=0.0`) implementing `call(self, prompt: str) -> str`. Subclasses define a custom `__init__` and forward `**kw` to `super().__init__(model=model, **kw)` (see `AnthropicBackend`).
- Provider interface for `llm_prob_kernel`: a subclass of `LLMProvider` (ABC) with attributes `name`, `model`, and `score_paragraph(self, text, category) -> float` in `[0,1]`. The SQLite cache partitions on `(provider.name, provider.model, ...)`.
- All heavy/optional imports (`mlx_lm`, `llama_cpp`) go **inside** functions/methods, never at module top, to keep `import puremacro` Pyodide-clean.
- Use `python3` for all commands. Tests are flat files under `tests/` (narrative neighbors: `test_narrative_llm_scoring.py`, `test_narrative_indices.py`).
- `tests/known_failures.json` is empty → the suite is expected fully green; do not leave new failures.
- Commit message trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Editable-install note:** if a newly-created module is `ModuleNotFoundError` despite existing on disk, the conda env has a stale strict editable finder; fix once with `pip install -e . --config-settings editable_mode=compat` from the package root.

---

## Task 1: `post_json` HTTP helper

**Files:**
- Modify: `puremacro/_http.py` (add `post_json`; extend `__all__`)
- Test: `tests/test_local_http_post.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_http_post.py
"""post_json: urllib JSON POST used by the local-LLM HTTP engine."""
import http.server
import json
import threading

import pytest

from puremacro._http import post_json


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        sent = json.loads(self.rfile.read(n) or b"{}")
        out = json.dumps({"echo": sent, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def echo_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_post_json_roundtrip(echo_server):
    out = post_json(f"{echo_server}/x", {"a": 1, "b": "hi"}, timeout=5)
    assert out["path"] == "/x"
    assert out["echo"] == {"a": 1, "b": "hi"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_http_post.py -q`
Expected: FAIL — `ImportError: cannot import name 'post_json'`.

- [ ] **Step 3: Implement `post_json`**

In `puremacro/_http.py`, add after `safe_get_json` (before the cached variants section):

```python
def post_json(url: str, payload: dict, *, timeout: float = DEFAULT_TIMEOUT,
              headers: dict | None = None) -> dict:
    """POST ``payload`` as JSON and return the decoded JSON response.

    urllib-only (Pyodide-safe). HTTP errors propagate (not retried); a
    transport/SSL error retries once with verification off, matching
    ``_request``. Used by the local-LLM HTTP engine (Ollama / OpenAI-compatible).
    """
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, ssl.SSLError):
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
```

Then add `"post_json"` to the `__all__` list at the bottom of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_http_post.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/_http.py tests/test_local_http_post.py
git commit -m "feat(_http): add post_json urllib helper for local-LLM engines

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Engine layer — exceptions + model aliases

**Files:**
- Create: `puremacro/narrative/_local_engines.py`
- Test: `tests/test_local_engines.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_engines.py
"""Local inference engine layer: aliases, resolution, engines, errors."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'puremacro.narrative._local_engines'`.

- [ ] **Step 3: Create the module with exceptions + aliases**

```python
# puremacro/narrative/_local_engines.py
"""Local LLM inference engines for puremacro's narrative LLM features.

Lets the two paid call sites (``narrative.scoring.llm.score_llm`` event
extraction and ``narrative.indices._llm_kernel.llm_prob_kernel`` index scoring)
run for $0 on a user's own machine. Three engines, picked by ``engine="auto"``:

  * ``MLXEngine``      — Apple-Silicon GPU via ``mlx-lm`` ([local-llm] extra)
  * ``LlamaCppEngine`` — cross-platform GGUF via ``llama-cpp-python`` ([local-llm])
  * ``HTTPEngine``     — Ollama / LM Studio / any OpenAI-compatible local server
                         over ``urllib`` (NO extra; the zero-dependency path)

All heavy imports are lazy (inside methods) so ``import puremacro`` stays
Pyodide-clean. The HTTP engine cannot reach ``localhost`` inside a browser; that
is expected — local inference is a desktop feature.
"""
from __future__ import annotations

import platform
import re


class BackendUnavailable(RuntimeError):
    """A scoring backend/provider could not reach its model (connection/setup
    failure), as opposed to the model returning unparseable output. Callers
    (e.g. ``score_llm``) let this propagate instead of dropping it as malformed.
    """


class LocalLLMUnavailable(BackendUnavailable):
    """No local inference engine is usable (none installed / no server)."""


# Friendly canonical name -> per-engine model id.
#   mlx:      a HuggingFace repo id of an MLX-converted model
#   llamacpp: a (repo_id, filename_glob) tuple for Llama.from_pretrained
#   ollama:   an `ollama pull` tag
#   openai:   passed through as the OpenAI-compatible "model" field
# An unrecognized model name passes through verbatim to the engine.
MODEL_ALIASES: dict[str, dict[str, object]] = {
    "qwen2.5-3b-instruct": {
        "mlx": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "llamacpp": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "*Q4_K_M.gguf"),
        "ollama": "qwen2.5:3b",
        "openai": "qwen2.5-3b-instruct",
    },
    "gemma2-2b": {  # Google
        "mlx": "mlx-community/gemma-2-2b-it-4bit",
        "llamacpp": ("bartowski/gemma-2-2b-it-GGUF", "*Q4_K_M.gguf"),
        "ollama": "gemma2:2b",
        "openai": "gemma2-2b",
    },
    "llama3.2-3b": {  # Meta
        "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "llamacpp": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "*Q4_K_M.gguf"),
        "ollama": "llama3.2:3b",
        "openai": "llama3.2-3b",
    },
    "phi3.5": {  # Microsoft
        "mlx": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "llamacpp": ("bartowski/Phi-3.5-mini-instruct-GGUF", "*Q4_K_M.gguf"),
        "ollama": "phi3.5",
        "openai": "phi3.5",
    },
}


def resolve_model_id(model: str, engine_name: str):
    """Map a friendly canonical name to its per-engine id; pass unknown names
    through unchanged (escape hatch for any id the user already has)."""
    alias = MODEL_ALIASES.get(model)
    if alias is None:
        return model
    return alias.get(engine_name, model)
```

> **Note (verify at execution):** the HuggingFace repo ids above are real as of
> 2026-05; if `Llama.from_pretrained`/`mlx_lm.load` reports a 404, confirm the
> current repo id on HuggingFace and update the alias. This is the only place
> model ids live.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/_local_engines.py tests/test_local_engines.py
git commit -m "feat(narrative): local-engine module scaffold — exceptions + model aliases

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `HTTPEngine` (Ollama + OpenAI-compatible)

**Files:**
- Modify: `puremacro/narrative/_local_engines.py` (add `HTTPEngine`)
- Test: `tests/test_local_engines.py` (add a stub-server fixture + tests)

- [ ] **Step 1: Write the failing test** (append to `tests/test_local_engines.py`)

```python
import http.server
import json
import threading


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'HTTPEngine'`.

- [ ] **Step 3: Implement `HTTPEngine`** (append to `_local_engines.py`)

```python
class HTTPEngine:
    """Talks to a local server over urllib. ``api='ollama'`` uses Ollama's
    native /api/chat (supports format:"json"); ``api='openai'`` uses the
    OpenAI-compatible /v1/chat/completions (LM Studio / vLLM / llama.cpp-server,
    and Ollama also serves it)."""

    def __init__(self, *, base_url: str = "http://localhost:11434",
                 api: str = "ollama", timeout: float = 120.0):
        if api not in ("ollama", "openai"):
            raise ValueError(f"api must be 'ollama' or 'openai'; got {api!r}")
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.timeout = timeout
        self.name = api  # so the index-kernel SQLite cache partitions per server

    def available(self) -> bool:
        from .._http import safe_get_json
        path = "/api/tags" if self.api == "ollama" else "/v1/models"
        try:
            safe_get_json(self.base_url + path, timeout=min(self.timeout, 3.0))
            return True
        except Exception:
            return False

    def complete(self, model, prompt: str, *, max_tokens: int,
                 temperature: float, json_mode: bool) -> str:
        import urllib.error

        from .._http import post_json
        try:
            if self.api == "ollama":
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature,
                                "num_predict": max_tokens},
                }
                if json_mode:
                    payload["format"] = "json"
                body = post_json(self.base_url + "/api/chat", payload,
                                 timeout=self.timeout)
                return body.get("message", {}).get("content", "")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            body = post_json(self.base_url + "/v1/chat/completions", payload,
                             timeout=self.timeout)
            choices = body.get("choices") or []
            return choices[0].get("message", {}).get("content", "") if choices else ""
        except urllib.error.URLError as e:
            raise LocalLLMUnavailable(
                f"Cannot reach a local LLM server at {self.base_url}. "
                f"Start Ollama (https://ollama.com) and run "
                f"`ollama pull qwen2.5:3b`, or point base_url at LM Studio/"
                f"vLLM. Underlying error: {e}"
            ) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/_local_engines.py tests/test_local_engines.py
git commit -m "feat(narrative): HTTPEngine (Ollama + OpenAI-compatible) over urllib

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `MLXEngine` + `LlamaCppEngine`

These wrap heavy optional packages. They are unit-tested by injecting a **fake
module** into `sys.modules` (no real `mlx_lm`/`llama_cpp` needed); the live
real-engine tests come in Task 10.

**Files:**
- Modify: `puremacro/narrative/_local_engines.py` (add both engines)
- Test: `tests/test_local_engines.py` (add fake-module tests)

- [ ] **Step 1: Write the failing test** (append)

```python
import sys
import types


def test_mlx_engine_complete_with_fake_module(monkeypatch):
    captured = {}

    class _Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            captured["messages"] = messages
            return "PROMPT:" + messages[-1]["content"]

    fake = types.ModuleType("mlx_lm")
    fake.load = lambda model_id: (captured.setdefault("loaded", model_id), _Tok())[::-1][0] \
        if False else ("MODEL", _Tok())
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_engines.py -k "fake_module" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'MLXEngine'`.

- [ ] **Step 3: Implement both engines** (append to `_local_engines.py`)

```python
class MLXEngine:
    """Apple-Silicon GPU inference via mlx-lm. Greedy (temperature ignored —
    deterministic, which is what extraction wants). Models cached per instance."""

    name = "mlx"

    def __init__(self):
        self._cache: dict[str, object] = {}

    def available(self) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self, model_id: str):
        if model_id not in self._cache:
            from mlx_lm import load
            self._cache[model_id] = load(model_id)
        return self._cache[model_id]

    def complete(self, model, prompt: str, *, max_tokens: int,
                 temperature: float, json_mode: bool) -> str:
        from mlx_lm import generate
        mdl, tok = self._load(model)
        text_prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
        )
        return generate(mdl, tok, prompt=text_prompt,
                        max_tokens=max_tokens, verbose=False)


class LlamaCppEngine:
    """Cross-platform GGUF inference via llama-cpp-python. ``model`` is either a
    local .gguf path (str) or a (repo_id, filename_glob) tuple auto-downloaded
    from HuggingFace. Models cached per instance."""

    name = "llamacpp"

    def __init__(self, *, n_ctx: int = 4096):
        self._cache: dict = {}
        self.n_ctx = n_ctx

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self, model):
        import os
        key = model if isinstance(model, str) else tuple(model)
        if key not in self._cache:
            from llama_cpp import Llama
            if isinstance(model, str) and os.path.exists(model):
                self._cache[key] = Llama(model_path=model, n_ctx=self.n_ctx,
                                         verbose=False)
            else:
                repo_id, filename = model
                self._cache[key] = Llama.from_pretrained(
                    repo_id=repo_id, filename=filename,
                    n_ctx=self.n_ctx, verbose=False,
                )
        return self._cache[key]

    def complete(self, model, prompt: str, *, max_tokens: int,
                 temperature: float, json_mode: bool) -> str:
        llm = self._load(model)
        kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = llm.create_chat_completion(**kwargs)
        return resp["choices"][0]["message"]["content"]
```

> **Note (verify at execution):** `mlx_lm.generate` accepts `max_tokens` and
> `verbose`; newer versions take a `sampler=` for temperature (greedy default is
> temp 0). If the installed `generate` rejects `max_tokens`, check
> `python3 -c "import mlx_lm, inspect; print(inspect.signature(mlx_lm.generate))"`
> and adjust this single method. Same for `Llama.create_chat_completion`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/_local_engines.py tests/test_local_engines.py
git commit -m "feat(narrative): MLXEngine + LlamaCppEngine (lazy optional engines)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `resolve_engine` (auto + explicit selection)

**Files:**
- Modify: `puremacro/narrative/_local_engines.py` (add `resolve_engine` + `chat`)
- Test: `tests/test_local_engines.py` (add selection tests)

- [ ] **Step 1: Write the failing test** (append)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_engines.py -k resolve -q`
Expected: FAIL — `AttributeError: ... has no attribute 'resolve_engine'`.

- [ ] **Step 3: Implement `resolve_engine` + `chat`** (append)

```python
def _make_engine(name: str, *, base_url: str, timeout: float):
    if name == "mlx":
        return MLXEngine()
    if name == "llamacpp":
        return LlamaCppEngine()
    if name == "ollama":
        return HTTPEngine(base_url=base_url, api="ollama", timeout=timeout)
    if name == "openai":
        return HTTPEngine(base_url=base_url, api="openai", timeout=timeout)
    raise ValueError(
        f"unknown engine {name!r}; expected one of "
        "auto/mlx/llamacpp/ollama/openai"
    )


_INSTALL_HINT = (
    "No local LLM engine available. Install one with "
    "`pip install puremacro[local-llm]` (MLX on Apple Silicon, or "
    "llama-cpp-python anywhere), or start Ollama (https://ollama.com) and run "
    "`ollama pull qwen2.5:3b`."
)


def resolve_engine(engine: str = "auto", *,
                   base_url: str = "http://localhost:11434",
                   timeout: float = 120.0):
    """Return a usable engine instance, or raise LocalLLMUnavailable.

    ``"auto"`` tries, in order: on Apple Silicon mlx -> llamacpp -> ollama;
    elsewhere llamacpp -> ollama. Explicit names are checked for availability.
    """
    if engine != "auto":
        eng = _make_engine(engine, base_url=base_url, timeout=timeout)
        if not eng.available():
            raise LocalLLMUnavailable(
                f"engine {engine!r} is not available. {_INSTALL_HINT}"
            )
        return eng

    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    order = ["mlx", "llamacpp", "ollama"] if is_apple else ["llamacpp", "ollama"]
    for name in order:
        eng = _make_engine(name, base_url=base_url, timeout=timeout)
        if eng.available():
            return eng
    raise LocalLLMUnavailable(_INSTALL_HINT)


def chat(model: str, prompt: str, *, engine: str = "auto",
         base_url: str = "http://localhost:11434", max_tokens: int = 1024,
         temperature: float = 0.0, json_mode: bool = False,
         timeout: float = 120.0) -> str:
    """One-shot convenience: resolve an engine and return its completion.
    (Wrappers hold a persistent engine instead, to keep models loaded.)"""
    eng = resolve_engine(engine, base_url=base_url, timeout=timeout)
    return eng.complete(resolve_model_id(model, eng.name), prompt,
                        max_tokens=max_tokens, temperature=temperature,
                        json_mode=json_mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_engines.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/_local_engines.py tests/test_local_engines.py
git commit -m "feat(narrative): resolve_engine auto-selection + chat() convenience

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `LocalBackend` / `OllamaBackend` / `MockBackend` + `get_default_backend`

**Files:**
- Modify: `puremacro/narrative/scoring/llm.py` (add classes + factory)
- Modify: `puremacro/narrative/scoring/__init__.py` (exports)
- Test: `tests/test_local_backends.py` (create)

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_backends.py -q`
Expected: FAIL — `ImportError: cannot import name 'LocalBackend' from 'puremacro.narrative.scoring'`.

- [ ] **Step 3: Implement** in `puremacro/narrative/scoring/llm.py`

Add these classes after `OpenAIBackend` (before the "Top-level scoring loop" section):

```python
class LocalBackend(_BackendBase):
    """Free local-LLM backend for ``score_llm`` (MLX / llama.cpp / HTTP).

    No API key. ``engine="auto"`` picks the best installed engine. ``base_url``
    only matters for the HTTP engine (Ollama / LM Studio / OpenAI-compatible).
    """

    def __init__(self, model: str = "qwen2.5-3b-instruct", *,
                 engine: str = "auto", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0, json_mode: bool = True, **kw):
        super().__init__(model=model, **kw)
        from .._local_engines import resolve_engine
        self.engine = engine
        self.base_url = base_url
        self.json_mode = json_mode
        self._engine = resolve_engine(engine, base_url=base_url, timeout=timeout)

    def call(self, prompt: str) -> str:
        from .._local_engines import resolve_model_id
        model_id = resolve_model_id(self.model, self._engine.name)
        return self._engine.complete(
            model_id, prompt, max_tokens=self.max_tokens,
            temperature=self.temperature, json_mode=self.json_mode,
        )


class OllamaBackend(LocalBackend):
    """Preset: ``LocalBackend(engine="ollama")`` for a running Ollama/LM Studio."""

    def __init__(self, model: str = "qwen2.5:3b", *,
                 base_url: str = "http://localhost:11434", **kw):
        super().__init__(model=model, engine="ollama", base_url=base_url, **kw)


class MockBackend(_BackendBase):
    """Offline fallback: returns an empty JSON array (zero events), so
    pipelines/notebooks run deterministically when no engine is installed."""

    def __init__(self, model: str = "mock", **kw):
        super().__init__(model=model, **kw)

    def call(self, prompt: str) -> str:
        return "[]"


def get_default_backend(model: str = "qwen2.5-3b-instruct", **kw) -> _BackendBase:
    """Best available local backend, or ``MockBackend`` if no engine is present.
    Prints which engine was selected. Lets demos/notebooks run free by default."""
    from .._local_engines import LocalLLMUnavailable
    try:
        be = LocalBackend(model=model, **kw)
        print(f"[puremacro] local LLM backend: engine={be._engine.name}")
        return be
    except LocalLLMUnavailable as e:
        print(f"[puremacro] {e}\n[puremacro] using MockBackend (zero events).")
        return MockBackend()
```

Update the module `__all__` at the bottom of `llm.py`:

```python
__all__ = ["AnthropicBackend", "OpenAIBackend", "score_llm",
           "LocalBackend", "OllamaBackend", "MockBackend", "get_default_backend",
           "_PROMPTS", "_build_prompt"]
```

In `puremacro/narrative/scoring/__init__.py`, extend the `from .llm import ...`
line and `__all__`:

```python
from .llm import (
    score_llm, AnthropicBackend, OpenAIBackend,
    LocalBackend, OllamaBackend, MockBackend, get_default_backend,
)
```
and add `"LocalBackend", "OllamaBackend", "MockBackend", "get_default_backend"`
to that file's `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_backends.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/scoring/llm.py puremacro/narrative/scoring/__init__.py tests/test_local_backends.py
git commit -m "feat(scoring): LocalBackend/OllamaBackend/MockBackend + get_default_backend

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `score_llm` propagates backend-unavailable (instead of dropping it)

Today `score_llm` wraps `backend.call` in `except Exception: drop as malformed`,
so a down engine silently yields zero events. Make `BackendUnavailable`
propagate while still dropping genuine parse/validation errors.

**Files:**
- Modify: `puremacro/narrative/scoring/llm.py` (the `score_llm` loop)
- Test: `tests/test_local_backends.py` (add two tests)

- [ ] **Step 1: Write the failing test** (append to `tests/test_local_backends.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_backends.py -k "propagates or drops" -q`
Expected: `test_score_llm_propagates_backend_unavailable` FAILS (currently the
exception is swallowed and the call returns `[]`); the drops test passes.

- [ ] **Step 3: Implement** — in `score_llm` (in `llm.py`), change the per-record
call from:

```python
        try:
            response = backend.call(prompt)
        except Exception:
            n_dropped_malformed += 1
            continue
```

to:

```python
        try:
            response = backend.call(prompt)
        except BackendUnavailable:
            raise
        except Exception:
            n_dropped_malformed += 1
            continue
```

Add the import near the top of `llm.py` (with the other `from ..` imports):

```python
from .._local_engines import BackendUnavailable
```

> This top-level import is safe: `_local_engines` imports only stdlib +
> `_http` at module scope (engines are lazy), so it does not break Pyodide.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_backends.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Regression-check the existing scoring tests**

Run: `python3 -m pytest tests/test_narrative_llm_scoring.py -q`
Expected: PASS (no regressions — the new behavior only fires on `BackendUnavailable`).

- [ ] **Step 6: Commit**

```bash
git add puremacro/narrative/scoring/llm.py tests/test_local_backends.py
git commit -m "fix(scoring): score_llm propagates BackendUnavailable, still drops parse errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `LocalProvider` / `OllamaProvider` + `get_default_provider`

**Files:**
- Modify: `puremacro/narrative/indices/_llm_kernel.py` (add classes + factory)
- Modify: `puremacro/narrative/indices/__init__.py` (exports)
- Test: `tests/test_local_backends.py` (add provider tests)

- [ ] **Step 1: Write the failing test** (append)

```python
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


def test_get_default_provider_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(
        le, "resolve_engine",
        lambda *a, **k: (_ for _ in ()).throw(le.LocalLLMUnavailable("none")),
    )
    assert isinstance(get_default_provider(), MockProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_backends.py -k provider -q`
Expected: FAIL — `ImportError: cannot import name 'LocalProvider'`.

- [ ] **Step 3: Implement** in `puremacro/narrative/indices/_llm_kernel.py`

Add after `AnthropicProvider`:

```python
class LocalProvider(LLMProvider):
    """Free local-LLM provider for ``llm_prob_kernel`` (MLX / llama.cpp / HTTP).

    No API key. ``name`` is set to the resolved engine so the SQLite score
    cache partitions per engine. Robustly extracts the first float in the reply.
    """

    def __init__(self, model: str = "qwen2.5-3b-instruct", *,
                 engine: str = "auto", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0):
        from .._local_engines import resolve_engine
        self._engine = resolve_engine(engine, base_url=base_url, timeout=timeout)
        self.name = self._engine.name
        self.model = model

    def score_paragraph(self, text: str, category: str) -> float:
        import re

        from .._local_engines import resolve_model_id
        prompt = (
            f"Score how strongly this paragraph expresses '{category}' "
            f"on a scale from 0.0 (not at all) to 1.0 (strongly). "
            f"Reply with ONLY a single decimal number.\n\nParagraph: {text}"
        )
        model_id = resolve_model_id(self.model, self._engine.name)
        out = self._engine.complete(model_id, prompt, max_tokens=8,
                                    temperature=0.0, json_mode=False)
        m = re.search(r"\d*\.?\d+", out or "")
        if not m:
            return 0.0
        try:
            return max(0.0, min(1.0, float(m.group())))
        except ValueError:
            return 0.0


class OllamaProvider(LocalProvider):
    """Preset: ``LocalProvider(engine="ollama")``."""

    def __init__(self, model: str = "qwen2.5:3b", *,
                 base_url: str = "http://localhost:11434", **kw):
        super().__init__(model=model, engine="ollama", base_url=base_url, **kw)


def get_default_provider(model: str = "qwen2.5-3b-instruct", **kw) -> LLMProvider:
    """Best available local provider, or ``MockProvider`` if no engine present."""
    from .._local_engines import LocalLLMUnavailable
    try:
        p = LocalProvider(model=model, **kw)
        print(f"[puremacro] local LLM provider: engine={p.name}")
        return p
    except LocalLLMUnavailable as e:
        print(f"[puremacro] {e}\n[puremacro] using MockProvider.")
        return MockProvider()
```

Extend `__all__` in `_llm_kernel.py`:

```python
__all__ = [
    "LLMProvider", "MockProvider", "AnthropicProvider",
    "LocalProvider", "OllamaProvider", "get_default_provider",
    "llm_prob_kernel",
]
```

In `puremacro/narrative/indices/__init__.py`, extend the `_llm_kernel` import and
`__all__`:

```python
from ._llm_kernel import (
    llm_prob_kernel, LLMProvider, MockProvider, AnthropicProvider,
    LocalProvider, OllamaProvider, get_default_provider,
)
```
and add `"LocalProvider", "OllamaProvider", "get_default_provider"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_backends.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add puremacro/narrative/indices/_llm_kernel.py puremacro/narrative/indices/__init__.py tests/test_local_backends.py
git commit -m "feat(indices): LocalProvider/OllamaProvider + get_default_provider

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Packaging — `[local-llm]` extra, version bump, Pyodide gate, green-suite reconciliation

**Files:**
- Modify: `pyproject.toml` (extra + version)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/fixtures/public_api_snapshot.json` + any version-literal test (keep green)
- Create: `tests/test_pyodide/test_local_engines_importable.py`

- [ ] **Step 1: Add the `[local-llm]` extra + bump version in `pyproject.toml`**

Under `[project.optional-dependencies]`, add (place it next to `llm`/`embeddings`):

```toml
local-llm = [
  "llama-cpp-python>=0.3",                      # cross-platform GGUF (CPU/Metal/CUDA)
  "mlx-lm>=0.20; sys_platform == 'darwin'",     # Apple-Silicon GPU
]
```

Change the version line at the top of `pyproject.toml`:

```toml
version = "0.92.0"
```

- [ ] **Step 2: Bump `__version__`**

In `puremacro/__init__.py`, set `__version__ = "0.92.0"` (find the existing
`__version__ = "0.91.0"` line and change it).

- [ ] **Step 3: Write the Pyodide importable test** (mirrors `test_narrative_importable.py`)

```python
# tests/test_pyodide/test_local_engines_importable.py
"""Regression: the local-engine layer and the two LLM call sites import in a
Pyodide-like environment where the optional inference engines (mlx_lm,
llama_cpp) are ABSENT. The engines must load LAZILY (inside methods) so the
HTTP path and the rest of the package import browser-clean.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import sys
    _BLOCKED = {"mlx_lm", "llama_cpp", "mlx"}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
    import importlib
    importlib.import_module(sys.argv[1])
    leaked = sorted(m for m in _BLOCKED if m in sys.modules)
    assert not leaked, "leaked engine deps on import path: " + repr(leaked)
    print("OK")
    """
)

_TARGETS = [
    "puremacro.narrative._local_engines",
    "puremacro.narrative.scoring",
    "puremacro.narrative.indices",
]


@pytest.mark.parametrize("target", _TARGETS)
def test_local_engine_imports_without_inference_deps(target):
    r = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"{target} failed to import with engines absent:\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
```

- [ ] **Step 4: Run the Pyodide test (verify lazy imports hold)**

Run: `python3 -m pytest tests/test_pyodide/test_local_engines_importable.py -q`
Expected: PASS (3 passed). If it fails, an engine import leaked to module scope —
move it inside the method that uses it.

- [ ] **Step 5: Reconcile version-literal tests + the public-API snapshot (keep suite green)**

Run the two suites that pin the version/API and read what they expect:

```bash
python3 -m pytest tests/test_import.py tests/test_release_check.py tests/test_public_api.py -q
```

- For any failure asserting `"0.91.0"`, update the literal to `"0.92.0"` in that
  test file (grep: `grep -rn "0\.91\.0" tests/`).
- For `tests/test_public_api.py`: open it to find how it loads
  `tests/fixtures/public_api_snapshot.json` and whether it offers a regenerate
  switch (e.g. an env var like `UPDATE_SNAPSHOT=1`, or it simply diffs the JSON).
  Add the six new public names to the snapshot so the diff is clean:
  `LocalBackend`, `OllamaBackend`, `MockBackend`, `get_default_backend` (scoring)
  and `LocalProvider`, `OllamaProvider`, `get_default_provider` (indices) — under
  whatever module keys the snapshot uses for `puremacro.narrative.scoring` and
  `puremacro.narrative.indices`. If the test exposes a regenerate command, prefer
  that; then re-read the diff to confirm only the intended names were added.

Re-run until green:

```bash
python3 -m pytest tests/test_import.py tests/test_release_check.py tests/test_public_api.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml puremacro/__init__.py tests/test_pyodide/test_local_engines_importable.py tests/fixtures/public_api_snapshot.json tests/test_import.py tests/test_release_check.py
git commit -m "feat(packaging): [local-llm] extra, v0.92.0, Pyodide gate + API/version reconcile

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Live engine tests + directional validation (skip if absent)

These exercise the real engines when present and **skip** otherwise (standing
rule: network/engine tests skip on empty, never assert against a missing engine).

**Files:**
- Test: `tests/test_local_llm_live.py` (create)

- [ ] **Step 1: Write the live tests**

```python
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
```

- [ ] **Step 2: Register the marker** — in `pyproject.toml` (or `pytest.ini`/
`setup.cfg`, wherever `[tool.pytest.ini_options]` markers live), add `local_llm`
to the `markers` list so `-m local_llm` is recognized and no "unknown marker"
warning appears. (Check first: `grep -n "markers" pyproject.toml`.) Example
addition under `[tool.pytest.ini_options] markers = [...]`:

```toml
  "local_llm: opt-in tests that need a real local LLM engine (skip if absent)",
```

- [ ] **Step 3: Run — confirm they SKIP cleanly here (no engine installed)**

Run: `python3 -m pytest tests/test_local_llm_live.py -q -rs`
Expected: all tests SKIPPED (reason: "engine ... not available"), 0 failures.

- [ ] **Step 4 (optional, do if validating locally): install MLX and run for real**

```bash
pip install mlx-lm
python3 -m pytest tests/test_local_llm_live.py -q -rs -k mlx
```
Expected: the `mlx` params run (first call downloads the model, ~1.5–2 GB);
the directional + structure assertions pass. Other engines still skip.

- [ ] **Step 5: Commit**

```bash
git add tests/test_local_llm_live.py pyproject.toml
git commit -m "test(local-llm): live engine tests + directional validation (skip if absent)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Runnable example script + smoke test

**Files:**
- Create: `examples/narrative_local_llm.py`
- Test: `tests/test_examples_local_llm.py` (create)

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_examples_local_llm.py
"""Smoke: the local-LLM example runs end-to-end (MockBackend fallback when no
engine is installed) without raising."""
import importlib


def test_example_main_runs(capsys):
    mod = importlib.import_module("examples.narrative_local_llm")
    mod.main()  # no engine in CI -> get_default_* falls back to Mock
    out = capsys.readouterr().out
    assert "engine=" in out or "MockBackend" in out or "MockProvider" in out
```

> If `examples` is not importable as a package, the test runner already adds the
> repo root to `sys.path` (it imports other `examples.*` in `tests/test_examples`).
> Confirm with `ls tests/test_examples` and mirror its import style if different.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_examples_local_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.narrative_local_llm'`.

- [ ] **Step 3: Write the example**

```python
# examples/narrative_local_llm.py
"""Run puremacro's narrative LLM features for FREE on a local model.

No API key, no paid API. Install a local engine once:

    pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp
    # OR install Ollama (https://ollama.com) and: ollama pull qwen2.5:3b

Then:  python -m examples.narrative_local_llm

With no engine installed it falls back to the offline Mock (so this script
always runs); install an engine to see real local inference.
"""
from __future__ import annotations

from puremacro.narrative.indices import get_default_provider, llm_prob_kernel
from puremacro.narrative.scoring import get_default_backend, score_llm

_CORPUS = [
    ("2020-03-15",
     "The government announced a $500 billion infrastructure investment package "
     "to be implemented next quarter.",
     "http://example.test/a"),
    ("2020-04-01",
     "Officials warned the outlook is highly uncertain and policy could shift "
     "abruptly amid unpredictable risks.",
     "http://example.test/b"),
]


def main() -> None:
    print("=== puremacro local LLM demo (free, $0) ===")

    backend = get_default_backend("qwen2.5-3b-instruct")
    events = score_llm(_CORPUS, backend=backend, kind="fiscal")
    print(f"[events] extracted {len(events)} fiscal event(s)")
    for ev in events:
        print(f"  - {ev.date.date()} sign={ev.sign} mag={ev.magnitude} "
              f"{ev.magnitude_unit}: {ev.source_text[:80]}")

    provider = get_default_provider("qwen2.5-3b-instruct")
    series = list(llm_prob_kernel(_CORPUS, provider=provider,
                                  category="economic uncertainty"))
    print("[index] P(uncertainty) per document:")
    for date, p in series:
        print(f"  - {date.date()}: {p:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_examples_local_llm.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Sanity-run the script**

Run: `python3 -m examples.narrative_local_llm`
Expected: prints the demo header, "using MockBackend/MockProvider" notes (no
engine here), 0 events, and two `P(uncertainty)` lines — no traceback.

- [ ] **Step 6: Commit**

```bash
git add examples/narrative_local_llm.py tests/test_examples_local_llm.py
git commit -m "docs(examples): runnable free local-LLM narrative demo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Desktop showcase notebook (excluded from the browser playground)

The notebook source is a jupytext `.py` in `notebooks/`, built by
`tools/build_notebooks.py` (globs `notebooks/*.py` minus `_*.py`). A
**non-numeric** name keeps it out of the playground (which selects
`[0-1][0-9]_*.ipynb`) while still being built + covered by the execute-all gate.
It uses `get_default_*` so it runs green with the Mock fallback when no engine
is present.

**Files:**
- Create: `notebooks/local_llm_uncertainty.py` (jupytext percent format)
- Generated (committed artifact): `notebooks/local_llm_uncertainty.ipynb`

- [ ] **Step 1: Confirm the builder discovers a non-numeric source**

Run: `python3 tools/build_notebooks.py --list`
Expected: the current list (01..11). After Step 2 the new stem will appear here.
Also confirm the playground selector is numeric-only:
`grep -rn "ipynb" playground/build_playground.sh | head` (look for a
`[0-1][0-9]_*` / numeric glob — our non-numeric name must NOT match it).

- [ ] **Step 2: Write the notebook source**

```python
# notebooks/local_llm_uncertainty.py
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Free local-LLM narrative analysis (no API key, $0)
#
# puremacro's LLM features can run on a model on **your own machine** — Apple
# MLX, llama.cpp, or a local Ollama/LM Studio server — instead of a paid API.
#
# **Desktop only:** local inference needs a real engine, so this notebook does
# not run a model inside the browser playground. With no engine installed it
# falls back to an offline Mock so the notebook still executes; install one with
# `pip install "puremacro[local-llm]"` (or run Ollama) to see real inference.

# %%
import _nbstyle  # noqa: F401  (grayscale figure style; see notebooks/_nbstyle.py)

from puremacro.narrative.scoring import get_default_backend, score_llm
from puremacro.narrative.indices import get_default_provider, llm_prob_kernel

CORPUS = [
    ("2020-03-15",
     "The government announced a $500 billion infrastructure investment package.",
     "http://example.test/a"),
    ("2020-04-01",
     "Officials warned the outlook is highly uncertain and could shift abruptly.",
     "http://example.test/b"),
]

# %% [markdown]
# ## 1. Pick the best available local engine
# `get_default_backend` / `get_default_provider` auto-select MLX → llama.cpp →
# Ollama, falling back to a Mock if none is installed (which is what happens in
# CI / the browser).

# %%
backend = get_default_backend("qwen2.5-3b-instruct")
provider = get_default_provider("qwen2.5-3b-instruct")

# %% [markdown]
# ## 2. Extract narrative fiscal events (free)

# %%
events = score_llm(CORPUS, backend=backend, kind="fiscal")
print(f"extracted {len(events)} event(s)")
for ev in events:
    print(ev.date.date(), ev.sign, ev.magnitude, ev.magnitude_unit)

# %% [markdown]
# ## 3. Build a per-document uncertainty index (free)

# %%
series = list(llm_prob_kernel(CORPUS, provider=provider,
                              category="economic uncertainty"))
for date, p in series:
    print(date.date(), round(p, 3))

# %% [markdown]
# With a real engine installed, the April "uncertain" document scores higher
# than the March "investment" document. Swap models via the `model=` argument
# (e.g. `"gemma2-2b"` for Google's Gemma, `"llama3.2-3b"` for Meta's Llama).
```

- [ ] **Step 3: Build (convert + execute) the notebook**

Run: `python3 tools/build_notebooks.py local_llm_uncertainty`
Expected: writes `notebooks/local_llm_uncertainty.ipynb` with outputs; the cells
run with the Mock fallback (no engine here) and print 0 events + two index lines.
**Do not hand-edit the `.ipynb`** — it's a build artifact (edit the `.py` and rebuild).

- [ ] **Step 4: Confirm the execute-all gate still passes and the playground excludes it**

Run: `python3 -m pytest tests/test_notebooks -q` (the notebooks execute-all gate;
confirm the exact path with `ls tests | grep notebook`).
Expected: PASS, now including `local_llm_uncertainty`.

Run: `grep -rn "local_llm" playground/ 2>/dev/null` after a dry inspection of the
selector — expected: NO match (non-numeric name is not selected by the playground).

- [ ] **Step 5: Commit**

```bash
git add notebooks/local_llm_uncertainty.py notebooks/local_llm_uncertainty.ipynb
git commit -m "docs(notebooks): desktop free-local-LLM showcase (excluded from playground)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Docs — README, CHANGELOG, ARCHITECTURE

**Files:**
- Modify: `README.md` (new section)
- Modify: `CHANGELOG.md` (0.92.0 entry)
- Modify: `ARCHITECTURE.md` (engine layer note)

- [ ] **Step 1: README — add a "Run the LLM features for free (local models)" section**

Insert after the existing narrative/LLM usage section (find it with
`grep -n "anthropic\|score_llm\|LLM\|narrative" README.md | head`). Add:

````markdown
### Run the LLM features for free (local models)

The narrative LLM features (`score_llm`, `llm_prob_kernel`) run on a **local
model** — no API key, no paid API, $0. Everything else in puremacro is already
free; this closes the last paid gap.

Install an engine once (any one):

```bash
pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp (any OS)
# or install Ollama (https://ollama.com) — no Python deps — then:  ollama pull qwen2.5:3b
```

Then swap in a local backend (same signatures as the paid backends):

```python
from puremacro.narrative.scoring import score_llm, LocalBackend
events = score_llm(records, backend=LocalBackend("qwen2.5-3b-instruct", engine="auto"))

from puremacro.narrative.indices import llm_prob_kernel, LocalProvider
idx = llm_prob_kernel(records, provider=LocalProvider("qwen2.5-3b-instruct"),
                      category="economic uncertainty")
```

`engine="auto"` picks the best installed engine (Apple GPU via MLX → llama.cpp →
a running Ollama/LM Studio). Models: `qwen2.5-3b-instruct` (default),
`gemma2-2b` (Google), `llama3.2-3b` (Meta), `phi3.5` (Microsoft), or any raw
engine model id. See `examples/narrative_local_llm.py` and the
`local_llm_uncertainty` notebook. (Local inference is desktop-only — it does not
run inside the browser playground.)
````

- [ ] **Step 2: CHANGELOG — add the 0.92.0 entry at the top**

```markdown
## 0.92.0 — 2026-05-30

### Added
- **Free local-LLM backends** (`puremacro.narrative._local_engines`): run the
  narrative LLM features at $0 on your own machine. `LocalBackend`/`OllamaBackend`
  for `score_llm`, `LocalProvider`/`OllamaProvider` for `llm_prob_kernel`, plus
  `get_default_backend`/`get_default_provider` (fall back to Mock when no engine
  is installed). `engine="auto"` selects MLX (Apple Silicon) → llama.cpp → a
  local Ollama/OpenAI-compatible HTTP server. New `[local-llm]` extra
  (`llama-cpp-python`, `mlx-lm` on darwin); the HTTP path needs no new deps.
- `puremacro._http.post_json` (urllib JSON POST).

### Changed
- `score_llm` now lets `BackendUnavailable` (a down engine/server) propagate
  instead of silently dropping every record as "malformed"; genuine parse errors
  are still dropped.

### Notes
- Engines are lazily imported; `import puremacro` stays Pyodide-clean. Local
  inference is desktop-only (not available in the browser playground).
```

- [ ] **Step 3: ARCHITECTURE — note the engine layer**

Find the narrative/LLM section (`grep -n "narrative\|scoring\|llm\|kernel" ARCHITECTURE.md | head`)
and add a short paragraph:

```markdown
#### Local LLM engines (`narrative/_local_engines.py`)

A lazily-imported engine layer lets the two LLM call sites run on a local model
at $0. `resolve_engine("auto")` selects `MLXEngine` (Apple GPU) → `LlamaCppEngine`
(GGUF) → `HTTPEngine` (Ollama / OpenAI-compatible, urllib). `LocalBackend`
(scoring) and `LocalProvider` (indices) are thin wrappers; `MODEL_ALIASES` maps a
friendly model name to per-engine ids. Heavy engine imports stay inside methods
so `import puremacro` remains Pyodide-clean; the `[local-llm]` extra carries the
optional packages.
```

- [ ] **Step 4: Verify docs build/parse (no broken fences) + version consistency**

Run: `grep -n "0.92.0" pyproject.toml puremacro/__init__.py CHANGELOG.md`
Expected: the version appears in all three.

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md ARCHITECTURE.md
git commit -m "docs: document free local-LLM backends (README/CHANGELOG/ARCHITECTURE) for 0.92.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (run after all tasks)

- [ ] **Full new-surface test run**

Run:
```bash
python3 -m pytest tests/test_local_http_post.py tests/test_local_engines.py \
  tests/test_local_backends.py tests/test_local_llm_live.py \
  tests/test_examples_local_llm.py tests/test_pyodide/test_local_engines_importable.py -q -rs
```
Expected: all pass; the live tests SKIP (no engine in CI).

- [ ] **Regression: narrative + pyodide + public-api + version suites**

Run:
```bash
python3 -m pytest tests/test_narrative_llm_scoring.py tests/test_narrative_indices.py \
  tests/test_pyodide tests/test_public_api.py tests/test_import.py tests/test_release_check.py -q
```
Expected: all pass (suite stays green; `known_failures.json` still empty).

- [ ] **Pyodide compat gate (engines must not leak)**

Run: `python3 -m pytest tests/test_pyodide_compat.py -q`
Expected: PASS.

---

## Self-review checklist (completed during authoring)

- **Spec coverage:** engine layer (T2–5), both call sites (T6/T8), `score_llm`
  fix (T7), `[local-llm]` extra + version + Pyodide gate (T9), tests incl.
  skip-if-absent + directional validation (T10), example (T11), desktop notebook
  excluded from playground (T12), docs (T13), `post_json` helper (T1). All spec
  §s mapped.
- **Type/name consistency:** engine `.complete(model, prompt, *, max_tokens,
  temperature, json_mode)` and `.name`/`.available()` are used identically across
  `HTTPEngine`/`MLXEngine`/`LlamaCppEngine`, `resolve_engine`, `chat`,
  `LocalBackend`, `LocalProvider`, and the fakes in tests. `BackendUnavailable` ⊃
  `LocalLLMUnavailable` defined in T2, imported in T7. `MODEL_ALIASES` keys match
  the model defaults (`qwen2.5-3b-instruct`).
- **Placeholders:** none — every code step is complete. The two "verify at
  execution" notes (external engine signatures, HF repo ids) are localized
  confirmations with the exact command to run, not deferred implementation.
- **Green suite:** T9 explicitly reconciles the version-literal tests and the
  public-API snapshot so `known_failures.json` stays empty.
