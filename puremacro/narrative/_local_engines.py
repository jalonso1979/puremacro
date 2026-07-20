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
MODEL_ALIASES: dict[str, dict[str, str | tuple[str, str]]] = {
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


def resolve_model_id(model: str, engine_name: str) -> str | tuple[str, str]:
    """Map a friendly canonical name to its per-engine id; pass unknown names
    through unchanged (escape hatch for any id the user already has)."""
    alias = MODEL_ALIASES.get(model)
    if alias is None:
        return model
    return alias.get(engine_name, model)


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
