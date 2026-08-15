# puremacro local-LLM backends — design spec

- **Date:** 2026-05-30
- **Status:** design approved (brainstorming), pending spec review → writing-plans
- **Target version:** 0.91.0 → **0.92.0**
- **Scope:** `puremacro/` package subtree only (the monorepo subdir)
- **Branch context:** authored on `feature/regime-uncertainty-companion-phase2a`

## 1. Motivation & goal

puremacro is a 4-dependency, MIT, browser-runnable macro stack whose explicit
mission is to let no-budget users do MATLAB/Stata/Bloomberg-class work at $0.
Today **one** capability still costs money: the two LLM-backed narrative paths,
which call paid Anthropic/OpenAI APIs. Everything else is already free
(lexicon/MNL scoring is pure numpy; embeddings run locally via
`sentence-transformers`; data connectors use free API keys).

**Goal.** A user on **Windows or macOS** runs puremacro's LLM features —
**both** narrative-event extraction *and* index scoring — entirely on their own
machine at **$0**, using the best free local inference engine available, with
the *same call signatures* as the paid backends.

**Success criteria.**
1. `pip install puremacro[local-llm]` + a one-line backend swap runs both LLM
   paths locally with no API key and no network egress.
2. `engine="auto"` selects the best installed engine without the user thinking
   about it (Apple GPU via MLX → cross-platform llama.cpp → a running
   Ollama/LM Studio HTTP server → a clear, actionable error).
3. Notebooks/examples that ask for "the default backend" run free out of the
   box, degrading to the offline `MockProvider` when no engine is present, so
   CI and the browser playground stay green.
4. `import puremacro` remains Pyodide-clean (engines are lazily imported
   optional extras; the HTTP engine is pure `urllib`).
5. A small validation shows a local 2–3B model reproduces the *direction* of a
   known uncertainty signal — i.e. local models are good enough to be useful,
   not merely that the plumbing runs.

## 2. Background: the two paid call sites (current, verified)

These interfaces are fixed points the new code must satisfy; both are already
backend-agnostic, which is why this is cheap.

**(a) Event extraction — `puremacro/narrative/scoring/llm.py`**

```python
score_llm(text_iter, *, backend, kind="fiscal", language="en",
          country="USA", dry_run=False) -> list[NarrativeEvent]
```
`backend` is any subclass of the `_BackendBase` dataclass
(`model: str`, `max_tokens: int = 1024`, `temperature: float = 0.0`) that
implements `call(self, prompt: str) -> str`. Existing subclasses
`AnthropicBackend`, `OpenAIBackend` are plain `urllib` (no SDK). Response
parsing (`_parse_response`) already strips ```` ``` ```` fences and extracts the
`[...]` JSON slice; `_validate_event_dict` drops malformed events. The loop
catches **every** exception around `backend.call` and counts it as "malformed".

**(b) Index scoring — `puremacro/narrative/indices/_llm_kernel.py`**

```python
llm_prob_kernel(records, *, provider, category, window="paragraph",
                language="en", max_calls=5000) -> Iterator[(date, mean_P)]
```
`provider` is any `LLMProvider` (ABC) with attributes `name`, `model` and
`score_paragraph(self, text, category) -> float` in `[0,1]`. Existing impls:
`MockProvider` (offline, deterministic), `AnthropicProvider` (uses the
`anthropic` SDK). Scores are cached in SQLite keyed on
`(provider.name, provider.model, prompt_hash, text_hash)`, and a `max_calls`
budget cap applies (lifted by `PUREMACRO_LLM_BUDGET=spend`). **The module
docstring already names "local Ollama" as an intended provider** — this design
delivers what was anticipated.

**Credentials.** `puremacro.credentials.require(service)` raises
`MissingCredentialError` on a miss. Local backends must **not** call it — they
need no key.

## 3. Non-goals (YAGNI)

- **No `torch`/`transformers` engine.** torch is ~2 GB and slow on CPU, which
  fights the no-budget goal; MLX + llama.cpp already cover the MLX/GGUF model
  zoo, including Google's Gemma. (Revisit later as a separate `[local-llm-full]`
  extra if demanded.)
- **No changes to embeddings** — `sentence-transformers` already runs locally
  for free.
- **No browser-playground local LLM** — a browser tab cannot reach `localhost`
  or load native engines; playground notebooks stay lexicon/MNL/Mock-based.
- **No new model training / fine-tuning.** Inference only.

## 4. Architecture

A single shared **engine layer** with three lazily-imported engines, fronted by
**thin wrappers** at each of the two call sites. Nothing in the call-site loops
(`score_llm`, `llm_prob_kernel`) changes except one small, justified robustness
fix (§7).

```
                       narrative/_local_engines.py
                       ┌──────────────────────────────────────┐
                       │  chat(model_id, prompt, *, opts) -> str│
                       │  resolve_engine("auto") -> LocalEngine │
                       │   ┌─────────┬───────────┬───────────┐ │
                       │   │MLXEngine│LlamaCppEng│ HTTPEngine │ │
                       │   │(mlx_lm) │(llama_cpp)│ (urllib)   │ │
                       │   └─────────┴───────────┴───────────┘ │
                       │  MODEL_ALIASES: friendly -> per-engine │
                       └──────────────▲─────────────▲──────────┘
                                      │             │
       narrative/scoring/llm.py       │             │  narrative/indices/_llm_kernel.py
       LocalBackend(_BackendBase)─────┘             └──LocalProvider(LLMProvider)
         .call(prompt)->str                            .score_paragraph(text,cat)->float
       OllamaBackend = preset(engine="ollama")        OllamaProvider = preset(engine="ollama")
```

### 4.1 Engine layer — `puremacro/narrative/_local_engines.py` (new)

A minimal internal protocol and three engines. **All heavy imports are inside
the engine `__init__`/first-use, never at module top**, so importing the module
(and thus `puremacro`) stays Pyodide-clean.

- `class LocalEngine` (informal protocol): `available() -> bool` (classmethod;
  cheap, no model load), `name: str`, and `complete(prompt, *, max_tokens,
  temperature, json_mode) -> str`.
- `MLXEngine` — lazy `from mlx_lm import load, generate`; loads the model once
  (cached on the instance), Apple-Silicon only (`available()` checks
  `platform.machine()` + import). *Exact `mlx_lm` API verified at plan time.*
- `LlamaCppEngine` — lazy `from llama_cpp import Llama`; `Llama.from_pretrained(
  repo_id, filename=...)` to auto-download a GGUF, or `Llama(model_path=...)`;
  `create_chat_completion(...)`. Cross-platform; `available()` checks the import.
- `HTTPEngine` — pure `urllib` via the new `_http.post_json`. Two wire formats
  selected by `api=`: Ollama-native (`POST {base_url}/api/chat`, supports
  `format:"json"`) and OpenAI-compatible (`POST {base_url}/v1/chat/completions`,
  supports `response_format`). `available()` does a fast `GET /api/tags`
  (Ollama) / `/v1/models` probe with a short timeout. Covers Ollama, LM Studio,
  vLLM, llama.cpp-server.
- `resolve_engine(engine, *, base_url)` — `"auto"` tries, in order:
  - darwin/arm64: `MLXEngine → LlamaCppEngine → HTTPEngine`
  - otherwise: `LlamaCppEngine → HTTPEngine`
  and raises `LocalLLMUnavailable` (actionable message, §7) if none is usable.
  Explicit values: `"mlx" | "llamacpp" | "ollama" | "openai"`.
- `MODEL_ALIASES` — a small dict mapping a friendly canonical name to a
  per-engine id, so one `model="qwen2.5-3b-instruct"` works across engines:
  ```
  "qwen2.5-3b-instruct": {
     "mlx":      "mlx-community/Qwen2.5-3B-Instruct-4bit",
     "llamacpp": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "*q4_k_m.gguf"),
     "ollama":   "qwen2.5:3b",
  }, ...   # also gemma2-2b (Google), llama3.2-3b (Meta), phi3.5 (MS)
  ```
  An unrecognized `model` is passed through verbatim to the engine (escape
  hatch for any model id the user already has).

### 4.2 Call-site wrappers

- **`narrative/scoring/llm.py`**: add `LocalBackend(_BackendBase)` whose
  `call(prompt)` delegates to `resolve_engine(...).complete(...)` with
  `json_mode=True`. Add `OllamaBackend = ` a thin subclass/factory pinned to
  `engine="ollama"` (keeps the friendly name from the brainstorm).
- **`narrative/indices/_llm_kernel.py`**: add `LocalProvider(LLMProvider)`
  (`name` = the resolved engine name so the SQLite cache partitions correctly,
  `model` = the canonical model id) whose `score_paragraph(text, category)`
  delegates to the same engine and robustly extracts the first float in `[0,1]`.
  Add `OllamaProvider` preset.
- **`get_default_backend()`** (in scoring) / **`get_default_provider()`** (in
  indices): return the best available local engine wrapper, or `MockProvider`
  (indices) / a `MockBackend` (scoring) when none is installed — so notebooks
  "just run free". Prints a one-line note of which engine was chosen.

### 4.3 Shared HTTP helper — `puremacro/_http.py`

Add `post_json(url, payload, *, timeout, headers=None) -> dict` (urllib POST +
the same one-shot SSL fallback already in `_request`). Only the new HTTPEngine
calls it; the existing paid Anthropic/OpenAI backends are left untouched
(small blast radius).

## 5. Public API (exact)

```python
# puremacro.narrative.scoring
LocalBackend(model="qwen2.5-3b-instruct", *, engine="auto",
             base_url="http://localhost:11434", timeout=120.0,
             max_tokens=1024, temperature=0.0, json_mode=True)
OllamaBackend(model="qwen2.5:3b", *, base_url="http://localhost:11434", ...)
get_default_backend(model="qwen2.5-3b-instruct") -> _BackendBase   # or MockBackend

# puremacro.narrative.indices
LocalProvider(model="qwen2.5-3b-instruct", *, engine="auto",
              base_url="http://localhost:11434", timeout=120.0)
OllamaProvider(model="qwen2.5:3b", *, base_url="http://localhost:11434", ...)
get_default_provider(model="qwen2.5-3b-instruct") -> LLMProvider   # or MockProvider
```
No API key parameter is required. `base_url` only matters for the HTTP engine.

## 6. Robustness & UX

- **JSON mode** for event extraction: pass `format:"json"` (Ollama),
  `response_format={"type":"json_object"}` (OpenAI-compat); for MLX/llama.cpp,
  rely on the existing robust `_parse_response`. Keeps small 3B models parseable.
- **Bigger default timeout** (120 s) — CPU inference is slow. The index
  kernel's `max_calls` budget still bounds total work.
- **Model download UX**: MLX and llama.cpp auto-download by id from Hugging Face
  on first use (cached under `~/.cache/huggingface`); no manual `.gguf` handling.
  Document the one-time download size (~1.5–2 GB for a 3B-Q4 model).

## 7. The one change to existing code: backend-unavailable errors

Today `score_llm` wraps `backend.call` in `except Exception: drop as malformed`.
If a local engine/server is unreachable, **every** record is silently dropped
and the user sees only "dropped N malformed events" — confusing.

**Change:** introduce `class LocalLLMUnavailable(RuntimeError)` (raised by
`resolve_engine`/engines on a connection/availability failure, with an
actionable message, e.g. *"No local LLM engine available. Install one with
`pip install puremacro[local-llm]` (MLX on Apple Silicon, or llama-cpp-python),
or start Ollama and run `ollama pull qwen2.5:3b`."*). In `score_llm`, let
`LocalLLMUnavailable` (and a shared `BackendUnavailable` base, which the paid
backends can also raise for connection errors) **propagate** instead of being
swallowed; keep dropping genuine parse/validation errors as today. This is a
net improvement for *all* backends (a down cloud API now surfaces too). It is
the only behavioral change to existing functions, and it is additive
(new exception types; existing happy paths unchanged).

## 8. Packaging & exports

- New extra in `pyproject.toml` (mirrors the existing `[backend]` shape):
  ```toml
  local-llm = [
    "llama-cpp-python>=0.3",            # cross-platform GGUF (CPU/Metal/CUDA)
    "mlx-lm>=0.20; sys_platform == 'darwin'",   # Apple-Silicon GPU
  ]
  ```
  *(version floors verified at plan time against what installs cleanly here)*
- Exports: `narrative/scoring/__init__.py` adds
  `LocalBackend, OllamaBackend, get_default_backend`;
  `narrative/indices/__init__.py` adds
  `LocalProvider, OllamaProvider, get_default_provider`; both re-export
  `LocalLLMUnavailable`.
- `credentials.py`: no new service (local needs no key); add a one-line note in
  the relevant docstring/README that local backends bypass credentials.
- Version bump 0.91.0 → 0.92.0 in `pyproject.toml` **and**
  `puremacro/__init__.__version__`; CHANGELOG + ARCHITECTURE entries.

## 9. Pyodide / lazy-import discipline

- `narrative/_local_engines.py` imports `mlx_lm`/`llama_cpp` **only inside**
  engine methods, never at module top. The module top-level imports are
  stdlib + `puremacro._http` only.
- Extend the existing Pyodide gate (`tests/test_pyodide_compat.py` /
  `tests/test_pyodide/`) to assert `puremacro.narrative._local_engines`,
  `narrative.scoring`, and `narrative.indices` import with `mlx_lm`/`llama_cpp`
  **blocked** via the meta_path finder (the real Pyodide condition). The HTTP
  engine imports fine in-browser (it just can't reach localhost at runtime,
  which is expected).

## 10. Testing & validation

- **Offline unit tests (no engine, deterministic, CI-safe):**
  - HTTPEngine against a threaded `http.server` fixture mimicking Ollama's
    `/api/chat` and an OpenAI-compatible `/v1/chat/completions` — exercises the
    real `urllib` POST, both wire formats, JSON mode, and error mapping.
  - `resolve_engine("auto")` selection logic with engines monkeypatched
    available/unavailable; `LocalLLMUnavailable` message asserted.
  - `MODEL_ALIASES` resolution per engine; passthrough for unknown ids.
  - `LocalBackend`/`LocalProvider` drive `score_llm`/`llm_prob_kernel`
    end-to-end against a stub engine (validates the wiring + the SQLite cache
    partitions on engine name).
  - The `score_llm` change: `BackendUnavailable` propagates; parse errors still
    drop.
- **Opt-in live tests (skip if the engine is absent — never assert against an
  empty/missing engine):** one test per engine (`mlx`, `llamacpp`, `ollama`),
  each `pytest.importorskip(...)` / probe-and-skip, marked
  `@pytest.mark.local_llm` (+ `slow`). On this Mac these run after
  `pip install puremacro[local-llm]` (MLX) and a small model pull.
- **Structural + directional validation (the "science"):** because LLM output
  is not bit-exact across engines, the oracle is *structural* (valid float in
  `[0,1]` / parseable event JSON for a canned prompt) and *directional* — on a
  synthetic corpus with an injected uncertainty spike (à la notebook 11's
  EPU/MPU demo), the local provider's mean P(uncertainty) is materially higher
  in-window than out-of-window, and tracks the free lexicon kernel's spike.
  Runs live when an engine is present, skips otherwise.

## 11. Demo & docs

- **`examples/narrative_local_llm.py`** (new, runnable): end-to-end on a tiny
  synthetic corpus; calls `get_default_backend()`/`get_default_provider()`;
  prints the chosen engine, or the actionable install message if none. A light
  smoke test imports/executes it with `MockProvider` (no engine required).
- **Desktop notebook** in the VFI/desktop showcase suite (`puremacro/notebooks/`,
  jupytext-paired, built by `tools/build_notebooks.py`) — **not** in the
  browser-playground content set, and named outside the playground's
  `[0-1][0-9]_*.ipynb` glob so the playground build never grabs it. It uses
  `get_default_backend()` so the execute-all slow gate passes with `MockProvider`
  in CI while showing real local inference on a user's machine. (Per the paired
  builder + no-clobber rules, the `.py` is source of truth; never run the
  builder over an executed `.ipynb`.)
- **README**: a "Run the LLM features for free (local models)" section — install
  Ollama *or* `pip install puremacro[local-llm]`, pull/choose a small model, swap
  in `LocalBackend(engine="auto")`. Short `docs/` how-to with model-size guidance
  and the Windows/macOS notes.

## 12. File-by-file change list (for the plan)

New:
- `puremacro/narrative/_local_engines.py` — engine layer + `MODEL_ALIASES` +
  `resolve_engine` + `LocalLLMUnavailable`.
- `examples/narrative_local_llm.py`.
- `puremacro/notebooks/<name>.py` (+ built `.ipynb`) — desktop demo.
- `tests/test_narrative/test_local_engines.py`, `..._local_backends.py`,
  `tests/test_pyodide/test_local_engines_importable.py`, a live
  `tests/test_narrative/test_local_llm_live.py` (skips), and a directional
  validation test.

Modified:
- `puremacro/_http.py` (+ `post_json`).
- `puremacro/narrative/scoring/llm.py` (+ `LocalBackend`, `OllamaBackend`,
  `get_default_backend`, a new `MockBackend` returning `"[]"` for the no-engine
  fallback, `BackendUnavailable` handling) and its `__init__.py`.
- `puremacro/narrative/indices/_llm_kernel.py` (+ `LocalProvider`,
  `OllamaProvider`, `get_default_provider`) and `indices/__init__.py`.
- `pyproject.toml` (`[local-llm]` extra, version), `puremacro/__init__.py`
  (version), `CHANGELOG.md`, `ARCHITECTURE.md`, `README.md`,
  `tests/test_pyodide*` (gate extension).

## 13. Risks & open questions

- **Engine API drift.** `mlx_lm.generate` and `llama_cpp` signatures change
  across versions. *Mitigation:* verify exact signatures against the installed
  versions when writing the plan (standing rule); pin sensible floors in the
  extra; wrap each engine so a signature change is localized.
- **Small-model JSON quality.** A 2–3B model may emit imperfect JSON for event
  extraction. *Mitigation:* JSON mode + the existing tolerant `_parse_response`;
  document Qwen2.5-3B-Instruct as the recommended default for the JSON task and
  gemma2-2b as the lightest option.
- **Speed.** CPU inference on a long corpus is slow. *Mitigation:* the index
  kernel's SQLite cache + `max_calls` cap; docs steer users to small models and
  (on Mac) MLX; the demo corpora are tiny.
- **Decided defaults** (vetoable at spec review): include the `score_llm`
  error-propagation change (§7); ship *both* the example script and the desktop
  notebook (§11). Default model `qwen2.5-3b-instruct` (confirm at plan time).

## 14. Decision log (from brainstorming)

1. Direction: extend capability + equity — let the paid LLM work run free locally.
2. Mechanism: local inference, **not** a hosted proxy; no API key.
3. Engines: **MLX + llama.cpp + HTTP (Ollama/OpenAI-compat)**, `engine="auto"`;
   **no torch** by default.
4. Naming: `LocalBackend`/`LocalProvider` primary; `Ollama*` presets retained.
5. `get_default_*` falls back to Mock so notebooks/CI/playground stay green.
6. Cover **both** paid call sites (events + index), not just one.
