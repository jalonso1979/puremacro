"""LLM-backed event scoring via direct HTTP calls.

Two backends share an interface so the rest of the package is
backend-agnostic:

    backend = AnthropicBackend(model="claude-sonnet-4-6")
    backend = OpenAIBackend(model="gpt-4o-mini")
    events = score_llm(text_iter, backend=backend, kind="fiscal", country="USA")

API keys are resolved via ``puremacro.credentials`` (env vars,
TOML config, or explicit kwarg). The HTTP layer uses ``urllib.request`` only —
works in Pyodide via the JS-fetch proxy with no SDK dependency.

Strict JSON-schema validation: malformed events are dropped (a warning
counter is kept) rather than raising — pipelines stay running.

The ``kind`` parameter (default ``"fiscal"``) selects one of five
prompt templates. The language preamble lets a single prompt extract
events from non-English text natively (Anthropic / OpenAI both handle
multilingual extraction).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from puremacro import credentials
from .._local_engines import BackendUnavailable
from ..types import NarrativeEvent, VALID_SIGNS


_LANGUAGE_PREAMBLE = (
    "The text below may be in any language; the field 'language' is "
    "{language} (ISO-639-1). Extract events regardless of language. "
    "Return all field labels in English."
)


_PROMPTS = {
    "fiscal": """You are extracting narrative fiscal-policy events from
press text. From the input below, output a JSON array of zero or more
events with this exact schema (one JSON object per event):

  {{
    "magnitude_usd_bn": float,             // size in USD billions (0 if not stated)
    "target": "investment" | "consumption" | "both",
    "subtarget": string | null,            // e.g. "defense", "infra", "transfers"
    "sign": -1 | 0 | +1,                   // +1 expansionary, -1 contractionary
    "confidence": float in [0, 1],
    "excerpt": string,                     // <=240 chars from the source
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include events that announce a *discrete change* in government
spending (consumption or capital), not retrospective reporting or
forecasts. Output ONLY the JSON array, no surrounding prose.

Field rules for the timing extraction:

- ``implementation_date``: if the text states an effective or
  implementation date for the announced policy, return it as an ISO
  ``YYYY-MM-DD`` string. If multiple effective dates are mentioned,
  return the FIRST one. Return ``null`` if no effective date is mentioned.

- ``horizon_quarters``: only return this if ``implementation_date`` is
  null AND the text gives a vague horizon. Return an integer 0–16
  representing quarters between announcement and first effective quarter.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "monetary": """You are extracting narrative MONETARY-policy events
from central-bank text. From the input below, output a JSON array of
zero or more events with this exact schema:

  {{
    "magnitude_bps": float,                // policy-rate change in basis points (0 if not stated)
    "target": "policy_rate" | "asset_purchase" | "forward_guidance"
              | "fx_intervention" | "lending_facility",
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 hawkish/tighten, -1 dovish/ease, 0 neutral
    "hawkish_dovish_prob": float in [0, 1],
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include discrete monetary-policy decisions or unambiguous
forward-guidance announcements. Skip retrospective discussion and
forecasts. Output ONLY the JSON array.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "macropru": """You are extracting MACROPRUDENTIAL-policy events.
JSON array, one object per event:

  {{
    "target": "capital_buffer" | "ltv_dsti" | "sector_limit" | "reserve_requirement",
    "magnitude_pct": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 tightening, -1 loosening
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Only include announced policy actions, not commentary. Output JSON
array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "fx": """You are extracting FX-INTERVENTION or peg-change events.
JSON array, one object per event:

  {{
    "target": "intervention" | "peg_change",
    "magnitude_usd_bn": float,
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 buying domestic / defending, -1 selling
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
    "structural": """You are extracting STRUCTURAL-REFORM events. JSON
array, one object per event:

  {{
    "target": "labor" | "product_market" | "trade" | "tax_admin",
    "magnitude_z": float,                  // qualitative scale, ~[0, 3]
    "subtarget": string | null,
    "sign": -1 | 0 | +1,                   // +1 liberalizing, -1 restrictive
    "confidence": float in [0, 1],
    "excerpt": string,
    "implementation_date": "YYYY-MM-DD" | null,
    "horizon_quarters": int 0..16 | null
  }}

Output JSON array only.

{language_preamble}

Country: {country}
Date: {date}

Text:
\"\"\"
{text}
\"\"\"
""",
}


_MAGNITUDE_KEY_BY_KIND = {
    "fiscal":     "magnitude_usd_bn",
    "monetary":   "magnitude_bps",
    "macropru":   "magnitude_pct",
    "fx":         "magnitude_usd_bn",
    "structural": "magnitude_z",
}

_MAGNITUDE_UNIT_BY_KIND = {
    "fiscal":     "USD_bn",
    "monetary":   "bps",
    "macropru":   "ratio",
    "fx":         "USD_bn",
    "structural": "z",
}


def _build_prompt(*, kind, language, country, date, text):
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")
    return _PROMPTS[kind].format(
        country=country,
        date=date,
        text=str(text)[:6000],
        language_preamble=_LANGUAGE_PREAMBLE.format(language=language),
    )


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
@dataclass
class _BackendBase:
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0

    def call(self, prompt: str) -> str:  # pragma: no cover (abstract)
        raise NotImplementedError


class AnthropicBackend(_BackendBase):
    """Anthropic Messages API client (urllib-based)."""

    def __init__(self, model: str = "claude-sonnet-4-6", *,
                 api_key: str | None = None, **kw):
        super().__init__(model=model, **kw)
        self.api_key = credentials.require("anthropic", explicit=api_key)

    def call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        return ""


class OpenAIBackend(_BackendBase):
    """OpenAI Chat Completions client (urllib-based)."""

    def __init__(self, model: str = "gpt-4o-mini", *,
                 api_key: str | None = None, **kw):
        super().__init__(model=model, **kw)
        self.api_key = credentials.require("openai", explicit=api_key)

    def call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")


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


# ---------------------------------------------------------------------------
# Top-level scoring loop
# ---------------------------------------------------------------------------
def _parse_response(text: str) -> list[dict]:
    """Best-effort JSON-array extraction from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            i = text.index("[")
            j = text.rindex("]")
            parsed = json.loads(text[i:j + 1])
        except (ValueError, json.JSONDecodeError):
            return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _validate_event_dict(d: dict, *, kind: str) -> bool:
    from ..types import VALID_TARGETS_BY_KIND
    if not isinstance(d, dict):
        return False
    if d.get("target") not in VALID_TARGETS_BY_KIND[kind]:
        return False
    try:
        if int(d.get("sign", 99)) not in VALID_SIGNS:
            return False
    except (ValueError, TypeError):
        return False
    mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
    if not isinstance(d.get(mag_key, None), (int, float)):
        return False
    if not isinstance(d.get("confidence", None), (int, float)):
        return False
    return True


def _profile_from_timing_fields(
    ev_dict: dict,
    *,
    announcement,
) -> list[tuple[pd.Timestamp, float]]:
    """Build implementation_profile from the LLM's timing fields.

    Prefers ``implementation_date`` (ISO string) over ``horizon_quarters``
    (int 0–16 fallback). Both null or unparseable → empty list.

    Refuses to emit a profile that predates the announcement quarter.
    """
    impl_date_str = ev_dict.get("implementation_date")
    horizon_q = ev_dict.get("horizon_quarters")

    if impl_date_str:
        try:
            impl_q = pd.Period(pd.Timestamp(impl_date_str), freq="Q").to_timestamp()
            ann_q = pd.Period(pd.Timestamp(announcement), freq="Q").to_timestamp()
            if impl_q >= ann_q:
                return [(impl_q, 1.0)]
        except (ValueError, TypeError):
            pass
        return []

    if horizon_q is not None:
        try:
            h = int(horizon_q)
        except (ValueError, TypeError):
            return []
        if 0 <= h <= 16:
            ann_q = pd.Period(pd.Timestamp(announcement), freq="Q")
            impl_q = (ann_q + h).to_timestamp()
            return [(impl_q, 1.0)]
    return []


def score_llm(
    text_iter: Iterable[tuple],
    *,
    backend: _BackendBase,
    kind: str = "fiscal",
    language: str = "en",
    country: str = "USA",
    dry_run: bool = False,
) -> list[NarrativeEvent]:
    """Score each record with an LLM, parameterized by ``kind`` and ``language``.

    ``text_iter`` accepts both 3-tuple ``(date, text, source_url)`` and
    4-tuple ``(date, text, source_url, metadata)`` records. When a 4-tuple
    is passed, ``metadata["language"]`` overrides the function-level
    ``language`` per record.

    With ``dry_run=True``, return an empty list and print an estimated
    cost (rough — based on character count → token approximation).

    A ``BackendUnavailable`` error (e.g. a local engine/server that is
    unreachable) propagates to the caller rather than being dropped as malformed.
    """
    if kind not in _PROMPTS:
        raise ValueError(f"kind {kind!r} not in {sorted(_PROMPTS)}")

    items = list(text_iter)
    if dry_run:
        chars = sum(len(r[1]) for r in items)
        approx_tokens = chars / 4.0
        print(f"[score_llm dry-run kind={kind} lang={language}] "
              f"{len(items)} items, ~{approx_tokens:,.0f} input tokens, "
              f"~{len(items) * 200:,.0f} output tokens.")
        return []

    out: list[NarrativeEvent] = []
    n_dropped_malformed = 0
    for record in items:
        if len(record) == 4:
            date, text, source_url, meta = record
            record_lang = (meta or {}).get("language", language)
        else:
            date, text, source_url = record
            record_lang = language
        prompt = _build_prompt(
            kind=kind, language=record_lang, country=country,
            date=str(date), text=text,
        )
        try:
            response = backend.call(prompt)
        except BackendUnavailable:
            raise
        except Exception:
            n_dropped_malformed += 1
            continue
        for ev_dict in _parse_response(response):
            if not _validate_event_dict(ev_dict, kind=kind):
                n_dropped_malformed += 1
                continue
            mag_key = _MAGNITUDE_KEY_BY_KIND[kind]
            magnitude_unit = _MAGNITUDE_UNIT_BY_KIND[kind]
            profile = _profile_from_timing_fields(ev_dict, announcement=date)
            out.append(NarrativeEvent(
                date=pd.Timestamp(date),
                country=country,
                magnitude=float(ev_dict[mag_key]),
                magnitude_unit=magnitude_unit,
                target=ev_dict["target"],
                subtarget=ev_dict.get("subtarget"),
                sign=int(ev_dict["sign"]),
                confidence=float(ev_dict["confidence"]),
                source_text=str(ev_dict.get("excerpt", text))[:500],
                source_url=str(source_url),
                scoring_method="llm",
                implementation_profile=profile,
                kind=kind,
                language=record_lang,
            ))
    if n_dropped_malformed:
        print(f"[score_llm] dropped {n_dropped_malformed} malformed events.")
    return out


__all__ = ["AnthropicBackend", "OpenAIBackend", "score_llm",
           "LocalBackend", "OllamaBackend", "MockBackend", "get_default_backend",
           "_PROMPTS", "_build_prompt"]
