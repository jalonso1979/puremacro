# puremacro 0.68.0 Implementation Plan — F1 Slice A (SE Asia + Africa CB connectors)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship F1 Slice A as `0.68.0`: 6 new central-bank narrative connectors (`bi`, `bnm`, `bsp`, `cbn`, `cbe`, `cbk`) for SE Asia and Africa. Each declares `PARSER_SCHEMA_VERSION` and `FALLBACK_POLICY`, uses `fetch_with_fallback`, and emits `parser_schema_mismatch` telemetry — first-class members of the Slice A + B contracts from inception.

**Architecture:** Five sub-slices landing in dependency order. Sub-slice 1 ships the framework tests (initially failing) + `bi` as the first concrete connector. Sub-slices 2–3 add the remaining 5 connectors, each verified per-site (landmarks, fallback policy, speeches-archive availability) rather than transcribed from the spec. Sub-slice 4 parametrizes cross-connector tests over the 6 names. Sub-slice 5 ships the demo notebook + version bump.

**Tech Stack:** Python ≥3.12. `sqlite3` stdlib. `pandas`. The new `_fallback.fetch_with_fallback` (Slice B) + `_schema_check.assert_landmarks` (Slice A) + `_telemetry.log_event` (Slice B) primitives. No new runtime dependencies. Pyodide-pure (Playwright stays lazy-imported through Slice B's existing guard).

**Spec:** `docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md`

---

## File map

### New files
- `puremacro/puremacro/narrative/sources/bi.py` — Bank Indonesia (IDN).
- `puremacro/puremacro/narrative/sources/bnm.py` — Bank Negara Malaysia (MYS).
- `puremacro/puremacro/narrative/sources/bsp.py` — Bangko Sentral ng Pilipinas (PHL).
- `puremacro/puremacro/narrative/sources/cbn.py` — Central Bank of Nigeria (NGA).
- `puremacro/puremacro/narrative/sources/cbe.py` — Central Bank of Egypt (EGY).
- `puremacro/puremacro/narrative/sources/cbk.py` — Central Bank of Kenya (KEN).
- `puremacro/puremacro/narrative/sources/_fixtures/bi_decision_v1.{html,xml}` and (if shipped) `bi_speeches_v1.{html,xml}`.
- (Same fixture-pair pattern for bnm, bsp, cbn, cbe, cbk.)
- `tests/test_narrative_f1_slice_a/__init__.py` (empty).
- `tests/test_narrative_f1_slice_a/test_parser_schema_versions.py` — each of 6 declares `PARSER_SCHEMA_VERSION` as int.
- `tests/test_narrative_f1_slice_a/test_fallback_policies.py` — each declares `FALLBACK_POLICY` in `SUPPORTED_STAGES`.
- `tests/test_narrative_f1_slice_a/test_per_connector_smoke.py` — parametrized: import + iter callable.
- `tests/test_narrative_f1_slice_a/test_landmark_fixtures.py` — parametrized: each fixture passes its connector's landmark check.
- `tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py` — parametrized: `iter_<cb>_decision` against fixture yields ≥1 tuple.
- `tests/test_narrative_f1_slice_a/test_coverage_assertion.py` — AST: 6 modules each have `iter_<cb>_decision` + call `assert_landmarks` + call `fetch_with_fallback`.
- `tools/make_notebook_R5_03.py` — paired builder.
- `notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb` — executed demo.

### Modified files
- `puremacro/pyproject.toml` — bump `version` to `"0.68.0"`.
- `puremacro/puremacro/__init__.py` — bump `__version__` to `"0.68.0"`.
- `puremacro/CHANGELOG.md` — prepend `## 0.68.0 (2026-05-26)` section.
- `puremacro/ARCHITECTURE.md` — append "F1 Slice A — SE Asia + Africa CBs (0.68.0+)" subsection.
- `tests/test_credentials/test_service_registry.py` — version smoke test bumped to `test_puremacro_version_is_068`.
- `tests/test_signal_contract/test_schema_extension.py` — same version-pin update (Slice B's implementer caught this parallel pin; same accommodation here).

### Working assumptions (verified 2026-05-26 via signature dumps + grep)

- `puremacro.narrative.sources._fallback.fetch_with_fallback(url, *, policy, source, timeout=30.0, use_cache=True) -> str` — Slice B (commit `8cbac28`). `source=` is REQUIRED (no default).
- `puremacro.narrative.sources._fallback.FallbackExhaustedError` — raised when every stage fails.
- `puremacro.narrative.sources._fallback.SUPPORTED_STAGES = frozenset({"live", "wayback", "playwright"})`.
- `puremacro.narrative.sources._schema_check.assert_landmarks(text, *, source, expected_version, landmarks) -> None` — Slice A. Raises `ParserSchemaMismatchError` on missing landmark.
- `puremacro.narrative.sources._telemetry.log_event(*, source, outcome, fallback_used="none") -> None` — Slice B. Validates `outcome` against `VALID_OUTCOMES = {success, 404, timeout, ssl_fail, server_5xx, wayback_no_snapshot, playwright_unavailable, parser_schema_mismatch, other_network_error}`.
- `puremacro.narrative.sources._ratedoc.strip_html(html) -> str` — crude HTML→text helper. Existing.
- `puremacro.narrative.sources._rss_filtered.iter_rss_filtered(url, *, bank_code, country, doctype, language, fetch_body=False) -> Iterator[tuple]` — RSS shortcut used by ~10 existing connectors (see `bcra.py:8` for the 8-line canonical example).
- Pre-existing connector pattern: each yields `(date, text, source_url)` 3-tuples OR `(date, text, source_url, metadata)` 4-tuples per RETRY_POLICY.md §4.2. Slice A connectors use the 4-tuple. We follow the 4-tuple for new connectors.
- `pyproject.toml` currently `version = "0.67.0"`, `requires-python = ">=3.12"`. Bump version to `"0.68.0"`; leave Python pin.
- `puremacro/__init__.py` currently `__version__ = "0.67.0"`; bump to `"0.68.0"`.
- Baseline test counts (post-Slice-B): test_narrative_fallback ~35, test_narrative_telemetry ~22, test_cache_db ~30, test_credentials ~26, test_narrative_schema_checks ~31, test_signal_contract ~37, test_pyodide_compat 2.
- Branch is `main`. Commits land directly on `main` per Slices 1, A, B convention.

### Per-CB verification protocol (critical for Tasks 2–7)

The spec **explicitly defers** per-site landmarks / fallback policies / speeches-archive availability to implementation time. For each per-connector task, the implementer:

1. **Use `WebFetch` (or `curl` via Bash) to GET the hypothesised decision URL.** If the response is 200 with HTML, the site is "live"-accessible — proceed with `FALLBACK_POLICY = ("live",)`.
2. **If the live fetch returns 403 / 429 / a Cloudflare interstitial / empty body**, the site is WAF-blocked from the agent's IP. Two options:
   - Try Wayback (`https://web.archive.org/web/2024*/{decision_url}`) — if there's a recent snapshot, set `FALLBACK_POLICY = ("live", "wayback")` and use the Wayback snapshot for the golden fixture.
   - If Wayback is also empty, the site genuinely needs Playwright — set `FALLBACK_POLICY = ("playwright",)` and synthesize a minimal real-shape fixture by hand (skeleton HTML with the expected landmark strings; document the synthesis in the commit message).
3. **From the successful fetch, save the body as the golden fixture** at `puremacro/narrative/sources/_fixtures/<cb>_decision_v1.{html,xml}`. Pick 2-3 substring landmarks that reliably identify the listing page (e.g., "Monetary Policy Statement", "Bank Indonesia", date pattern).
4. **Repeat for the speeches URL.** If the speeches URL returns nothing usable (e.g., the CB doesn't publish English-language speeches in a separate archive), drop `iter_<cb>_speeches` from the module, drop the speeches fixture, document the omission in the module docstring + the per-CB commit message + the CHANGELOG.

This is the verified-not-hypothesised pattern. The plan provides HYPOTHESISED URLs / landmarks below; the implementer overrides with VERIFIED values during execution.

---

## Sub-slice 1 — Framework + first connector

(Tasks 1-2.)

## Task 1: Framework tests (initially failing)

**Files:**
- Create: `tests/test_narrative_f1_slice_a/__init__.py` (empty).
- Create: `tests/test_narrative_f1_slice_a/test_parser_schema_versions.py`.
- Create: `tests/test_narrative_f1_slice_a/test_fallback_policies.py`.

These tests are parametrized over the 6 connector names; they fail for every connector until Tasks 2-7 ship each one. The pattern is the same as Slice A's `test_landmark_assertions.py` and Slice B's `test_per_connector_policies.py`.

- [ ] **Step 1: Create the test directory and write the failing tests**

Create `tests/test_narrative_f1_slice_a/__init__.py` (empty file).

Create `tests/test_narrative_f1_slice_a/test_parser_schema_versions.py`:

```python
"""F1 Slice A — each of the 6 SE Asia + Africa CB connectors declares
PARSER_SCHEMA_VERSION (adopting the Slice A schema-versioning contract
from inception)."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = [
    "bi", "bnm", "bsp", "cbn", "cbe", "cbk",
]


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_declares_parser_schema_version(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "PARSER_SCHEMA_VERSION"), (
        f"{name}.py must declare PARSER_SCHEMA_VERSION (F1 Slice A contract)"
    )
    assert isinstance(mod.PARSER_SCHEMA_VERSION, int)
    assert mod.PARSER_SCHEMA_VERSION >= 1


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_imports_assert_landmarks(name):
    """AST scan: the module imports or references assert_landmarks."""
    import ast
    import pathlib
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    src = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "assert_landmarks" for alias in node.names):
                found = True
                break
        if isinstance(node, ast.Name) and node.id == "assert_landmarks":
            found = True
            break
    assert found, (
        f"{name}.py must import or reference `assert_landmarks` "
        f"(F1 Slice A contract)"
    )
```

Create `tests/test_narrative_f1_slice_a/test_fallback_policies.py`:

```python
"""F1 Slice A — each connector declares FALLBACK_POLICY (adopting
the Slice B fallback contract from inception)."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = [
    "bi", "bnm", "bsp", "cbn", "cbe", "cbk",
]


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_declares_fallback_policy(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "FALLBACK_POLICY"), (
        f"{name}.py must declare FALLBACK_POLICY (F1 Slice A contract)"
    )
    assert isinstance(mod.FALLBACK_POLICY, tuple), (
        f"{name}.FALLBACK_POLICY must be a tuple"
    )
    assert len(mod.FALLBACK_POLICY) >= 1, (
        f"{name}.FALLBACK_POLICY must have at least one stage"
    )


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_fallback_policy_stages_are_supported(name):
    from puremacro.narrative.sources._fallback import SUPPORTED_STAGES
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    bad = set(mod.FALLBACK_POLICY) - SUPPORTED_STAGES
    assert not bad, (
        f"{name}.FALLBACK_POLICY contains unsupported stages: {sorted(bad)}. "
        f"Supported: {sorted(SUPPORTED_STAGES)}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_f1_slice_a/ -v
```
Expected: every parametrized test fails with `ModuleNotFoundError` (the 6 connector modules don't exist yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_narrative_f1_slice_a/__init__.py tests/test_narrative_f1_slice_a/test_parser_schema_versions.py tests/test_narrative_f1_slice_a/test_fallback_policies.py
git commit -m "$(cat <<'EOF'
test(0.68.0): F1 Slice A test scaffolding (PARSER_SCHEMA_VERSION + FALLBACK_POLICY)

Parametrized tests over the 6 SE Asia + Africa CB connectors (bi, bnm,
bsp, cbn, cbe, cbk). Initially failing — each module doesn't exist
yet. Tasks 2-7 add the connectors; each task's commit will see
incrementally more tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `bi.py` — Bank Indonesia (canonical exemplar)

**Files:**
- Create: `puremacro/puremacro/narrative/sources/bi.py`.
- Create: `puremacro/puremacro/narrative/sources/_fixtures/bi_decision_v1.{html|xml}`.
- (Optional, if speeches archive exists in English) Create `bi_speeches_v1.{html|xml}`.

This is the CANONICAL example. Tasks 3-7 follow this pattern verbatim with per-CB substitutions.

### Hypothesised URLs (VERIFY at execution time)
- Decision listing: `https://www.bi.go.id/en/publikasi/ruang-media/news-release/Default.aspx` OR `https://www.bi.go.id/en/publikasi/RSS/RSS-news.xml`
- Speech archive: `https://www.bi.go.id/en/publikasi/ruang-media/speech/Default.aspx`

### Hypothesised `FALLBACK_POLICY`
`("live",)` — BI's English site is generally publicly accessible.

### Per-CB verification protocol (executes the spec's verified-not-hypothesised pattern)

- [ ] **Step 1: Verify the live decision URL is fetchable**

Use the `WebFetch` tool (or `curl` via Bash) to GET the hypothesised decision URL. Three outcomes:

A) **HTTP 200 + non-empty HTML/XML body** → site is live-accessible. Note 2-3 substrings that reliably appear in the page (e.g., "Bank Indonesia", "BI 7-Day Reverse Repo Rate", "Monetary Policy"). Save the body verbatim as the fixture (next step). `FALLBACK_POLICY = ("live",)`.

B) **HTTP 403 / 429 / Cloudflare interstitial** → try `https://web.archive.org/web/2024*/{decision_url}` via Wayback. If a snapshot exists, fetch it and proceed with `FALLBACK_POLICY = ("live", "wayback")`. Save the Wayback snapshot body as the fixture.

C) **No live, no Wayback** → synthesize a minimal real-shape fixture by hand (skeleton HTML containing the expected landmark strings) and set `FALLBACK_POLICY = ("playwright",)`. Document the synthesis in the commit message.

- [ ] **Step 2: Save the decision fixture**

```bash
# Example for the live-success path:
mkdir -p puremacro/puremacro/narrative/sources/_fixtures
curl -A "Mozilla/5.0 (puremacro/narrative)" \
     "<verified_decision_url>" \
     > puremacro/puremacro/narrative/sources/_fixtures/bi_decision_v1.html
```

If the response is RSS/XML, save as `bi_decision_v1.xml`. If it's HTML, `.html`. Keep the file ≤ 100 KB if possible (truncate trailing entries if the listing is huge — only the most-recent N entries needed for the landmark check).

- [ ] **Step 3: Repeat Steps 1-2 for the speech URL**

If the speech URL also returns a usable archive in English, save `bi_speeches_v1.{html|xml}`. If it doesn't (e.g., speeches are only in Indonesian, or there's no separate speeches archive), SKIP — drop `iter_bi_speeches` from the module entirely and document in the module docstring.

- [ ] **Step 4: Write the failing per-connector test**

The framework tests from Task 1 fail for `bi`. Additionally write a per-connector smoke test in the same suite to confirm the iter function returns expected shape on the fixture. Create / append to `tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py`:

```python
"""F1 Slice A — each connector's iter_<cb>_decision yields ≥1 tuple
when run against its golden fixture."""
from __future__ import annotations

import importlib
import pathlib

import pytest


_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _decision_fixture_text(cb: str) -> str:
    """Return the bytes of <cb>_decision_v1.html or .xml, whichever exists."""
    for ext in ("html", "xml"):
        p = _FIXTURE_DIR / f"{cb}_decision_v1.{ext}"
        if p.exists():
            return p.read_text()
    raise FileNotFoundError(f"no decision fixture for {cb!r}")


# Will be expanded in Task 8 to all 6 connectors; for Task 2 we test bi only.
def test_bi_decision_fixture_yields_at_least_one(monkeypatch):
    from puremacro.narrative.sources import bi, _fallback
    text = _decision_fixture_text("bi")
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: text,
    )
    records = list(bi.iter_bi_decision())
    assert len(records) >= 1, (
        f"bi.iter_bi_decision returned {len(records)} records against "
        f"fixture; expected ≥1."
    )
    # Each record is a tuple of len 3 or 4: (date, text, url[, metadata])
    for r in records:
        assert isinstance(r, tuple)
        assert len(r) in (3, 4)
```

- [ ] **Step 5: Run to verify failure**

```bash
pytest tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'puremacro.narrative.sources.bi'`.

- [ ] **Step 6: Write `bi.py`**

Create `puremacro/puremacro/narrative/sources/bi.py`. The exact body depends on the fixture format from Step 2:

**If the fixture is RSS/XML** (use `iter_rss_filtered` — 8-line pattern):

```python
"""Bank Indonesia (BI) — decisions + speeches.

Live: https://www.bi.go.id/en
Decision listing: <verified URL>
Speech archive: <verified URL or N/A>

Verified 2026-05-26 by the F1 Slice A rollout (commit <this commit SHA>).
"""
from __future__ import annotations

import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._rss_filtered import iter_rss_filtered
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)  # verified live-accessible

_DECISION_URL = "<verified URL from Step 1>"
_DECISION_LANDMARKS = ["<landmark1>", "<landmark2>"]   # verified-from-fixture

# If speeches archive exists, also:
_SPEECHES_URL = "<verified URL>"
_SPEECHES_LANDMARKS = ["<landmark1>", "<landmark2>"]


def iter_bi_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, source_url, metadata) tuples for BI MPC decisions."""
    try:
        listing = fetch_with_fallback(
            _DECISION_URL, policy=FALLBACK_POLICY, source="bi",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bi.iter_bi_decision: listing fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bi", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bi", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bi.iter_bi_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    # RSS path: hand off to iter_rss_filtered. The body is already
    # in `listing`; if iter_rss_filtered needs a URL not bytes, refactor
    # to use ._rss.iter_rss(text=listing, ...) instead.
    yield from iter_rss_filtered(
        _DECISION_URL, bank_code="BI", country="IDN",
        doctype="decision", language="en",
        fetch_body=fetch_body,
    )


def iter_bi_speeches(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, source_url, metadata) tuples for BI speeches.

    Drop this function entirely if BI has no clean English speeches archive."""
    # Same structure as iter_bi_decision, with _SPEECHES_URL / _SPEECHES_LANDMARKS.
    try:
        listing = fetch_with_fallback(
            _SPEECHES_URL, policy=FALLBACK_POLICY, source="bi",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bi.iter_bi_speeches: listing fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bi", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_SPEECHES_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bi", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bi.iter_bi_speeches: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from iter_rss_filtered(
        _SPEECHES_URL, bank_code="BI", country="IDN",
        doctype="speech", language="en",
        fetch_body=fetch_body,
    )


__all__ = ["iter_bi_decision", "iter_bi_speeches"]
# If iter_bi_speeches dropped, __all__ = ["iter_bi_decision"]
```

**If the fixture is HTML** (custom listing parser — ~50 lines):

Use a small `re.findall` over a stable per-CB pattern to extract `(date, title, link)` triples. Same overall structure as the RSS version above; the only difference is the body of `iter_<cb>_decision` after the landmark assertion. Pattern example for HTML listings:

```python
import re
import pandas as pd
from ._ratedoc import strip_html

# Per-CB regex against the listing HTML. Adjust to actual page structure.
_DECISION_ENTRY_RX = re.compile(
    r'<a\s+href="(?P<href>[^"]+)"[^>]*>(?P<title>[^<]+)</a>'
    r'.{0,400}?(?P<date>\d{1,2}\s+\w+\s+\d{4})',
    re.DOTALL,
)


def iter_bi_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    # ... (header + try blocks as above) ...

    for m in _DECISION_ENTRY_RX.finditer(listing):
        try:
            date = pd.to_datetime(m.group("date"))
        except (ValueError, TypeError):
            continue
        href = m.group("href")
        title = m.group("title").strip()
        url = href if href.startswith("http") else f"https://www.bi.go.id{href}"
        text = title
        if fetch_body and url:
            try:
                body_html = fetch_with_fallback(
                    url, policy=FALLBACK_POLICY, source="bi",
                )
                body_text = strip_html(body_html)
                if body_text:
                    text = body_text
            except FallbackExhaustedError:
                pass
        yield (date, text, url, {"bank_code": "BI", "country": "IDN",
                                  "doctype": "decision", "language": "en"})
```

Choose whichever path matches the verified fixture format. Replace `<verified ...>` placeholders with real values from Step 1.

- [ ] **Step 7: Run tests to verify pass**

```bash
pytest tests/test_narrative_f1_slice_a/ -v -k "bi or test_decision_fixture_yields"
```
Expected: the `bi` parametrized tests pass + `test_bi_decision_fixture_yields_at_least_one` passes.

- [ ] **Step 8: Commit**

```bash
git add puremacro/puremacro/narrative/sources/bi.py puremacro/puremacro/narrative/sources/_fixtures/bi_decision_v1.* tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py
# Also add bi_speeches_v1.* if it was created.
git commit -m "$(cat <<'EOF'
feat(0.68.0): Bank Indonesia connector (bi.py)

F1 Slice A first connector. iter_bi_decision (and iter_bi_speeches
if available) for Bank Indonesia. Adopts the Slice A
PARSER_SCHEMA_VERSION + assert_landmarks contract, the Slice B
FALLBACK_POLICY + fetch_with_fallback contract, and emits
parser_schema_mismatch telemetry events on landmark failure.

FALLBACK_POLICY: <verified value>. Verified against the live BI site
on 2026-05-26; golden fixture <bi_decision_v1.{html|xml}> captured
from <verified URL>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Adjust the commit message to reflect the actual verified policy and whether the speeches function shipped.)

---

## Task 3: `bnm.py` — Bank Negara Malaysia

**Files:**
- Create: `puremacro/puremacro/narrative/sources/bnm.py`.
- Create: `puremacro/puremacro/narrative/sources/_fixtures/bnm_decision_v1.{html|xml}`.
- (Optional) Create `bnm_speeches_v1.{html|xml}`.

Follow the verified-per-site protocol from Task 2 exactly, substituting `bi` → `bnm` everywhere and using the BNM hypothesised URLs.

### Hypothesised URLs (VERIFY)
- Decision listing: `https://www.bnm.gov.my/monetary-policy-decisions`
- Speech archive: `https://www.bnm.gov.my/speeches`

### Hypothesised `FALLBACK_POLICY`
`("live",)` — BNM has a public site.

### Steps

- [ ] **Step 1: Verify live access** via WebFetch on the hypothesised decision URL. Three outcomes (A/B/C) as documented in Task 2 Step 1. Determine `FALLBACK_POLICY`.

- [ ] **Step 2: Save the decision fixture** as `bnm_decision_v1.{html|xml}` per Task 2 Step 2.

- [ ] **Step 3: Verify + save the speeches fixture** per Task 2 Step 3. Drop `iter_bnm_speeches` if no clean English archive.

- [ ] **Step 4: Append the parametrized fixture test entry**

Extend `tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py` with:

```python
def test_bnm_decision_fixture_yields_at_least_one(monkeypatch):
    from puremacro.narrative.sources import bnm, _fallback
    text = _decision_fixture_text("bnm")
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: text,
    )
    records = list(bnm.iter_bnm_decision())
    assert len(records) >= 1
    for r in records:
        assert isinstance(r, tuple)
        assert len(r) in (3, 4)
```

- [ ] **Step 5: Write `bnm.py`** using the canonical structure from Task 2 Step 6 with `bi` → `bnm` substitutions (module name, source string, URLs, landmarks). Pick the RSS or HTML branch per the verified fixture format.

- [ ] **Step 6: Run tests to verify pass**

```bash
pytest tests/test_narrative_f1_slice_a/ -v -k "bnm"
```
Expected: `bnm` parametrized tests pass + `test_bnm_decision_fixture_yields_at_least_one` passes.

- [ ] **Step 7: Commit**

```bash
git add puremacro/puremacro/narrative/sources/bnm.py puremacro/puremacro/narrative/sources/_fixtures/bnm_decision_v1.* tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py
git commit -m "feat(0.68.0): Bank Negara Malaysia connector (bnm.py)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(Expand the message with the verified-policy + fixture-source detail as in Task 2.)

---

## Task 4: `bsp.py` — Bangko Sentral ng Pilipinas

**Hypothesised URLs (VERIFY):**
- Decision: `https://www.bsp.gov.ph/SitePages/MediaAndResearch/MonetaryPolicyDecisions.aspx`
- Speech: `https://www.bsp.gov.ph/SitePages/MediaAndResearch/Speeches.aspx`

**Hypothesised policy:** `("live",)`.

**Steps:** identical to Task 3 with `bnm` → `bsp` substitutions. Verify URLs, save fixtures, write module, add test entry, commit.

---

## Task 5: `cbn.py` — Central Bank of Nigeria

**Hypothesised URLs (VERIFY):**
- Decision: `https://www.cbn.gov.ng/MonetaryPolicy/decisions.asp`
- Speech: `https://www.cbn.gov.ng/Out/Speeches/Default.asp` (or similar under "Public Engagement")

**Hypothesised policy:** `("live", "wayback")` — site historically slow / sometimes unreachable. If live works at verification time, `("live",)` is fine; if it times out, `("live", "wayback")` is the safer choice (verify Wayback has recent snapshots before committing).

**Steps:** identical to Task 3 with `bnm` → `cbn` substitutions. Pay attention to the policy choice — if you can't reach the live site, document the Wayback fallback in the commit message.

---

## Task 6: `cbe.py` — Central Bank of Egypt

**Hypothesised URLs (VERIFY):**
- Decision: `https://www.cbe.org.eg/en/monetary-policy/monetary-policy-decisions`
- Speech: `https://www.cbe.org.eg/en/media-center/speeches`

**Hypothesised policy:** `("live",)` upgrade to `("live", "playwright")` if Cloudflare-WAF returns a challenge. Verify carefully — if the live fetch returns a Cloudflare interstitial HTML page (recognizable by the `cf-challenge` script tag or the phrase "Just a moment..."), Wayback may not have snapshots either, in which case `("playwright",)` is needed AND the fixture must be synthesized (Step 1 outcome C from Task 2).

**Steps:** identical to Task 3 with `bnm` → `cbe` substitutions. If a Cloudflare WAF is found, that's the most-complex case in this slice — document carefully in the commit message and in the module docstring.

---

## Task 7: `cbk.py` — Central Bank of Kenya

**Hypothesised URLs (VERIFY):**
- Decision: `https://www.centralbank.go.ke/monetary-policy-statements/`
- Speech: `https://www.centralbank.go.ke/speeches/`

**Hypothesised policy:** `("live",)`.

**Steps:** identical to Task 3 with `bnm` → `cbk` substitutions.

---

## Sub-slice 4 — Cross-connector polish

(Tasks 8-11.)

## Task 8: Parametrize `test_landmark_fixtures.py`

**Files:**
- Create: `tests/test_narrative_f1_slice_a/test_landmark_fixtures.py`.

This test mirrors Slice A's `test_landmark_fixtures.py` — each connector's fixture, when fed to that connector's `assert_landmarks` call, passes without raising.

- [ ] **Step 1: Write the test**

Create `tests/test_narrative_f1_slice_a/test_landmark_fixtures.py`:

```python
"""F1 Slice A — each connector's golden fixture passes its
landmark check. Regression guard for upstream layout drift."""
from __future__ import annotations

import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbn", "cbe", "cbk"]

_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _fixture_text(cb: str, kind: str) -> str:
    """Return the bytes of <cb>_<kind>_v1.html or .xml, whichever exists."""
    for ext in ("html", "xml"):
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
```

- [ ] **Step 2: Run to verify pass**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_narrative_f1_slice_a/test_landmark_fixtures.py -v
```
Expected: 6 decision tests pass; 6 speeches tests pass-or-skip (depending on whether each connector ships speeches).

- [ ] **Step 3: Commit**

```bash
git add tests/test_narrative_f1_slice_a/test_landmark_fixtures.py
git commit -m "$(cat <<'EOF'
test(0.68.0): F1 Slice A — parametrized landmark fixture check

Verifies each of the 6 new connectors' golden fixture passes its
own landmark check. Acts as a regression guard for upstream layout
drift. Speeches test is parametrized + skipped per-connector if the
function or fixture is absent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Parametrize `test_decision_fixture_yields.py` across all 6

By the end of Tasks 2-7, `test_decision_fixture_yields.py` has 6 separate per-connector test functions (one added per task). This task collapses them into a single parametrized test.

**Files:**
- Modify: `tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py`.

- [ ] **Step 1: Rewrite as parametrized**

Replace the contents of `tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py` with:

```python
"""F1 Slice A — each connector's iter_<cb>_decision yields ≥1 tuple
when run against its golden fixture."""
from __future__ import annotations

import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbn", "cbe", "cbk"]

_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _decision_fixture_text(cb: str) -> str:
    for ext in ("html", "xml"):
        p = _FIXTURE_DIR / f"{cb}_decision_v1.{ext}"
        if p.exists():
            return p.read_text()
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
```

- [ ] **Step 2: Run to verify pass**

```bash
pytest tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py -v
```
Expected: 6 passed (one per connector).

- [ ] **Step 3: Commit**

```bash
git add tests/test_narrative_f1_slice_a/test_decision_fixture_yields.py
git commit -m "$(cat <<'EOF'
test(0.68.0): F1 Slice A — parametrize decision fixture yields across all 6

Consolidates the 6 per-connector smoke tests (added one-per-task in
T2-T7) into a single parametrized test. Same coverage; less
boilerplate; one test name per connector in pytest output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `test_per_connector_smoke.py` — import + iter callable

**Files:**
- Create: `tests/test_narrative_f1_slice_a/test_per_connector_smoke.py`.

- [ ] **Step 1: Write the test**

Create `tests/test_narrative_f1_slice_a/test_per_connector_smoke.py`:

```python
"""F1 Slice A — each module imports cleanly and exposes iter_<cb>_decision."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbn", "cbe", "cbk"]


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_module_imports_cleanly(cb):
    importlib.import_module(f"puremacro.narrative.sources.{cb}")


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_iter_decision_function_exists(cb):
    mod = importlib.import_module(f"puremacro.narrative.sources.{cb}")
    iter_name = f"iter_{cb}_decision"
    assert hasattr(mod, iter_name), (
        f"{cb}.py must export {iter_name} (F1 Slice A contract)"
    )
    assert callable(getattr(mod, iter_name))
```

- [ ] **Step 2: Run to verify pass**

```bash
pytest tests/test_narrative_f1_slice_a/test_per_connector_smoke.py -v
```
Expected: 12 passed (6 connectors × 2 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_narrative_f1_slice_a/test_per_connector_smoke.py
git commit -m "$(cat <<'EOF'
test(0.68.0): F1 Slice A — per-connector smoke (import + iter callable)

12 parametrized tests (6 connectors × 2). Catches module-level import
errors and missing iter_<cb>_decision exports. iter_<cb>_speeches is
NOT required (best-effort per the spec).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: AST coverage assertion

**Files:**
- Create: `tests/test_narrative_f1_slice_a/test_coverage_assertion.py`.

- [ ] **Step 1: Write the test**

Create `tests/test_narrative_f1_slice_a/test_coverage_assertion.py`:

```python
"""F1 Slice A — coverage assertion: the 6 named connectors all
declare PARSER_SCHEMA_VERSION + call assert_landmarks + call
fetch_with_fallback. Fails the build if any of them regresses."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ("bi", "bnm", "bsp", "cbn", "cbe", "cbk")


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text()


def _has_call(name: str, fn_name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == fn_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == fn_name:
                return True
    return False


def _has_parser_schema_version(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "PARSER_SCHEMA_VERSION") and isinstance(
        mod.PARSER_SCHEMA_VERSION, int
    )


def _has_fallback_policy(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "FALLBACK_POLICY") and isinstance(
        mod.FALLBACK_POLICY, tuple
    )


def test_every_f1a_connector_has_parser_schema_version():
    missing = [n for n in _F1A_CONNECTORS if not _has_parser_schema_version(n)]
    assert not missing, (
        f"F1 Slice A contract violation: connectors missing "
        f"PARSER_SCHEMA_VERSION: {missing}"
    )


def test_every_f1a_connector_has_fallback_policy():
    missing = [n for n in _F1A_CONNECTORS if not _has_fallback_policy(n)]
    assert not missing, (
        f"F1 Slice A contract violation: connectors missing "
        f"FALLBACK_POLICY: {missing}"
    )


def test_every_f1a_connector_calls_assert_landmarks():
    missing = [n for n in _F1A_CONNECTORS if not _has_call(n, "assert_landmarks")]
    assert not missing, (
        f"F1 Slice A contract violation: connectors not calling "
        f"assert_landmarks: {missing}"
    )


def test_every_f1a_connector_calls_fetch_with_fallback():
    missing = [n for n in _F1A_CONNECTORS if not _has_call(n, "fetch_with_fallback")]
    assert not missing, (
        f"F1 Slice A contract violation: connectors not calling "
        f"fetch_with_fallback: {missing}"
    )
```

- [ ] **Step 2: Run to verify pass**

```bash
pytest tests/test_narrative_f1_slice_a/test_coverage_assertion.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Run the full F1 Slice A suite**

```bash
pytest tests/test_narrative_f1_slice_a/ -v 2>&1 | tail -10
```
Expected: all pass — 12 framework (parser_schema + fallback_policy) + 12 landmark fixture (6 decision + 6 speeches-or-skip) + 6 decision yields + 12 smoke + 4 coverage = ~46 tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_narrative_f1_slice_a/test_coverage_assertion.py
git commit -m "$(cat <<'EOF'
test(0.68.0): F1 Slice A — coverage assertion (4 contract checks)

AST-scans each of the 6 connectors and asserts:
(1) PARSER_SCHEMA_VERSION declared as int.
(2) FALLBACK_POLICY declared as tuple.
(3) Module body contains a call to assert_landmarks(...).
(4) Module body contains a call to fetch_with_fallback(...).

Fails the build if any future refactor removes any of these.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Sub-slice 5 — Release

(Tasks 12-13.)

## Task 12: R5_03 notebook + paired builder

**Files:**
- Create: `tools/make_notebook_R5_03.py`.
- Create: `notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb` (built + executed by the builder).

Per the memory rule: notebooks + builders ship together; nbconvert run from controller / foreground (not subagent).

- [ ] **Step 1: Create the builder**

Create `tools/make_notebook_R5_03.py`:

```python
"""Build R5_03_f1_sea_africa_demo.ipynb — F1 Slice A demo.

Demonstrates the 6 new SE Asia + Africa CB connectors plus their
telemetry signature in connector_health().

Run:
    python tools/make_notebook_R5_03.py
Then execute (foreground, controller-side):
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat


_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "notebooks" / "R5_data_infra" / "R5_03_f1_sea_africa_demo.ipynb"


def _md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src)


def build() -> None:
    nb = nbformat.v4.new_notebook()
    cells = []

    cells.append(_md("""# R5_03 — F1 Slice A: SE Asia + Africa CB connectors (0.68.0)

Demonstrates the 6 new central-bank narrative connectors added in
F1 Slice A:

- **SE Asia:** `bi` (Bank Indonesia), `bnm` (Bank Negara Malaysia),
  `bsp` (Bangko Sentral ng Pilipinas).
- **Africa:** `cbn` (Central Bank of Nigeria), `cbe` (Central Bank
  of Egypt), `cbk` (Central Bank of Kenya).

Each adopts the Slice A schema-versioning contract (PARSER_SCHEMA_VERSION
+ assert_landmarks) and the Slice B fallback + telemetry contracts
(FALLBACK_POLICY + fetch_with_fallback + parser_schema_mismatch events).

Demo is fully offline: connector iter functions are exercised against
their golden fixtures; connector_health() is shown on synthetic seeded
events for the 6 sources.

Spec: `docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md`
"""))

    cells.append(_code("""\
from __future__ import annotations
import os, tempfile
from pathlib import Path
import pandas as pd
import puremacro
print('puremacro', puremacro.__version__)

# Use a tmp cache DB so this demo is fully self-contained.
tmpdir = Path(tempfile.mkdtemp())
os.environ['PUREMACRO_HTTP_CACHE_DIR'] = str(tmpdir)
os.environ.pop('PUREMACRO_NARRATIVE_TELEMETRY', None)
import puremacro._cache_db as _db
_db.close_conn()
"""))

    cells.append(_md("## 1. Per-connector smoke (against golden fixtures)"))

    cells.append(_code("""\
from puremacro.narrative.sources import bi, bnm, bsp, cbn, cbe, cbk, _fallback
import pathlib

_FIXTURE_DIR = (
    pathlib.Path(puremacro.__file__).resolve().parent
    / 'narrative' / 'sources' / '_fixtures'
)

def _decision_fixture_text(cb: str) -> str:
    for ext in ('html', 'xml'):
        p = _FIXTURE_DIR / f'{cb}_decision_v1.{ext}'
        if p.exists():
            return p.read_text()
    raise FileNotFoundError(cb)


def _demo(cb_module, name):
    text = _decision_fixture_text(name)
    original = _fallback._stage_live
    _fallback._stage_live = lambda url, *, timeout, use_cache: text
    try:
        iter_fn = getattr(cb_module, f'iter_{name}_decision')
        records = list(iter_fn())
        print(f'{name:5s}: {len(records)} record(s) from fixture')
        if records:
            first = records[0]
            print(f'         first: date={first[0]}, '
                  f'url={first[2][:60]}...' if len(first[2]) > 60 else first[2])
    finally:
        _fallback._stage_live = original


for mod, name in [(bi,'bi'), (bnm,'bnm'), (bsp,'bsp'),
                   (cbn,'cbn'), (cbe,'cbe'), (cbk,'cbk')]:
    _demo(mod, name)
"""))

    cells.append(_md("## 2. Seed synthetic events + inspect connector_health"))

    cells.append(_code("""\
from puremacro.narrative.sources._telemetry import log_event, connector_health

# Realistic seed: most fetches succeed; a couple of timeouts.
events = [
    *[('bi',  'success', 'live')] * 8,
    *[('bnm', 'success', 'live')] * 6,
    *[('bsp', 'success', 'live')] * 5,
    *[('cbn', 'success', 'live')] * 4,
    *[('cbn', 'timeout', 'live')] * 2,
    *[('cbn', 'success', 'wayback')] * 2,
    *[('cbe', 'success', 'live')] * 3,
    *[('cbk', 'success', 'live')] * 4,
]
for source, outcome, fb in events:
    log_event(source=source, outcome=outcome, fallback_used=fb)

connector_health(window=pd.Timedelta(days=1)).set_index('source')
"""))

    cells.append(_md("""## What's next

- **F1 Slice B** — business surveys (IFO Germany, Tankan via BoJ,
  ZEW, Conference Board, Michigan Consumer Sentiment).
- **F1 Slice C** — forecaster + uncertainty surveys (BoE DMP, ECB SPF,
  SNB Survey, Atlanta Fed BIE/BU).
- **F1 Slice D** — alt-data (Google Trends, earnings call transcripts).

Reference: `docs/CONNECTOR_HEALTH.md`, the F1 Slice A spec at
`docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md`.
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        nbformat.write(nb, f)
    print(f"wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 2: Build the notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python tools/make_notebook_R5_03.py
```
Expected: `wrote notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb`.

- [ ] **Step 3: Execute foreground**

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb
```
Expected: notebook executes cleanly. Verify cell outputs populated (especially the 6-connector smoke + the `connector_health()` DataFrame).

- [ ] **Step 4: Commit builder + executed notebook**

```bash
git add tools/make_notebook_R5_03.py notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb
git commit -m "$(cat <<'EOF'
feat(0.68.0): R5_03 F1 Slice A demo notebook + paired builder

Slice's visible deliverable. R5_03 demonstrates the 6 new SE Asia +
Africa CB connectors end-to-end against golden fixtures (no live
network), plus a synthetic-seeded connector_health() query showing
the 6 new sources appearing in the dashboard alongside Slice A + B
participants. Fully offline-runnable. Shipped with the paired builder
per the notebook ↔ builder pairing rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Version bump + CHANGELOG + ARCHITECTURE + final sanity sweep

**Files:**
- Modify: `puremacro/pyproject.toml` (`version = "0.68.0"`)
- Modify: `puremacro/puremacro/__init__.py` (`__version__ = "0.68.0"`)
- Modify: `puremacro/CHANGELOG.md` (prepend 0.68.0 section)
- Modify: `puremacro/ARCHITECTURE.md` (append "F1 Slice A — SE Asia + Africa CBs (0.68.0+)" subsection)
- Modify: `tests/test_credentials/test_service_registry.py` (rename + update version test)
- Modify: `tests/test_signal_contract/test_schema_extension.py` (same — Slice B's implementer caught this parallel pin; same accommodation here)

- [ ] **Step 1: Update both version smoke tests**

In `tests/test_credentials/test_service_registry.py`, find `test_puremacro_version_is_067` and rename + update:

```python
def test_puremacro_version_is_068():
    import puremacro
    assert puremacro.__version__ == "0.68.0"
```

In `tests/test_signal_contract/test_schema_extension.py`, find the parallel `test_puremacro_version_is_067` (added by Slice B's T11 implementer) and apply the same rename + update.

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_068 -v
```
Expected: FAIL — version still `"0.67.0"`.

- [ ] **Step 3: Bump `puremacro/__init__.py`**

Find `__version__ = "0.67.0"` and change to `__version__ = "0.68.0"`.

- [ ] **Step 4: Bump `pyproject.toml`**

Find `version = "0.67.0"` and change to `version = "0.68.0"`. `requires-python = ">=3.12"` stays.

- [ ] **Step 5: Prepend CHANGELOG entry**

Insert this block in `puremacro/CHANGELOG.md` IMMEDIATELY AFTER the `# Changelog` header + intro paragraph, BEFORE the `## 0.67.0 (2026-05-26)` section:

```markdown
## 0.68.0 (2026-05-26)

**F1 Slice A — SE Asia + Africa CB connectors.**

### Added
- Six new central-bank narrative connectors:
  - `puremacro.narrative.sources.bi` — Bank Indonesia (IDN).
  - `puremacro.narrative.sources.bnm` — Bank Negara Malaysia (MYS).
  - `puremacro.narrative.sources.bsp` — Bangko Sentral ng Pilipinas (PHL).
  - `puremacro.narrative.sources.cbn` — Central Bank of Nigeria (NGA).
  - `puremacro.narrative.sources.cbe` — Central Bank of Egypt (EGY).
  - `puremacro.narrative.sources.cbk` — Central Bank of Kenya (KEN).
- Each connector exports `iter_<cb>_decision()` (mandatory) and,
  where the bank has a clean separate English speeches archive,
  `iter_<cb>_speeches()`.
- Each adopts the Slice A + B contracts from inception:
  `PARSER_SCHEMA_VERSION = 1` + `assert_landmarks(...)`,
  `FALLBACK_POLICY: tuple[str, ...]` + `fetch_with_fallback(...)`,
  and emits `parser_schema_mismatch` events on schema mismatch.
- Per-CB `FALLBACK_POLICY` values verified at implementation time
  against each site's actual behaviour (see per-commit messages).
- Twelve golden HTML/XML fixtures under
  `narrative/sources/_fixtures/<cb>_{decision|speeches}_v1.*`
  (fewer if some speeches functions were dropped per scope decision).
- `notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb` + paired
  builder `tools/make_notebook_R5_03.py`.

### Changed
- `connector_health()` now surfaces up to 6 new source rows once the
  new connectors are called (in addition to the 7 fallback connectors
  and 8 schema-checked connectors from Slices A + B).

### Roadmap
- **F1 Slice B** queued: business surveys (IFO, Tankan, ZEW,
  Conference Board, Michigan Consumer Sentiment).
- **F1 Slice C** queued: forecaster + uncertainty surveys (BoE DMP,
  ECB SPF, SNB Survey, Atlanta Fed BIE/BU).
- **F1 Slice D** queued: alt-data (Google Trends, earnings calls).
- Sibling sub-projects still queued: F3 unified panel-builder, S2
  interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2
  onboarding.
- Full spec: `docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md`.

### Internal
- New test directory: `tests/test_narrative_f1_slice_a/` (~46 tests
  across 6 test files).
- Per-CB `FALLBACK_POLICY` is single-stage where the site is
  live-accessible (`("live",)`) and multi-stage where it isn't
  (e.g., CBN's `("live", "wayback")` for slow / unreliable upstream).
  See each connector's module docstring for the verified rationale.

```

- [ ] **Step 6: Append the ARCHITECTURE.md subsection**

In `puremacro/ARCHITECTURE.md`, find the existing "F2 closure (0.67.0+)" subsection (shipped in Slice B) and APPEND immediately after it:

```markdown
### F1 Slice A — SE Asia + Africa CBs (0.68.0+)

First slice of F1 source-coverage expansion. Six new central-bank
connectors: `bi` (Indonesia), `bnm` (Malaysia), `bsp` (Philippines),
`cbn` (Nigeria), `cbe` (Egypt), `cbk` (Kenya). Each adopts the Slice
A schema-versioning contract + the Slice B fallback + telemetry
contracts from inception. Per-CB `FALLBACK_POLICY` values verified
against each site's actual behaviour at implementation time. Full
spec: `docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md`.
Subsequent F1 slices queued: B (business surveys), C (forecaster
surveys), D (alt-data).
```

- [ ] **Step 7: Run the version tests to verify pass**

```bash
pytest tests/test_credentials/test_service_registry.py::test_puremacro_version_is_068 tests/test_signal_contract/test_schema_extension.py::test_puremacro_version_is_068 -v
```
Expected: 2 passed.

- [ ] **Step 8: Final full-suite sanity sweep**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && \
pytest tests/test_narrative_f1_slice_a/ -v 2>&1 | tail -10 && \
pytest tests/test_narrative_fallback/ tests/test_narrative_telemetry/ -v 2>&1 | tail -10 && \
pytest tests/test_cache_db/ tests/test_credentials/ tests/test_vintages_alfred_store/ tests/test_narrative_schema_checks/ -v 2>&1 | tail -10 && \
pytest tests/test_pyodide_compat.py -v && \
pytest tests/test_signal_contract/ -v 2>&1 | tail -5
```
Expected:
- All F1 Slice A tests pass (~46).
- All F2 Slice B test directories still green.
- All F2 Slice A test directories still green.
- Pyodide compat passes.
- Signal-contract (Slice 1) still green.

- [ ] **Step 9: Commit**

```bash
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/CHANGELOG.md puremacro/ARCHITECTURE.md tests/test_credentials/test_service_registry.py tests/test_signal_contract/test_schema_extension.py
git commit -m "$(cat <<'EOF'
chore(puremacro): bump to 0.68.0 — F1 Slice A (SE Asia + Africa CBs)

Ships 6 new CB connectors (bi, bnm, bsp, cbn, cbe, cbk) each adopting
the Slice A + B contracts. iter_<cb>_decision shipped per-CB;
iter_<cb>_speeches shipped where the CB has a clean separate English
archive (best-effort, documented per-CB). Per-CB FALLBACK_POLICY
verified against actual site behaviour.

Subsequent F1 slices queued: B (business surveys), C (forecaster
surveys), D (alt-data).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done-definition for F1 Slice A (0.68.0)

- [ ] 6 new connector modules ship; each exports `iter_<cb>_decision`. `iter_<cb>_speeches` ships per-CB where the site has a clean separate English archive.
- [ ] Each connector declares `PARSER_SCHEMA_VERSION = 1` + `FALLBACK_POLICY` (verified) + uses `fetch_with_fallback` + emits `parser_schema_mismatch` events.
- [ ] Per-`iter` golden fixtures shipped (one per function).
- [ ] AST coverage assertion (4 checks) enforces the contract on all 6.
- [ ] R5_03 notebook executes cleanly with `connector_health()` showing the 6 new sources.
- [ ] `docs/specs/2026-05-26-puremacro-068-f1-slice-a-sea-africa-cb-design.md` committed; ARCHITECTURE.md gains "F1 Slice A" subsection.
- [ ] `pyproject.toml` at `version = "0.68.0"`; CHANGELOG 0.68.0 entry.
- [ ] Pyodide compat passes; full narrative test suite shows zero new regressions vs. the post-Slice-B baseline.

## Out of scope for F1 Slice A (queued for follow-up)

- F1 Slice B: business surveys (IFO Germany, Tankan, ZEW, Conference Board, Michigan).
- F1 Slice C: forecaster + uncertainty surveys (BoE DMP, ECB SPF, Atlanta Fed BIE/BU, SNB Survey, ECB CIS).
- F1 Slice D: alt-data (Google Trends, earnings calls, container shipping, satellite night-lights).
- F1 Slice E+: local-language coverage (Bahasa, Arabic, Swahili), additional CB coverage (Eastern Europe NBP/CNB/MNB/BNR/CBRT; smaller advanced SNB/DNB/CBI), minutes for the 6 CBs where they're published in English.
- F2 Slice C+: per-event `url_hash` + `latency_ms`, retention controls, new fallback stages, OpenTelemetry exporter.
- Sibling sub-projects: F3 unified panel-builder, S2 interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding.
