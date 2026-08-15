# puremacro 0.68.0 — F1 Slice A: SE Asia + Africa central-bank connectors

**Status:** Drafted 2026-05-26. First slice of the F1 source-coverage-expansion sub-project from the post-Slice-1 sibling roadmap. Implementation in five sub-slices within one 0.68.0 release window.
**Target releases:** 0.68.0 (Slice A — 6 CB connectors). F1 Slices B (business surveys), C (forecaster surveys), and D (alt-data) queued for subsequent releases.
**Driving lenses:** the most acute regional coverage gap in puremacro's narrative connector set — 9 of 10 listed African and SE Asian central banks have no current connector; adoption of the Slice A schema-versioning and Slice B fallback + telemetry contracts so the new connectors are first-class from day one; zero new runtime dependencies.

## Motivation

After F2 closed (0.67.0), puremacro ships ~60 narrative connectors covering Fed / ECB / BoE / BoJ / RBA / RBI / RBNZ / BoK / Riksbank / Norges / SARB / PBoC / Banxico / BCRA / BCB / BCCL / BoT / MAS / BanRep central banks, plus US, EU, and ministry sources. Two acute regional gaps remain:

1. **SE Asia is thin.** Of the major Asian central banks, only BoJ, RBI, RBNZ, BoK, and MAS are covered. Indonesia (BI), Malaysia (BNM), Philippines (BSP) — three of the four largest ASEAN economies — have no connector.
2. **Africa is thin.** SARB is the only African central bank in the set. Nigeria (CBN), Egypt (CBE), Kenya (CBK) are missing — the three other dominant African economies by GDP. A researcher building a cross-Africa or cross-emerging-market narrative-IV instrument set has to scrape each by hand.

F1 Slice A closes these two gaps by adding six new connectors (BI, BNM, BSP, CBN, CBE, CBK), each adopting the Slice A schema-versioning + Slice B fallback + telemetry contracts from inception. Subsequent F1 slices will add business surveys (IFO / Tankan / ZEW / Conference Board / Michigan), forecaster surveys (BoE DMP, ECB SPF, etc.), and alt-data (Google Trends, earnings calls).

## Non-goals

- **No** rollout to F1 Slice B / C / D scope (business surveys, forecaster surveys, alt-data) — each is its own brainstorm + spec.
- **No** minutes (committee proceedings) for the 6 CBs — most don't publish formal minutes in English. If and when they do, `iter_<cb>_minutes()` is a one-line addition in a follow-up.
- **No** local-language coverage (Bahasa, Arabic, Swahili). Slice A is English-only; multilingual is a Slice E+ possibility.
- **No** retroactive PARSER_SCHEMA_VERSION rollout to the ~45 other connectors that don't yet have it. The 6 new connectors adopt the contract; existing ones outside the original Slice A 8-list keep their current shape.
- **No** new runtime dependencies. Same constraint as F2.

## Architecture

### Module map (deltas only)

```
puremacro/puremacro/narrative/sources/
├── bi.py                          [NEW]   Bank Indonesia (IDN)
├── bnm.py                         [NEW]   Bank Negara Malaysia (MYS)
├── bsp.py                         [NEW]   Bangko Sentral ng Pilipinas (PHL)
├── cbn.py                         [NEW]   Central Bank of Nigeria (NGA)
├── cbe.py                         [NEW]   Central Bank of Egypt (EGY)
├── cbk.py                         [NEW]   Central Bank of Kenya (KEN)
└── _fixtures/
    ├── bi_decision_v1.{html|xml}             [NEW]
    ├── bi_speeches_v1.{html|xml}             [NEW, if function shipped]
    ├── bnm_decision_v1.html                  [NEW]
    ├── bnm_speeches_v1.html                  [NEW, if function shipped]
    ├── bsp_decision_v1.{html|xml}            [NEW]
    ├── bsp_speeches_v1.{html|xml}            [NEW, if function shipped]
    ├── cbn_decision_v1.html                  [NEW]
    ├── cbn_speeches_v1.html                  [NEW, if function shipped]
    ├── cbe_decision_v1.html                  [NEW]
    ├── cbe_speeches_v1.html                  [NEW, if function shipped]
    ├── cbk_decision_v1.html                  [NEW]
    └── cbk_speeches_v1.html                  [NEW, if function shipped]

notebooks/R5_data_infra/
└── R5_03_f1_sea_africa_demo.ipynb           [NEW]  + paired
                                                     tools/make_notebook_R5_03.py

tests/test_narrative_f1_slice_a/             [NEW directory]
├── __init__.py
├── test_per_connector_smoke.py
├── test_parser_schema_versions.py
├── test_fallback_policies.py
├── test_landmark_fixtures.py
├── test_decision_fixture_yields.py
└── test_coverage_assertion.py

pyproject.toml + __init__.py + CHANGELOG.md + ARCHITECTURE.md   [MODIFIED]
```

### Six target connectors

| Module | Bank | Country | Hypothesised site shape | Hypothesised policy | Notes |
|---|---|---|---|---|---|
| `bi.py` | Bank Indonesia | IDN | RSS + HTML | `("live",)` | `bi.go.id/en/news` likely has RSS for press releases. |
| `bnm.py` | Bank Negara Malaysia | MYS | HTML listing | `("live",)` | `bnm.gov.my` has dedicated press / speeches pages. |
| `bsp.py` | Bangko Sentral ng Pilipinas | PHL | RSS + HTML | `("live",)` | `bsp.gov.ph/news/speeches` well-structured. |
| `cbn.py` | Central Bank of Nigeria | NGA | HTML listing | `("live", "wayback")` likely | Site historically slow / unreliable. |
| `cbe.py` | Central Bank of Egypt | EGY | HTML listing | `("live",)` → `("live", "playwright")` if WAF | Cloudflare possible; verify at implementation time. |
| `cbk.py` | Central Bank of Kenya | KEN | HTML listing | `("live",)` | `centralbank.go.ke` has "Speeches" archive. |

The implementer verifies each site's actual shape before locking `FALLBACK_POLICY`, landmark strings, and the speeches-archive availability (same verified-not-hypothesised pattern Slice B used for `eu_eurlex` / `eu_parliament`).

### Canonical per-connector structure

```python
"""<Bank name> (<CODE>) — decisions + speeches.

Live: <root URL>
Decision listing: <decision URL>
Speech archive: <speech URL>   # omit + drop iter_<cb>_speeches if N/A
"""
from __future__ import annotations

import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._ratedoc import strip_html              # or relevant helper
from ._telemetry import log_event             # local imports also acceptable


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)  # verify + adjust per site

_DECISION_URL = "https://..."
_SPEECHES_URL = "https://..."

_DECISION_LANDMARKS = ["...", "..."]    # 2-3 substrings reliably present
_SPEECHES_LANDMARKS = ["...", "..."]


def iter_<cb>_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, source_url, metadata) tuples for MPC decisions."""
    try:
        listing = fetch_with_fallback(
            _DECISION_URL, policy=FALLBACK_POLICY, source="<cb>",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"<cb>.iter_<cb>_decision: listing fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="<cb>", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="<cb>", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"<cb>.iter_<cb>_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    # ... per-CB listing parse + per-entry yield ...
```

The `iter_<cb>_speeches` function uses the same structure with `_SPEECHES_URL` / `_SPEECHES_LANDMARKS`. If a CB doesn't have a clean separate speeches archive in English, the function is OMITTED entirely — best-effort scope, not contractual.

For RSS-based sites: use the existing `iter_rss_filtered(_DECISION_URL, bank_code="<CB>", ...)` helper (see `bcra.py` for an 8-line example). For HTML-based sites: parse the listing with a small `re.findall` over a stable per-CB pattern (avoid bringing in BeautifulSoup — `_ratedoc.py` already eschews it).

### Per-CB scope decisions

| CB | `iter_<cb>_decision` | `iter_<cb>_speeches` |
|---|---|---|
| `bi` | ✓ | ✓ if BI's "Speeches" sub-page exists in English |
| `bnm` | ✓ | ✓ — BNM has a dedicated `/speeches` page |
| `bsp` | ✓ | ✓ — BSP's `/news/speeches` is well-structured |
| `cbn` | ✓ | ✓ — speeches under "Public Engagement" |
| `cbe` | ✓ | ✓ — Governor speeches under "Media Center" |
| `cbk` | ✓ | ✓ — Governor speeches under "Speeches" |

All 6 hypothesised to have a speeches archive. If implementation reveals otherwise for any CB, drop that connector's `iter_<cb>_speeches` and document in the module docstring + CHANGELOG.

### Slice integration

Each new connector adopts contracts from all three prior slices:

- **Slice A signal contract (0.65.0)** — N/A. The signal contract is about index outputs; raw narrative connectors don't produce indices directly.
- **Slice A parser schema versioning (0.66.0, Sub-slice 4)** — every new connector declares `PARSER_SCHEMA_VERSION = 1` and calls `assert_landmarks` at the top of its body parser. This makes the 6 new connectors first-class members of the schema-versioning population.
- **Slice B governed fallback (0.67.0, F2.4)** — every new connector declares `FALLBACK_POLICY` as a `tuple[str, ...]` of `SUPPORTED_STAGES`, and calls `fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<cb>")` instead of direct `safe_get_*`.
- **Slice B health telemetry (0.67.0, F2.5)** — `fetch_with_fallback` already emits telemetry events. Each connector's `except ParserSchemaMismatchError` block also emits `log_event(outcome="parser_schema_mismatch", fallback_used="none")` (same pattern as the 8 Slice-A schema-checked connectors did in 0.67.0).

Result: the 6 new connectors show up immediately in `connector_health()` as soon as they're called. A researcher running `connector_health(window=pd.Timedelta(days=7))` sees the 7 existing fallback connectors, the 8 schema-checked connectors, and the 6 new SE Asia + Africa connectors — 13 distinct sources (with overlap: `eu_eurlex`/`eu_parliament`/`us_cbo` appear in both prior groups).

## Data flow

```
caller (researcher_notebook or pipeline)
  ↓
from puremacro.narrative.sources import bi
records = list(bi.iter_bi_decision(fetch_body=True))
  ↓
iter_bi_decision()
  ↓
fetch_with_fallback(_DECISION_URL, policy=FALLBACK_POLICY, source="bi")
  ├── stage="live" → safe_get_text_cached(_DECISION_URL)
  │   log_event(source="bi", outcome="success", fallback_used="live")
  │   return listing HTML
  └── (only if policy includes other stages and live failed)
  ↓
assert_landmarks(listing, source="bi", expected_version=1,
                  landmarks=["Bank Indonesia", "BI 7-Day Reverse Repo Rate"])
  ├── present → continue
  └── missing → ParserSchemaMismatchError
       ↓ caught locally
       log_event(source="bi", outcome="parser_schema_mismatch",
                 fallback_used="none")
       warnings.warn(...)
       return (zero records yielded)
  ↓
parse listing → for each entry:
  if fetch_body:
      body = fetch_with_fallback(entry_url, policy=FALLBACK_POLICY, source="bi")
        (one telemetry event per stage attempt)
  yield (date, text_or_title, source_url, metadata)
```

## Failure semantics

All failure paths inherited from Slice A + B. The new connectors introduce zero new failure categories.

| Failure | Where | Behavior |
|---|---|---|
| Listing fetch fails every stage | `fetch_with_fallback` → `FallbackExhaustedError` | caught locally → `warnings.warn` + iter yields empty |
| Schema mismatch on listing | `assert_landmarks` → `ParserSchemaMismatchError` | caught locally → `log_event(parser_schema_mismatch)` + `warnings.warn` + iter yields empty |
| Per-entry body fetch fails | inside the listing loop | per-entry catch → continue to next entry; record yielded with title-only text if available |
| Per-entry parse fails (bad HTML for one entry) | inside the loop | per-entry skip; continue iteration |
| Site rate-limits (429) | `safe_get_*` | classified by `_classify`; logged via fetch_with_fallback's normal telemetry |
| CB has no speeches archive | implementer-time decision | function dropped from the module; documented in docstring + CHANGELOG |

## Pyodide contract

- All 6 new modules: pure stdlib + pandas + `puremacro.narrative.sources._http` family. Pyodide-pure (no Playwright unless `FALLBACK_POLICY` includes `"playwright"`, which lazy-imports per Slice B).
- Fixture files (HTML / XML) are static resources, not Python. Pyodide-irrelevant.
- The new test directory follows the same pytest convention as Slices A + B; no Pyodide-specific test changes.

Extended `tests/test_pyodide_compat.py` is re-run at release time; no source change unless a new forbidden import slips in.

## Testing

### Headline tests (`tests/test_narrative_f1_slice_a/`)

1. **Module imports cleanly** — parametrized over the 6 module names: `importlib.import_module(f"puremacro.narrative.sources.{name}")` succeeds without side-effects.
2. **`PARSER_SCHEMA_VERSION` present** — `hasattr(mod, "PARSER_SCHEMA_VERSION") and isinstance(..., int)`.
3. **`FALLBACK_POLICY` valid** — `tuple` of strings drawn from `SUPPORTED_STAGES`.
4. **`iter_<cb>_decision` exists + callable** — every module exports this function.
5. **`iter_<cb>_speeches` is OPTIONAL** — if the module exports it, callable; if not, no error.
6. **Decision fixture roundtrip** — parametrized: `fetch_with_fallback` patched to return the fixture; `list(iter_<cb>_decision())` yields ≥1 tuple of `(date, text, url, metadata)`.
7. **Speeches fixture roundtrip** — same, for connectors that export `iter_<cb>_speeches`.
8. **Landmark assertion fires** — parametrized: patch `fetch_with_fallback` to return a body MISSING the landmarks; expect `UserWarning` + zero records yielded.
9. **Telemetry event on mismatch** — parametrized: patch + verify `parser_schema_mismatch` event in `connector_events`.
10. **AST coverage** — each of the 6 modules imports `assert_landmarks` and `fetch_with_fallback`, declares `PARSER_SCHEMA_VERSION` and `FALLBACK_POLICY`, defines `iter_<cb>_decision`.

All tests mock `fetch_with_fallback` (or `_fallback._stage_live`) to return fixture text — no real network calls.

### Cross-cutting

- Pyodide-compat re-run at release time.
- Slice 1 signal-contract tests + all F2 test directories stay green.

## Staging

Five sub-slices inside the 0.68.0 release window.

### Sub-slice 1 — framework + first connector (~3 commits)

1. Create `tests/test_narrative_f1_slice_a/__init__.py` + `test_coverage_assertion.py` + `test_parser_schema_versions.py` + `test_fallback_policies.py` (initially failing — connectors don't exist yet).
2. First connector end-to-end: `bi.py` + fixture(s) + landmark calibration. Test files start passing for `bi`.

### Sub-slice 2 — Asian batch (~2 commits)

3. `bnm.py` + fixtures.
4. `bsp.py` + fixtures.

### Sub-slice 3 — African batch (~3 commits)

5. `cbn.py` + fixtures.
6. `cbe.py` + fixtures.
7. `cbk.py` + fixtures.

### Sub-slice 4 — cross-connector polish (~3 commits)

8. `test_landmark_fixtures.py` parametrized across all 6.
9. `test_decision_fixture_yields.py` parametrized across all 6.
10. `test_per_connector_smoke.py` parametrized — import-and-call smoke.

### Sub-slice 5 — release (~2 commits)

11. `notebooks/R5_data_infra/R5_03_f1_sea_africa_demo.ipynb` + paired builder. Shows all 6 connectors yielding from offline fixtures + a `connector_health()` query on synthetic seeded events for the 6 sources.
12. Version bump to 0.68.0 + CHANGELOG + ARCHITECTURE update + final sanity sweep.

**Total**: ~13 commits.

**Critical-path note**: each connector commit (steps 2, 3, 4, 5, 6, 7) requires the implementer to **verify the actual site shape** before writing the parser — landmark strings, FALLBACK_POLICY, presence of speeches archive. Verified-not-hypothesised. Same pattern as Slice B's eu_eurlex / eu_parliament rollout.

## Done-definition for F1 Slice A (0.68.0)

- 6 new connector modules ship (`bi`, `bnm`, `bsp`, `cbn`, `cbe`, `cbk`); each has `iter_<cb>_decision`; speeches functions ship where the site has a clean separate English archive (best-effort, documented per-CB).
- Each connector declares `PARSER_SCHEMA_VERSION = 1` + `FALLBACK_POLICY` + uses `fetch_with_fallback` + emits `parser_schema_mismatch` telemetry events on schema mismatch.
- Per-connector golden fixtures shipped (one per `iter_*` function).
- AST coverage assertion enforces the 6-module list.
- R5_03 demo notebook executes cleanly with `connector_health()` showing the 6 new sources.
- `pyproject.toml` at `version = "0.68.0"`; CHANGELOG entry; ARCHITECTURE.md "F1 Slice A — SE Asia + Africa CBs" subsection.
- Pyodide compat passes; full narrative suite no new regressions vs. the post-Slice-B baseline.

## Open follow-ups (queued for later slices)

- **F1 Slice B** — business surveys (IFO Germany, Tankan via BoJ, ZEW Germany, Conference Board, Michigan Consumer Sentiment).
- **F1 Slice C** — forecaster + uncertainty surveys (BoE DMP, ECB SPF, SNB Survey, Atlanta Fed BIE, Atlanta Fed BU, ECB CIS).
- **F1 Slice D** — alt-data (Google Trends, earnings call transcripts via SEC EDGAR, 10-K risk factors, container-shipping rates, satellite night-lights).
- **F1 Slice E+** — local-language coverage (Bahasa, Arabic, Swahili), `iter_<cb>_minutes` for the 6 CBs where they're published, additional CB coverage (Eastern Europe: NBP/CNB/MNB/BNR/CBRT; smaller advanced: SNB, DNB, CBI).
- Sibling sub-projects from the original brainstorm: F3 unified panel builder, S2 interpretation, S4 cross-source synthesis 2.0, T1 cookbook, T2 onboarding.

---

## Spec self-review (inline)

- **Placeholder scan**: no TBD/TODO. The hypothesised per-CB policies and landmarks are explicitly flagged as "verify-at-implementation" — that's the verified-not-hypothesised pattern from Slice B, not a placeholder.
- **Internal consistency**: 6 connectors listed identically in module map, scope decisions, staging, and done-definition. `FALLBACK_POLICY` integration described identically across Section 3 (per-connector), Section 4 (data flow), and Section 5 (failure semantics).
- **Scope check**: 6 connectors × 1-2 functions = ~12 entries; 13-commit slice. Larger than Slice B (12 commits) but smaller than Slice A (23 commits). Single implementation plan handles it.
- **Ambiguity check**: "best-effort" speeches functions are flagged explicitly — the spec says drop the function if a CB doesn't have a clean separate archive. No ambiguity about what "shipped" means per-CB.
