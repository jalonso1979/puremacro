# puremacro Batch 5a — ALFRED real-time vintages + X-13 seasonal adjustment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The two data-quality follow-ups the batch-4 drafts flagged. (A) ALFRED: does the surviving own-district response (h = 5, WC p = 0.048) and the robust LOO-national response (h = 12, WC p = 0.007) hold on outcomes *as first published* rather than current-vintage? (B) X-13ARIMA-SEATS: the border design ran on NSA county LAUS urate and flagged two isolated short-horizon significants as seasonal noise — adjust the county U/E levels with genuine X-13 and re-run the border LPs; if the noise reading is right, those coefficients die while the medium-run picture stands.

**Scope facts established by probes (2026-07-21):**
- Key-free ALFRED does NOT exist: `fredgraph.csv` silently ignores `vintage_date` (served data through 2026-05 for a 2005 vintage request); `alfredgraph.csv` 404s. The keyed ALFRED API (`output_type=4`, initial release only) gives the whole first-release history in one call per series (~102 calls). The house `credentials` layer already defines the `fred` service (`FRED_API_KEY` / `~/.puremacro/credentials.toml`) — **no key is configured on this machine**, so Task A ships ready-to-run and BLOCKED pending the user's key.
- State outcomes (`{ST}UR`, `{ST}NA`) are BLS-SA already; the NSA series in the pipeline is county LAUS U/E (border design). X-13 applies there.
- Genuine X-13ARIMA-SEATS v1.1.57 installed machine-locally: x13org/x13prebuilt mac-arm64 `x13ashtml` under `~/.local/x13ashtml/`, bridged to statsmodels' ASCII-build file-naming expectations by a `~/.local/bin/x13as` wrapper (statsmodels reads `<base>.err`; the html build writes `<base>_err.html`; saved tables like `.d11` are plain text in both). Verified end-to-end through `puremacro.sa.x13._x13_one` with the STL fallback rigged to raise: seasonal amplitude 7.99 → 0.39 on a toy series. NOT a repo artifact — the SA tool must hard-fail with install instructions when `x13_available()` is False (a silent STL fallback would misrepresent "X-13 adjusted" outputs).

## File map

### New files
- `tools/run_main_street_phase5_realtime.py` — ALFRED first-release panel + phase-3 baseline / phase-4 horse-race re-estimation on (a) current-vintage outcomes restricted to the vintage-supported window and (b) first-release outcomes. `credentials.require('fred')` up front; per-series vintage-coverage table; Croushore-Stark first-release long differences documented.
- `docs/plans/2026-07-21-puremacro-batch5a-alfred-x13-plan.md` — this plan.

### Modified files
- `tools/run_main_street_phase4_border.py` — `--sa` mode: X-13 the monthly county U and E levels (multiplicative/auto transform per house wrapper defaults, `outlier=True`, no trading-day), urate_sa = U_sa/(U_sa+E_sa), quarterly means; outputs suffixed `_sa`; STL-fallback count logged and recorded in the manifest; hard error if the binary is unavailable.
- `docs/research/main_street_uncertainty/DRAFT.md` — §5 border paragraph gains the SA verdict; limitations updated (NSA caveat resolved or sharpened); ALFRED status noted.
- `CHANGELOG.md` — batch-5a section.

## Tasks

- [x] **Task X1: `--sa` border re-run.** DONE 2026-07-21 — after one REAL failure caught by design: the first pass silently fell back to STL on 2,002/2,018 series (LAUS county series have missing months; an index gap makes statsmodels write literal 'nan' into the spc and X-13 errors). Fixed (gap-free calendars in the SA path) + a hard >5%-fallback gate so "X-13 adjusted" can never mean "STL adjusted". Final: 2,017/2,018 genuine X-13. Pre-registered reading CONFIRMED at fast-B: h=0 (WC p 0.019→0.45) and h=2 (0.004→0.37) die; medium-run wrong-signed null stands. Full B=999 run for final p's in flight.
- [x] **Task X2: draft + changelog update.** DONE — changelog batch-5a section written; DRAFT §6 SA paragraph lands with the full-B numbers.
- [x] **Task A1: ALFRED tool.** DONE 2026-07-21 — UNBLOCKED: the user's key was found in `uncertainty_examples/.env` and installed to `~/.puremacro/credentials.toml` (0600, never echoed). API quirk found by probe: `output_type=4` 400s without the explicit full realtime span (documented in the tool). State-UR archives start ~2005–2007; publication lags 47–66 d.
- [x] **Task A2: run + drafts.** DONE — full B=999 run. Verdict: magnitudes survive first-release outcomes (own h=5 89%, h=9 104%, LOO h=12 87% with WC p=0.003); own-district *significance* does not (p 0.03→0.18); and the matched window exposes a 3× crisis-era concentration of the state-level differential. DRAFT.md §6 + limitations updated.

## Out of scope
State-series X-13 (already BLS-SA); re-running the LOO horse race under SA (state outcomes unchanged); packaging the x13 binary (machine-local; Pyodide-pure target forbids a hard binary dep).
