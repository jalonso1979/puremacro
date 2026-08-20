# Running anywhere: iPad, Juno, and the browser

`puremacro`'s numerical core is pure numpy + scipy + pandas + matplotlib, which
is what makes it *importable* on an iPad. Importable is not the same as usable.
Four things a workstation provides silently are missing on a tablet, and this
page is about closing that gap rather than restating the promise.

## Start here: what can this machine do?

```python
from puremacro import runtime
print(runtime.report())
```

```
puremacro runtime
  host       : pyodide 3.12.7 (wasm32)
  device     : tablet
  network    : js-fetch (call runtime.enable_browser_network())
  parquet    : unavailable -> use puremacro.runtime.store / pocket
  threads    : no (1 cpu, unknown)
  writable fs: yes
  backends   : numpy
```

Detection is heuristic by necessity — no Python API answers "am I inside Juno?"
— so it reads `sys.platform`, the iOS sandbox path (`/var/mobile/`), and the
browser user agent, and every field can be pinned when it guesses wrong:

| variable | effect |
|---|---|
| `PUREMACRO_HOST` | `cpython` / `pyodide` / `unknown` |
| `PUREMACRO_DEVICE` | `workstation` / `tablet` / `browser` / `unknown` |
| `PUREMACRO_SOCKETS` | force the socket verdict |
| `PUREMACRO_PARQUET` | force the parquet verdict |

`runtime.capabilities()` returns the same information as a frozen dataclass, and
records which fields came from an override in `.overridden`.

## 1. There are no sockets

Under Pyodide there is no TCP stack, so `requests` and `urllib` both fail and
every `puremacro.fetch.*` call dies — even though the estimator core imported
perfectly. The browser can still make requests; it just does them in
JavaScript. One call reroutes the whole existing fetch layer over that:

```python
from puremacro import runtime
from puremacro.fetch import fetch_xrate_monthly

runtime.enable_browser_network()
fx = fetch_xrate_monthly(["MEX"])
```

Nothing in `fetch` or `narrative.sources` is modified. The switch replaces the
two chokepoints those modules already funnel through: the urllib call in
`puremacro._http` and the `requests` module object in `puremacro.fetch._http`.
`runtime.disable_browser_network()` puts both back.

Two browser limits are worth knowing before you depend on this:

- **CORS.** The browser refuses cross-origin responses without
  `Access-Control-Allow-Origin`. Some public statistical endpoints send it;
  many WAF-fronted government sites do not. A blocked request raises
  `TransportError` naming CORS as the likely cause. Pass `proxy=` to route
  through a CORS proxy you control.
- **No timeouts, no custom User-Agent.** A synchronous `XMLHttpRequest` on the
  main thread cannot set either, so both are accepted and ignored rather than
  raising — which also means the WAF-bypass user-agent trick in
  `narrative/sources/RETRY_POLICY.md` §7 does not work in a browser.

## 2. There is no pyarrow

`pyarrow` is a base dependency with no Pyodide wheel, so every parquet path
(`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`) is unreachable. numpy's
own `.npz` container has no such problem — it is zlib plus a header, implemented
in numpy itself.

`runtime.store` is a DataFrame ⇄ npz codec built on that: one array per column
plus a JSON schema recording dtypes, index structure and column labels, so a
frame survives the round trip with its index intact. It handles `PeriodIndex`,
tz-aware datetimes, `Categorical`, pandas nullable extension dtypes and
`MultiIndex`, and refuses to pickle arbitrary objects rather than writing an
archive that will not load elsewhere. On a 5,000×8 quarterly panel it is also
smaller than parquet (310 KB vs 409 KB).

Above it sits **`puremacro.pocket`**: pack data where the network and pyarrow
are, open it where they are not.

```python
from puremacro import pocket

# workstation, online
pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

# iPad, airplane mode
cart = pocket.load("g7.pmz")
panel = cart.frame()          # sha256-checked on read
cart.provenance.vintage       # '2026-08-19'
print(cart.summary())
```

A `.pmz` is a plain zip: a JSON manifest plus one npz per frame, both stdlib or
numpy, so it opens without `puremacro` installed at all. `pocket.snapshot(fn,
*args, path=...)` runs a call and records the reproducing expression in the
manifest. And because getting a *file* onto an iPad is often more friction than
the analysis, a cartridge also travels as text:

```python
blob = pocket.to_base64("g7.pmz")     # paste into a message, a note, an email
pocket.from_base64(blob, "g7.pmz")    # on the other machine
```

Cartridges are a transport format, not a trust boundary: the checksums detect
corruption in transit, and nothing more.

## 3. The app gets suspended

iPadOS stops a backgrounded app. A four-minute bootstrap does not survive
someone answering a message, and neither does a Metropolis-Hastings chain or a
Krusell-Smith solve. The tablet is not so much slow as *interruptible*, and an
ordinary estimator call is one opaque block that either finishes or is lost.

```python
import numpy as np
from puremacro import longrun

job = longrun.bootstrap(one_draw, 2000, checkpoint="irf.ckpt")
job.run(seconds=30)     # 240/2000 · 12% · ~220s of compute left
job.run(seconds=30)     # ... and again, in a later session
bands = np.percentile(job.result(), [5, 95], axis=0)
```

Draw *i* always uses `default_rng([seed, i])`, so results are **invariant to
chunk size and to how many sessions the job took** — a run resumed across five
sittings is bit-identical to one that went straight through. That is what makes
a resumed run publishable rather than merely finished. `job.result()` raises
unless the job is complete, so a half-run bootstrap cannot be mistaken for a
full one, and a checkpoint carries a fingerprint of the job that wrote it and
refuses to be resumed by a different one.

Checkpoints are plain npz loaded with `allow_pickle=False`, so a job started on
the iPad can be finished on the workstation.

## 4. Compute is smaller

```python
runtime.fit(n_boot=2000)              # -> {'n_boot': 400} on a tablet
svar = runtime.budgeted(cholesky_svar)  # clamps cost arguments of a call
```

Both are **opt-in**. No estimator consults the budget, so every default is
exactly what it always was and a script that runs on your laptop produces the
same numbers after this feature landed. Only parameters that change *cost* are
clamped (`n_boot`, `n_draws`, `n_grid`, `n_sim`); `horizon` changes what is
being estimated, so it is deliberately left alone.

`with runtime.override("tablet"):` rehearses tablet budgets on a laptop.

## Installing on Juno

`pyarrow` has no Pyodide wheel, so a dependency-resolving install fails. Install
the core without resolution and add what you need:

```python
import micropip
await micropip.install("puremacro", deps=False)
await micropip.install(["numpy", "scipy", "pandas", "matplotlib", "requests"])
```

## What is actually verified

`python tools/release_check.py --pyodide` (gate 6) boots a real Pyodide kernel
and runs the `pyodide_smoke`-marked suite: **29 tests green under Pyodide
0.28.3**, including all ten `runtime.store` frame round-trips, cartridge
pack/verify/base64 transport, the `longrun` invariance property, and
`dsge.build` solving a model to its closed form.
`test_detection_matches_the_interpreter_it_is_running_on` cross-examines the
interpreter and confirms on the target that `host == "pyodide"`,
`sockets is False`, `js_fetch is True`, `parquet is False` and
`backends == ("numpy",)`.

**One gap, stated plainly:** the synchronous-`XMLHttpRequest` body of
`runtime.transport` has never executed. Node-hosted Pyodide has no
`XMLHttpRequest` — it is a browser API — so gate 6 cannot reach it either. Its
callers, error paths and module patching are tested; the XHR call itself awaits
confirmation from a real browser or Juno session.
