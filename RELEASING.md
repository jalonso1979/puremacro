# Releasing puremacro

How to cut a release. Read §1 once; after that §3 is the whole procedure.

*Release preparation rechecked for **4.3.0**, 2026-09-20: full baseline gate,
clean artifact build, installed-wheel checks, documentation and playground build.
See the [verification record](reviews/2026-09-20-release-4.3.0/REPORT.md) and
[release page](https://github.com/jalonso1979/puremacro/releases/tag/v4.3.0) for
results and publication workflows. The baseline permits eleven documented
failures already present in 4.2.0; it is not an entirely green raw test suite.*

## 1. What the setup actually is

- **Repo:** [`github.com/jalonso1979/puremacro`](https://github.com/jalonso1979/puremacro),
  public, standalone (the monorepo split is done — there is no `puremacro/` subdirectory
  any more, the package root *is* the repo root). Default branch **`main`**.
- **PyPI:** [`pypi.org/project/puremacro`](https://pypi.org/project/puremacro/), published
  by **Trusted Publishing (OIDC)** — no API token exists and none is needed.
- **Docs / playground:** <https://jalonso1979.github.io/puremacro/>, deployed by
  `pages.yml` on every push to `main`.

### The three workflows

| file | trigger | what it does |
|---|---|---|
| `ci.yml` | push / PR to `main` | pytest on 9 targets (ubuntu + macos + windows × Python 3.11–3.13; 3.10 is below `requires-python` and was dropped), then `release_check.py --no-tests` on ubuntu/3.11 and ubuntu/3.12, and a strict `mkdocs build` on ubuntu/3.12 |
| `release.yml` | push of a `v*` tag | build → `twine check` → publish to PyPI (`environment: pypi`, `id-token: write`) |
| `pages.yml` | push to `main`, or manual | builds the JupyterLite playground + mkdocs site, deploys to Pages |

**There is exactly one publishing workflow.** A second one (`publish.yml`) used to exist on
the same `v*` trigger; only `release.yml` is registered with the PyPI trusted publisher, so
`publish.yml` failed on every single tag while publishing nothing. It was deleted in 1.3.1.
If you ever see two PyPI workflows again, one of them is wrong.

## 2. The gate

`tools/release_check.py` is the pre-tag check. Five gates run by default, two are opt-in:

| gate | what it proves | notes |
|---|---|---|
| 1 test baseline | pytest `FAILED` + `ERROR` node ids are a subset of `tests/known_failures.json` | At 4.3.0, 11 independently confirmed previous-release failures remain: flexible configuration replacement, harmonic-mean covariance validation, legacy welfare closure and numerical parity edge cases. Each entry records a workaround and prior CI evidence; see the changelog. Recovered cases are reported for removal. Setup errors count. The wall-clock budget is 5400 s, overridable with `PUREMACRO_BASELINE_TIMEOUT_S`; complete runs take tens of minutes. |
| 2 Pyodide contract | `tests/test_pyodide_compat.py` green | static check of the import contract |
| 3 public API snapshot | regenerated API == `tests/fixtures/public_api_snapshot.json` | the fixture is the count (404 modules with `__all__`, 285 result classes at 3.4.0); the gate prints every symbol that moved |
| 4 version sync | `pyproject.toml` == `puremacro/__init__.py` == `CHANGELOG.md` == `CITATION.cff` == the wheel pin in `playground/jupyter_lite_config.json` | all five |
| 7 min-Python syntax | every `.py` under `puremacro/`, `tools/` and `tests/` parses on the `requires-python` floor (3.11) | `ast.parse(feature_version=(3, 11))` plus a PEP 701 f-string scan (quote reuse, backslashes and comments inside `{...}`), and a real `compile()` under `python3.11` when one is on PATH. Gates 1–6 run under whatever interpreter invokes the script, so a 3.11-only SyntaxError passed all of them while every 3.11 CI leg died at collection. `notebooks/` is not scanned: jupytext sources may carry bare `%magics`. Seconds |
| 5 examples gallery | `--examples` | reads `docs/examples_gallery.json` |
| 6 Pyodide smoke | `--pyodide` | builds the wheel and boots a real Pyodide kernel |

```bash
python tools/release_check.py                 # the five defaults
python tools/release_check.py --no-tests      # fast: gates 2, 3, 4 and 7, seconds
python tools/release_check.py --pyodide       # add the real-kernel smoke test
```

> Gate 4 reads **five** version-bearing files. `CITATION.cff` was added to it after it
> silently went stale at 1.3.0 while the package shipped 1.3.1 — three files were bumped
> and the fourth was not, and nothing in the release path noticed. The playground wheel
> pin followed at 3.4.0: `build_playground.sh` rewrites it from the wheel it builds, so
> the deployed site never lagged, but the tracked file read 1.9.0 while the package
> shipped 3.3.0. If you add a sixth place the version is written, add it to
> `gate_version_sync` at the same time.

**When Gate 3 fails** it prints the exact symbols added or removed. If the change is
intended, regenerate the fixture **from a clean copy of the commit**, not from the live
working tree (a synced tree can carry modules that are not in git, and they would be
baked into the fixture):

```bash
rm -rf /tmp/pm-clean && mkdir /tmp/pm-clean && git archive HEAD | tar -x -C /tmp/pm-clean
cd /tmp/pm-clean && PYTHONPATH=. python -c "import sys, json, pathlib; sys.path.insert(0, 'tests'); \
from test_public_api import collect_current_api; \
pathlib.Path('tests/fixtures/public_api_snapshot.json').write_text(json.dumps(collect_current_api(), indent=2) + '\n')"
cp /tmp/pm-clean/tests/fixtures/public_api_snapshot.json tests/fixtures/
```

Commit that as its own change and say in the message *why* the surface moved — a widened
API and a renamed one look identical in the diff otherwise.

**When Gate 7 fails** it names the file and line. The usual cause is an f-string that
only Python 3.12 accepts: a string inside `{...}` that reuses the f-string's own quote,
or a backslash inside `{...}` (`f"{s.replace('_', r'\_')}"` is the one that took the
3.11 CI legs down before 3.4.0). Hoist the expression into a local first.

## 3. Cutting a release

Everything here is local and reversible until step 7.

1. **Land the work on `main`.** Nothing else in this list matters if the fix you are
   releasing is not in the commit you are about to tag. See §4.
2. **Write the CHANGELOG section** — `## X.Y.Z (YYYY-MM-DD)`, a one-line summary in bold,
   then `### Added` / `### Fixed` / `### Internal` / `### Known issues`. Describe what a
   user can now do, or now no longer trips over.
3. **Bump the version in all five places:** `pyproject.toml`, `puremacro/__init__.py`,
   the CHANGELOG heading, **`CITATION.cff`**, and the wheel pin in
   `playground/jupyter_lite_config.json` (`./wheels/puremacro-X.Y.Z-py3-none-any.whl`).
4. **Run the gate:** `python tools/release_check.py`. All five must pass.
5. **Sanity-build and inspect the artifact**, because this is the last point at which a
   mistake is free. Build from a clean export, not from the live tree: setuptools seeds
   the sdist from a stale, gitignored `puremacro.egg-info/SOURCES.txt` when one is
   present, and a live-tree build here once produced a 110 MB sdist carrying
   `playground/dist/`, `tests/` and `notebooks/` (and `matlab/`, removed in 4.0.0),
   plus a 9 MB wheel with 63
   research PNGs — nothing like the 4 MB / 3.4 MB artifacts `release.yml` ships from a
   fresh checkout.
   ```bash
   rm -rf /tmp/pm-build && mkdir /tmp/pm-build && git archive HEAD | tar -x -C /tmp/pm-build
   (cd /tmp/pm-build && python -m build && python -m twine check dist/*)
   python - <<'EOF'
   import zipfile, glob
   z = zipfile.ZipFile(glob.glob("/tmp/pm-build/dist/*.whl")[0])
   names = z.namelist()
   print(len(names), "entries;", sum(n.endswith(".png") for n in names), "png (expect 0)")
   print(sorted(n for n in names if n.endswith("__init__.py"))[:5])
   EOF
   ```
   Confirm the wheel really contains the module you just wrote. A file that was never
   `git add`ed is in your working tree, in your tests, and **not** in the wheel — and
   `git archive` is exactly what makes that visible.
6. **Tag, annotated, on the exact commit you verified:**
   ```bash
   git log --oneline -1                      # is this really the commit?
   git tag -a vX.Y.Z -m "puremacro X.Y.Z ..."
   ```
7. **Push — this is the irreversible step.**
   ```bash
   git push origin HEAD:main
   git push origin refs/tags/vX.Y.Z          # ← fires release.yml, publishes to PyPI
   ```
8. **Watch it land:**
   ```bash
   gh run list --workflow=release.yml --limit 1
   ```
9. **Verify from PyPI, not from your checkout:**
   ```bash
   python -m venv /tmp/v && /tmp/v/bin/pip install --no-cache-dir puremacro==X.Y.Z
   /tmp/v/bin/python -c "import puremacro; print(puremacro.__version__)"
   ```
   Expect a few minutes' lag: the JSON API shows the new version before pip's index CDN
   does, so an immediate `pip install` can still fail with *"no matching distribution"*.
   That is propagation, not a failed release.

## 4. The traps, all of which have actually bitten

**A tag is a commit, not a branch.** `v1.3.0` was created, then a fix landed, then the tag
was pushed — still pointing at the pre-fix commit. The release workflow published exactly
what the tag pointed at, and the shipped 1.3.0 lacked the fix it was cut for. **Before
pushing any tag:** `git merge-base --is-ancestor <fix-commit> vX.Y.Z && echo OK`.

**PyPI is append-only.** A version can be yanked but never replaced or re-uploaded. If a
release goes out wrong, the only remedy is another version number — which is why 1.3.0 was
followed within the hour by 1.3.1. Yanking hides a release from resolvers; it does not free
the version string.

**A green local suite is not a green CI.** Dependencies are floors, not pins
(`pandas>=2.0`), so CI resolves the newest release while your machine sits on whatever it
installed months ago. `runtime.store` was broken on pandas 3 for two releases while every
local run was green. Before a release, check CI on `main` — and when a subsystem is
version-sensitive, test against the resolved version deliberately:
```bash
python -m venv /tmp/pd3 && /tmp/pd3/bin/pip install "pandas==3.0.5" numpy scipy matplotlib pytest requests
PYTHONPATH=$PWD /tmp/pd3/bin/python -m pytest tests/ -q -k "<subsystem>"
```

**Shipping a known bug is a decision, not an accident.** If you release with something
broken, put it under `### Known issues` in that version's CHANGELOG section, name the
workaround, and name the version that will fix it. A user should find it in the changelog,
not in a traceback.

**The changelog must describe the artifact, not the branch.** When a fix slips to the next
version, move its entry too. 1.3.0's section was edited after the fact to stop claiming a
fix that only shipped in 1.3.1.

## 5. What only you can do

- Authorize the push and the publish.
- PyPI account actions: the trusted publisher is registered for
  `jalonso1979/puremacro` + workflow `release.yml` + environment `pypi`. Changing the repo
  name, the workflow filename, or the environment name breaks publishing until the
  publisher entry is updated to match.
- GitHub settings: Pages source, branch protection, environment approvals.
- Yanking a bad release on PyPI.

## 6. JOSS submission

*Requirements checked against joss.readthedocs.io on 2026-08-20. Check them again
before submitting — they have tightened at least once.*

`paper/paper.md` (+ `paper.bib`, `scorecard.png`) is drafted but **is not currently
submittable**. The length and section requirements (§6.1) and the ORCID (§6.2) were
closed in the 3.3.0–3.4.0 rewrite and are kept below as the record of what the paper now
has to keep true. What is left is §6.3: the public repository's first commit is dated
2026-07-20, so the roughly six months of public history JOSS looks for are not there
until early 2027. Until then the paper is a draft that has to stay in step with the
release it describes (§6.5).

### 6.1 Length and sections — done, keep it that way

JOSS asks for **750–1750 words**. The rewritten draft is **~1,670** (body text, front
matter and HTML comments excluded), and it carries all six required sections:

| section | state |
|---|---|
| Summary | ✓ |
| Statement of need | ✓ |
| State of the field | ✓ — statsmodels / arch / linearmodels as oracles, Dynare, `sequence-jacobian`, HARK, QuantEcon.py, and the build-vs-contribute argument |
| Software design | ✓ — the import invariant, the stored-oracle strategy, graceful degradation, and the costs |
| Research impact statement | ✓ — teaching use at ITAM, reproducible material, community readiness |
| AI usage disclosure | ✓ — drafted; the AUTHOR comment in that section still has to be worked through before submitting |

There is no longer a `# Features` section. The draft tracks **3.4.0**: whenever the module
count, line count, test count, validation-check count or notebook count moves, refresh
them in the same pass (§6.5).

### 6.2 The ORCID

`paper/paper.md` carries `orcid: 0000-0002-5941-9928`, the only "Jorge Alonso Ortiz"
(ITAM) record in the public ORCID registry as of 2026-09-13. Confirm it is yours and
delete the AUTHOR comment that follows the front matter.

### 6.3 The public history is short — disclose it, do not rewrite it

JOSS looks for roughly **six months of public development history with activity spanning
it**. This repo's first commit is a single `Initial public release (v0.92.0)` squash of
1,256 files dated 2026-07-20, so a reviewer sees a short public log for a library of ~750
shipped modules. The earlier history is real but lives in the private `uncertainty_examples`
monorepo, under `puremacro/`, from 2026-04-28.

**This was investigated and rejected.** Re-splitting with
`git filter-repo --subdirectory-filter puremacro` against the monorepo branch
`feature/puremacro-v0.93.0` does work — it yields 965 commits spanning 2026-04-28 to
2026-07-25 — and a scan of every text blob in that history for AWS, FRED and Banxico key
formats came back clean. It was still the wrong trade:

- it buys ~11 weeks of earlier eligibility (2026-10-28 instead of 2027-01-19);
- it rewrites **every** SHA, so `v1.0.0`–`v1.3.1` move and the commits PyPI actually built
  from stop existing under those hashes;
- it publishes 965 commit messages out of a private repository;
- the trees do not line up — the private history carries `matlab/`, `.claude/`,
  `egg-info` and a v0.93.0 tip against the public repo's v0.92.0 root, so the version
  timeline would read "0.93.0" and then "Initial public release (v0.92.0)";
- and the first cleaning pass still missed the nested `.DS_Store` files, which is the
  real argument: after a force-push, every miss is permanent.

**Do this instead.** Say it plainly in the *comments to the editor* box on the submission
form, and again in the review thread if asked:

> `puremacro` was developed from 2026-04-28 inside a private monorepo
> (`uncertainty_examples`), as the `puremacro/` subdirectory, and was extracted into this
> standalone public repository on 2026-07-20. The extraction squashed the prior history
> into the initial commit, so the commit log here begins in July; development did not.
> Release history is continuous across the move, from v0.92.0 through the current
> release, and is visible in `CHANGELOG.md` and in the tag list.

Costs nothing, is true, and editors accept it. Revisit only if an editor specifically
asks for the history to be present in the repository itself.

### 6.4 Zenodo comes *after* acceptance, not before

**This doc previously said the opposite.** JOSS: *"Upon successful completion of the
review, authors will deposit a copy of the repository with a data-archiving service such
as Zenodo or figshare, get a DOI for the archive."* You do **not** need a DOI to submit.

When you do get there, note that Zenodo archives on a **GitHub Release**, not on a tag.
This repo has tags through `v3.3.0` but only one Release (`v1.0.0`), so
`gh release create vX.Y.Z --generate-notes` is a step you will need.

### 6.5 Refresh the paper's numbers with every release it claims to describe

`paper/paper.md` quotes counts that move under it. Recompute all of them in one pass and
edit the text; each was last refreshed for 3.4.0 on 2026-09-16.

```bash
# modules and lines (the "Scale" paragraph)
git ls-files 'puremacro/*.py' 'puremacro/**/*.py' | wc -l
git ls-files 'puremacro/*.py' 'puremacro/**/*.py' | xargs cat | wc -l
# tests collected (the same paragraph)
PYTHONPATH=. python -m pytest --collect-only -q tests/ | tail -1
# library modules in the import sweep ("An import invariant")
PYTHONPATH=. python -c "import sys; sys.path.insert(0,'tests'); \
from test_pyodide_compat import _shippable_modules as m; print(len(m()))"
# validation gallery: checks, subsystems, and the figure's legend counts
PYTHONPATH=. python -c "import puremacro.validation as v; s=v.scorecard(); \
print(len(s), s['subsystem'].nunique(), s['mechanism'].value_counts().to_dict(), bool(s['passed'].all()))"
# bilingual showcase pairs ("Reproducible material")
ls notebooks/[0-9][0-9]_*.py | grep -v '_es\.py$' | wc -l
# commit counts (the AI usage disclosure)
git rev-list --count origin/main
git log origin/main -i --grep='co-authored-by: claude' --oneline | wc -l
```

If the mechanism split (`package` + `scipy` + `published` = external reference,
`analytical`, `internal`) or the subsystem counts move, `scorecard.png` is stale too:
regenerate it with `python paper/make_scorecard_fig.py` before touching the caption.

### 6.6 Then submit

```bash
# preview the compiled paper exactly as JOSS will build it
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) \
  --env JOURNAL=joss openjournals/inara
```

Then open <https://joss.theoj.org/papers/new> with the repository URL. Review is
conversational and public on GitHub; you are expected to answer reviewers within 2 weeks
and land changes within 4–6.

What JOSS checks that this repo already satisfies: an OSI licence (MIT, and GitHub's
licensee detects it), documentation, automated tests, and a functioning CI.
