# Releasing puremacro

How to cut a release. Read §1 once; after that §3 is the whole procedure.

*Last verified against a real release: **1.3.1**, 2026-08-20.*

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
| `ci.yml` | push / PR to `main` | pytest on 12 targets (ubuntu + macos + windows × Python 3.10–3.13), then `release_check.py --no-tests` on ubuntu/3.12 |
| `release.yml` | push of a `v*` tag | build → `twine check` → publish to PyPI (`environment: pypi`, `id-token: write`) |
| `pages.yml` | push to `main`, or manual | builds the JupyterLite playground + mkdocs site, deploys to Pages |

**There is exactly one publishing workflow.** A second one (`publish.yml`) used to exist on
the same `v*` trigger; only `release.yml` is registered with the PyPI trusted publisher, so
`publish.yml` failed on every single tag while publishing nothing. It was deleted in 1.3.1.
If you ever see two PyPI workflows again, one of them is wrong.

## 2. The gate

`tools/release_check.py` is the pre-tag check. Four gates run by default, two are opt-in:

| gate | what it proves | notes |
|---|---|---|
| 1 test baseline | pytest failures == `tests/known_failures.json` | that file currently holds **zero** entries, i.e. the suite must be fully green. ~20 min |
| 2 Pyodide contract | `tests/test_pyodide_compat.py` green | static check of the import contract |
| 3 public API snapshot | regenerated API == `tests/fixtures/public_api_snapshot.json` | 301 modules, 137 result classes |
| 4 version sync | `pyproject.toml` == `puremacro/__init__.py` == `CHANGELOG.md` == `CITATION.cff` | all four |
| 5 examples gallery | `--examples` | reads `docs/examples_gallery.json` |
| 6 Pyodide smoke | `--pyodide` | builds the wheel and boots a real Pyodide kernel |

```bash
python tools/release_check.py                 # the four defaults
python tools/release_check.py --no-tests      # fast: gates 2-4 only, seconds
python tools/release_check.py --pyodide       # add the real-kernel smoke test
```

> Gate 4 reads **four** version-bearing files. `CITATION.cff` was added to it after it
> silently went stale at 1.3.0 while the package shipped 1.3.1 — three files were bumped
> and the fourth was not, and nothing in the release path noticed. If you add a fifth
> place the version is written, add it to `gate_version_sync` at the same time.

**When Gate 3 fails** it prints the exact symbols added or removed. If the change is
intended, regenerate the fixture:

```bash
python -c "import sys, json, pathlib; sys.path.insert(0, 'tests'); \
from test_public_api import collect_current_api; \
pathlib.Path('tests/fixtures/public_api_snapshot.json').write_text(json.dumps(collect_current_api(), indent=2) + '\n')"
```

Commit that as its own change and say in the message *why* the surface moved — a widened
API and a renamed one look identical in the diff otherwise.

## 3. Cutting a release

Everything here is local and reversible until step 7.

1. **Land the work on `main`.** Nothing else in this list matters if the fix you are
   releasing is not in the commit you are about to tag. See §4.
2. **Write the CHANGELOG section** — `## X.Y.Z (YYYY-MM-DD)`, a one-line summary in bold,
   then `### Added` / `### Fixed` / `### Internal` / `### Known issues`. Describe what a
   user can now do, or now no longer trips over.
3. **Bump the version in all four places:** `pyproject.toml`, `puremacro/__init__.py`,
   the CHANGELOG heading, and **`CITATION.cff`**.
4. **Run the gate:** `python tools/release_check.py`. All four must pass.
5. **Sanity-build and inspect the artifact**, because this is the last point at which a
   mistake is free:
   ```bash
   rm -rf dist/ && python -m build && python -m twine check dist/*
   python - <<'EOF'
   import zipfile, glob
   z = zipfile.ZipFile(glob.glob("dist/*.whl")[0])
   print(sorted(n for n in z.namelist() if n.endswith("__init__.py"))[:5])
   EOF
   ```
   Confirm the wheel really contains the module you just wrote. A file that was never
   `git add`ed is in your working tree, in your tests, and **not** in the wheel.
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
submittable**. Four things stand between it and the submit button, in rough order of effort.

### 6.1 The paper is too short and missing sections

JOSS now asks for **750–1750 words**; the draft is **~500**. It also now requires six
sections, and the draft has three of them:

| section | state |
|---|---|
| Summary | ✓ |
| Statement of need | ✓ |
| State of the field | ✗ — needs an explicit comparison with statsmodels, linearmodels, EconML, Dynare/`sequence-jacobian`, and a build-vs-contribute justification |
| Software design | ✗ — the pure-NumPy/Pyodide constraint and what it cost is exactly this section |
| Research impact | ✗ — publications, courses or external users that actually use it |
| AI usage disclosure | ✗ — required, and non-trivially true here |

The present `# Features` section is not one of the six; fold it into *Software design* or
*State of the field*.

### 6.2 The ORCID is a placeholder

`paper/paper.md` still reads `orcid: 0000-0000-0000-0000`. Register at
<https://orcid.org> and put the real one in.

### 6.3 The public history is one month long, and squashed

JOSS looks for roughly **six months of public development history with activity spanning
it**. This repo was created 2026-06-01, and its first commit is a single
`Initial public release (v0.92.0)` squash of 1,256 files dated 2026-07-20 — the
pre-split history from the monorepo did not come across. A reviewer sees one month and
47 commits.

Options, in preference order:
1. **Wait.** Keep shipping; the calendar fixes this around 2026-12.
2. **Restore the history** by re-doing the split with `git filter-repo --subdirectory-filter`
   against the original monorepo, which preserves per-file history rather than squashing.
   Rewrites every SHA, so do it before more people depend on the repo, not after.
3. **Explain it** in the submission thread. Editors do accept "the code has a longer
   history in <repo>, extracted on <date>" — but you have to say it up front.

### 6.4 Zenodo comes *after* acceptance, not before

**This doc previously said the opposite.** JOSS: *"Upon successful completion of the
review, authors will deposit a copy of the repository with a data-archiving service such
as Zenodo or figshare, get a DOI for the archive."* You do **not** need a DOI to submit.

When you do get there, note that Zenodo archives on a **GitHub Release**, not on a tag.
This repo has tags through `v1.3.1` but only one Release (`v1.0.0`), so
`gh release create vX.Y.Z --generate-notes` is a step you will need.

### 6.5 Then submit

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
