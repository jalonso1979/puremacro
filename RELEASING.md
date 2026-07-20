# Releasing puremacro

How to ship puremacro to real users. Pick **one release path** (§2), then follow
its steps; §3 (GitHub Pages playground) and §4 (what only you can do) apply to all.

## 1. Current state (verified 2026-05-31)

- **Remote:** `github.com/jalonso1979/uncertainty_examples` (public). Working "main"
  is `feature/subnational-labor-uncertainty-us` (= `origin/HEAD`); `origin/main` also exists.
- **This branch** `feature/regime-uncertainty-companion-phase2a` is **not pushed** and is
  **~467 commits ahead** of the working-main (~326 ahead of `origin/main`). It carries the
  regime-uncertainty + VFI + local-LLM + hardening + bilingual work; **~339 commits touch
  `puremacro/`**.
- **Monorepo-subdir layout:** the git root is `uncertainty_examples/`; its own
  `pyproject.toml` (v0.1.0) **excludes `puremacro*`**. The shippable package is the
  **`puremacro/` subdirectory** (own `pyproject.toml`, currently **v0.92.0**), with its
  CI workflow at the monorepo root (`.github/workflows/puremacro-ci.yml`,
  `working-directory: puremacro`).
- **Pre-flight (any path):** `cd puremacro && python -m pytest -q` (5,535 passed),
  `python -m mypy` (clean), `python tools/release_check.py` if present. Verify the name
  `puremacro` is free on PyPI (https://pypi.org/project/puremacro/) — if taken, choose an
  alternative (e.g. `puremacro-econ`) and update `pyproject [project] name`.

## 2. Release paths — pick one

### Path A — split `puremacro/` into its own public repo + PyPI  **(recommended)**
The cleanest home for a standalone public good: a clean `pip install puremacro`, its own
CI/Pages/issues, no monorepo baggage.
1. **(you)** Create an empty public repo, e.g. `github.com/<you>/puremacro` (no README).
2. Extract the subdir *with history* into a fresh tree:
   ```bash
   pip install git-filter-repo                              # not currently installed
   git clone "<path to a CLEAN clone of uncertainty_examples>" /tmp/pm-split
   cd /tmp/pm-split && git checkout feature/regime-uncertainty-companion-phase2a
   git filter-repo --subdirectory-filter puremacro          # rewrites history -> subdir as root
   ```
   The repo-root CI/release/Pages workflows are ALREADY placed at
   `puremacro/.github/workflows/{ci,release,pages}.yml`, so the extraction **promotes them
   to the new repo's root `.github/` automatically — no manual copy/adapt needed**. (The
   monorepo keeps its own root `.github/workflows/puremacro-ci.yml` for the transition.)
3. `git remote add origin https://github.com/<you>/puremacro.git && git branch -M main && git push -u origin main`.
4. CI (`ci.yml`) runs on push — its first run validates the extras set + the 3.11 floor.
5. **PyPI (trusted publishing, no API token):** on PyPI create the project `puremacro`
   under your account → add a *trusted publisher* for `<you>/puremacro`, workflow
   `release.yml`, environment `pypi`. Add a tag-triggered `release.yml` that builds
   (`python -m build`) and publishes via `pypa/gh-action-pypi-publish`. Then
   `git tag v0.92.0 && git push --tags`.
6. Deploy the playground to Pages (§3) from the new repo.

### Path B — publish the subdir to PyPI from the monorepo
Faster to PyPI; keeps the monorepo (clutter and all).
1. **(you)** `git push -u origin feature/regime-uncertainty-companion-phase2a`; open a PR
   to `main` (or merge). NB the 467-commit divergence — expect a large PR; squash-merge is
   reasonable for the package history if you don't need every intermediate commit on main.
2. Add a root `.github/workflows/release.yml` that, on a `puremacro-v*` tag, runs
   `python -m build` **in `puremacro/`** and publishes with trusted publishing (publisher =
   this repo, this workflow, env `pypi`).
3. **(you)** Configure the PyPI trusted publisher for `jalonso1979/uncertainty_examples` +
   `release.yml`. Tag `puremacro-v0.92.0` and push.
4. Pages playground per §3 (root-level workflow building `puremacro/playground`).

### Path C — GitHub-only (no PyPI yet)
Leanest; zero PyPI setup.
1. **(you)** `git push -u origin feature/regime-uncertainty-companion-phase2a` (and/or PR to main).
2. Users install from the subdir:
   ```bash
   pip install "git+https://github.com/jalonso1979/uncertainty_examples.git@feature/regime-uncertainty-companion-phase2a#subdirectory=puremacro"
   ```
3. Enable GitHub Pages for the playground (§3).
Trade-off: a clunky install URL and no version on PyPI; trivial to upgrade to B later.

## 3. GitHub Pages playground (bilingual, browser-runnable, $0)
The JupyterLite playground (`playground/`) is one build from live and now bundles the
**EN + ES** notebooks.
1. Add a Pages workflow (root for B/C; repo-root for A) that runs
   `bash playground/build_playground.sh` and uploads `playground/dist/` via
   `actions/upload-pages-artifact` + `actions/deploy-pages`.
2. **(you)** Settings → Pages → Source = GitHub Actions.
3. Result: a public URL where anyone runs the EN/ES showcase notebooks in-browser at $0.
*(Local check: `cd puremacro && bash playground/build_playground.sh` → `playground/dist/`.)*

## 4. What only you can do
- Decide the path (A/B/C) — it's your repo/account/public presence.
- Authorize the **push** (this branch isn't pushed; I won't push without your go-ahead).
- Create the GitHub repo (A) and/or the **PyPI project + trusted publisher** (A/B) — account actions.
- Enable GitHub Pages (Settings).
- Confirm the PyPI name `puremacro` is free (else pick another).

## 5. What I can do for you (on your word)
- Path A: run the `git filter-repo` extraction in a fresh clone, adapt the CI to repo-root,
  write `release.yml` + the Pages workflow, stage everything for your `git push`.
- Path B/C: write `release.yml` (trusted-publishing) + the Pages workflow; prepare the push.
- Either way: a final `release_check` pass and the version-tag commands.
- I will not push, create repos, or publish without your explicit go-ahead (per repo policy).

## 6. JOSS submission (after the public release)
The paper is prepared at `paper/paper.md` (+ `paper.bib`, `scorecard.png`). JOSS
requires the software to be public and archived, so submit **after** a release path
(§2) is live:
1. Fill the author **ORCID** in `paper/paper.md` (currently a `0000-…` placeholder).
2. Tag a release and archive the repo to get a **Zenodo DOI** (Zenodo ↔ GitHub integration).
3. Open a submission at https://joss.theoj.org/papers/new (repo URL + the Zenodo DOI).
4. Regenerate the figure if the validation gallery changed: `python paper/make_scorecard_fig.py`.
Reviewers check: open-source license (MIT ✓), docs (✓), tests (✓), and a clear
statement of need (✓). I prepare the paper; you perform the submission.
