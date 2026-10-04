# Independent audit of retained release evidence

**Result: the inspected artifact, freeze, documentation and installed-workflow
evidence is internally consistent.** This audit was completed while the second
full release gate was still running. It does not report that pending gate as a
pass; final acceptance must use its actual result.

The checks below independently read archive bytes, JSON manifests, installed
files and saved CSV outputs. They did not rerun the full test suite, reinstall
the candidate, modify source or replace scientific results. The recorded
findings are based on successful console tool output; no separate raw audit
log was captured.

## Retained wheel and source distribution

Both retained archives match their recorded byte lengths and SHA-256 hashes,
and are identical by hash to the original archives in the temporary candidate
directory.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `puremacro-4.3.0-py3-none-any.whl` | 16,861,366 | `191cafa80eef2f8663cd6301cc87a80dd69f1a87d8f94f91d4545705577d357c` |
| `puremacro-4.3.0.tar.gz` | 16,472,075 | `c15646a30d8342db8506cdcaf46439276cc9ede83232603a84ef625186d60267` |

The wheel contains **884 entries**, including **878 package source/resource
files**. Every one of those 878 files matches all of the following:

- Its bytes in the source distribution.
- Its original candidate-freeze hash.
- Its amended candidate-freeze hash.
- The current repository file.
- Its copy in the isolated wheel installation.

All current package Python modules are present. No wheel entry is a PNG,
bytecode file, shared library or MAT file. All **883 hashed `RECORD` entries**
were independently checked against archive contents, including URL-safe
base64 SHA-256 encoding and byte length; the `RECORD` file itself appropriately
has no self-hash.

The sdist's non-package entries are packaging metadata, `LICENSE`, `README.md`,
`pyproject.toml` and `setup.cfg`. The repaired notebooks and documentation
pages are not shipped package content. Retaining the same wheel and sdist
after those repairs is therefore consistent with the unchanged shipped bytes.

## Original and amended freezes

The original provenance records **977 package/build inputs** and **3,295 gate
inputs**. The amended gate map has **zero mismatches** against the inspected
working tree. All **81 changes** from the original gate freeze are explicitly
accounted for: 70 notebook paths, six documentation paths, and five generated
PNG files absent from the wheel. There are no additional unrecorded changed
paths among the original frozen inputs.

The retained copies of both the original and amended provenance JSON files
are byte-identical to the review-folder copies. The amended provenance's
notebook-repair and inline-display manifest hashes match those files.

All **six immutable original failure-evidence files** still match
`original-evidence-sha256.json`. The first full-suite failure record has not
been overwritten or relabeled as a pass. The pending second full gate has its
own log. Targeted notebook checks are correctly distinguished from a completed
full release gate in the candidate README.

## Strict documentation

All **381 recorded strict-documentation source hashes** match the inspected
files, with zero mismatches. The recorded strict MkDocs process completed with
exit status zero. The final documentation has separate provenance from the
original package build, as the README explains.

The five rebuilt notebooks had already received a separate read-only review:
their rendered cell sources match Jupytext exactly, execution counts are
complete and consecutive, and no error outputs remain. The two bilingual
pairs match cell-type sequences and normalized code ASTs, with Dynare equations
and calibrations checked separately. Notebook 45's scientific text matches
the refreshed results after the documented Markdown corrections.

## Installed workflow and benchmark evidence

Saved artifact hashes verify for both installed SW07 phases and all three
installed trade runs: **7 calibration**, **8 validation**, and **12 artifacts
each** for GE, zero-tariff and CLI outputs. The two installed SW07 phases
contain two samples each and **16 variant fits**. Variant records share their
sample seed, and calibration and validation streams are disjoint. The saved
tiny-reference comparison explicitly retains unbounded critical values.

The installed smoke script checks the full exact-moment API against the fast
expectation API and exact completed-checkpoint replay. It checks that all
loaded `puremacro` modules come from the isolated installation. Its process
finished with status zero. The network and statsmodels-denial implementation
was inspected, including both positive controls; both retained process-control
markers report successful controls.

The GE API and CLI incidence CSVs are identical. Saved zero-tariff EV, CV,
income changes and transfers are zero within numerical tolerance. The fiscal
CSV substantiates maximum residuals of `3.814697265625e−6` MXN for allocation
and `0.01229095458984375` MXN for income reconciliation; the latter agrees up to
ordinary CSV decimal parsing. These are small numerical residuals relative to
the national monetary totals, not evidence that the synthetic trade economy
has been empirically estimated.

The installed benchmark report contains **8 cases, 7 passing, overall FAIL**,
with no execution errors. The sole failure is RR2010's published horizon-10
t-statistic: observed `−3.5249960750002387`, printed reference `−3.53`, unchanged
absolute tolerance `0.005`, and excess `3.9249997611e−6`. Its response and
trough-horizon comparisons pass. The discrepancy remains visible and has not
been relaxed or converted to a software pass.

## Claim review and scope

The candidate README correctly distinguishes installed workflow checks from
scientific validation, identifies oracle weights and fixed-DGP limits, retains
the synthetic trade assumptions, and leaves final gate acceptance pending.
No package or provenance blocker was found.

One wording qualification was reported: the scientific study's unit/application
tests use other seeds, but installed-wheel smoke checks replay two samples per
phase using study seed `20261002`. The study README now distinguishes these
cases and explicitly excludes the installed smoke replays from its scientific
sample counts. This required only review prose, not a source or artifact
change.

The original 1,398 scientific samples and 5,592 variant fits are separate from
these installation checks. Neither the small smoke runs nor an eventual
software-gate pass establishes composite-null inference, parameter confidence
coverage, or an empirically identified tariff-policy effect.
