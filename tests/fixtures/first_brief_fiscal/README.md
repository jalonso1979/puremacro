# Fiscal brief record

`headline.json` is the frozen record behind the fiscal policy brief pages,
`docs/first_brief_fiscal.md` and `docs/es/first_brief_fiscal.md`. It holds the
default run (the full sample, 1890Q1-2015Q4) and the experiment without World
War II (1941Q3-1945Q4, by Ramey and Zubairy's rule), estimated on the bundled
data file whose path and SHA-256 it records.

Regenerate it from actual estimates, from the repository root, with

    python -m tools.reference_validation.refresh_fiscal_brief --output /tmp/fiscal-brief

The command writes the brief's English and Spanish outputs into the output
directory, copies `headline.json` here, rewrites the four marker blocks of both
pages (`<!-- fiscal-brief:{block} -->`) and copies the two figures to
`docs/assets/first_brief_fiscal/`. Commit the record, the pages and the figures
together.

`tests/test_first_brief_fiscal.py` checks that a fresh run reproduces the record:
estimates, standard errors, bands and effective F to a relative tolerance of
1e-8, Anderson-Rubin bounds to 1e-6, and every weak flag exactly. The package
version is recorded but not compared, because it changes at each release. The
same tests require each page block to equal what the record generates.
Regenerate after an intended change to the estimates or to the wording, never to
silence a failing comparison.
