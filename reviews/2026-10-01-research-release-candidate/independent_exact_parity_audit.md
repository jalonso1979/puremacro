# Independent audit of exact notebook source parity

This records an observed independent, read-only audit of the 30 rendered
notebooks listed in `notebook-exact-source-parity-repair.json`. The audit ran
with `/tmp/puremacro-research-candidate-venv/bin/python`: Python 3.12.13 and
Jupytext 1.19.5. Results were observed in the audit command's console output;
no separate raw log was retained. No source, notebook, test, or release-check
file was changed by the audit.

## Observed results

- **30 of 30 notebooks passed.** All 566 cell sources matched the corresponding
  `jupytext.read()` cell sources exactly, without stripping whitespace. Cell
  counts, order, and types matched in every notebook.
- All 30 current Python source hashes matched both the repair manifest and the
  second freeze in `amended-candidate-provenance.json`, amended at
  `2026-10-01T22:27:09.070283+00:00`.
- All 30 current rendered notebook hashes matched the repair manifest's
  `rendered_after_sha256` values.
- The 66 changed cells were all code cells. Each pre-repair and post-repair
  pair had identical `ast.dump(ast.parse(source), include_attributes=False)`.
- Across the reviewed notebooks there were 228 code cells and 274 stored
  outputs, with zero error outputs.

## Independent reconstruction of the frozen prior files

The audit read the actual per-cell trailing-newline counts from
`second-gate-all-display-parity.json`; it did not assume the newline delta.
For each notebook it copied the current JSON in memory, restored the recorded
removed trailing newlines only in the recorded cells, converted each restored
source with `splitlines(keepends=True)`, and serialized the reconstructed
notebook using `json.dumps(ensure_ascii=False, indent=1) + "\n"`.

**All 30 reconstructed SHA-256 hashes matched both the repair manifest's
`rendered_before_sha256` and the second freeze's `gate_source_sha256`.**
The changed-cell indices also matched both records exactly, totaling 66.
The audit explicitly checked that every notebook-level field and every
non-source cell field was unchanged between current and reconstructed JSON.

The exact frozen-hash reconstruction independently establishes that the repair
preserved all other content, including stored outputs, cell and notebook
metadata, execution counts, cell identifiers, and cell order. Combined with
the AST checks, the evidence supports a trailing-newline-only correction with
no numerical code or scientific-output change.

## Scope

This is a record of the observed independent parity and preservation audit,
not a new notebook execution or a full test-suite run. The final full release
gate is a separate check with its own result and evidence. This audit does
not imply that the full release gate has passed.
