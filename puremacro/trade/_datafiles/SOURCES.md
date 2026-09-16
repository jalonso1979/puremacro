# Third-party data bundled with `puremacro.trade`

puremacro's own code is MIT-licensed (see `LICENSE` at the repository root).
**That licence does not cover the data files in this directory.** They are
third-party material redistributed with puremacro so that the trade model and
its parity suites run without asking you for a file. Their terms are below.

If you publish results computed from these files, cite the original source,
not puremacro.

---

## `icio_77c_11s.npz` — OECD Inter-Country Input-Output tables

| | |
|---|---|
| Array | `data`, shape `(850, 1078)`, float64 |
| Origin | OECD Inter-Country Input-Output (ICIO) tables |
| Publisher | Organisation for Economic Co-operation and Development (OECD) |
| Source page | <https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm> |
| Coverage | 77 countries × 11 composite sectors, 3 final-demand categories |
| Terms | Redistributed under the OECD terms of use, which permit reuse and redistribution with attribution to the OECD as the source |

**Attribution.** Work using this matrix should credit the OECD ICIO tables.
The OECD is not affiliated with puremacro and does not endorse it.

**What was changed.** The published OECD tables were aggregated to the
77-country, 11-sector layout used by the sectoral-misallocation computation,
then converted from MATLAB `data_77c_11s.mat` to a compressed `.npz`. The
conversion is bit-exact: `numpy.array_equal` against the MATLAB array was
asserted when the file was written, and `tests/test_no_matlab_dependency.py`
checks the file still loads from the installed package.

---

## `trade_reference_solutions.npz` and `trade_results_workbook.npz` — reference outputs

| | |
|---|---|
| Origin | MATLAB `results_77c_11s_*.mat` and `results.xls` of the sectoral-misallocation computation |
| Author | Jorge Alonso-Ortiz |
| Terms | Same MIT licence as puremacro's own code |

These are the equilibrium arrays, calibration arrays and workbook sheets that
puremacro's trade parity suites compare against. They are verbatim copies,
dtype included, so the comparison remains an **external** check rather than
puremacro grading its own output. Regenerating them with puremacro would turn
every one of those tests into a tautology.

`results_77c_11s_t10_54.mat` could not be included: the source file is a
dataless placeholder that reads zero bytes. Tests needing that scenario skip
by name rather than substitute a value.

---

## Machine-readable metadata

`MANIFEST.json`, `REFERENCE_MANIFEST.json` and `WORKBOOK_MANIFEST.json` in this
directory record each file's arrays, shapes, provenance and SHA-256.
