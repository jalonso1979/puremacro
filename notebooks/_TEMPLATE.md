# Enriched showcase template

The structure every showcase notebook (`NN_topic.py`) follows. Edit the `.py`
(jupytext percent), then `python tools/build_notebooks.py NN_topic`. Keep the
Spanish twin `NN_topic_es.py` code-identical (translate prose only).

Cells, in order:

1. **Motivating question** — 1-2 sentences. What economic question does this answer?
2. **The method in math** — one `# %% [markdown]` cell, the governing equations in
   LaTeX ($...$ / $$...$$), compact (~4-8 lines).
3. **Intuition** — a `**Intuition.**` lead-in paragraph mapping the math to economics.
   No emoji; bold used sparingly.
4. **Worked code** — the runnable example; 1-2 inline comments per block say *why*.
   Pure-numpy, fixed seed, inline `assert`s on headline numbers.
5. **Read the output** — a `# %% [markdown]` cell interpreting the printed numbers and
   the hero figure explicitly: state the headline result and what it means, not
   only which diagnostics exist.
6. **Your turn** — a fill-in cell: a working default marked `# ← change this …` (any
   `# ←` line plus a short description) with its advertised range, a downstream
   `assert` that checks the answer, then 2-3 graded prompts
   (basic → stretch). The assertion must hold over the whole advertised range
   (test its corners and each prompt's values) and must be able to fail: nothing
   guaranteed by a root-finder's bracket, a library identity or the definition of
   an interval. Prefer "predict, then run, then assert a sign, direction or
   ordering", grading against a planted truth where the data are simulated, and
   invariance checks (ordering, numeraire, units). The committed notebook must
   execute green.
7. **How comprehensive is this?** — 2-3 lines pointing to the other puremacro entry
   points that use the same machinery.

## Evidence and interpretation

- State the data source, units, sample, and availability dates before estimation.
  Label simulated data and hand-built calibrations at the top, including when
  series or country names are real. A model preset is not an empirical dataset.
- Define the estimand and shock normalization. Label nominal wages, real wages,
  gross output, GDP, and welfare according to what the code actually computes.
- For forecasts, hold out later observations before fitting preprocessing,
  tuning parameters, or model coefficients. Compare against a simple benchmark.
- Attach confidence intervals to the statistic being plotted. Resample the
  aggregate statistic itself; averaging component interval endpoints is invalid.
  Distinguish pointwise intervals from simultaneous bands. A seeded interval can
  miss the truth; report that outcome rather than choosing another seed.
- Numerical identities and convergence checks establish internal consistency.
  They do not establish identification, empirical validity, or forecast accuracy.
- Make exercise assertions valid for the advertised parameter choices. Compare
  counterfactuals with the same starting conditions and random draws, and derive
  displayed numbers from results rather than hard-coding them into prose.
- Name each oracle and say whether it is independent of puremacro (a published
  table, a closed form derived by hand, another package's frozen output) or an
  internal consistency check. A second route through puremacro code is not an
  independent oracle. Cite the source that actually states a result.
- Round printed floats (`{:.4g}`, `{:.1e}` for residuals) and end figure cells with
  `plt.show()`, so rebuilding a notebook does not rewrite unchanged outputs.

Use the Pyodide-compatible numerical stack (`numpy`, `scipy`, `pandas`,
`matplotlib`), the `_nbstyle` preamble (`apply_style()`, `palette(n)`), and fixed
seeds. Document optional desktop dependencies when an example needs them.
Rebuild both language editions, inspect the figures, and validate the resulting
artifacts with `python tools/build_notebooks.py --validate-only NN_topic NN_topic_es`.
