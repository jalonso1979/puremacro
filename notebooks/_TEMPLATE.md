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
   the hero figure explicitly.
6. **Your turn** — a fill-in cell: a working default marked `# ← change this …` (any
   `# ←` line plus a short description), a downstream `assert` that holds for the
   default, then 2-3 graded prompts
   (basic → stretch). The committed notebook must execute green.
7. **How comprehensive is this?** — 2-3 lines pointing to the other puremacro entry
   points that use the same machinery.

Constraints: numpy-only / Pyodide-safe (no statsmodels/linearmodels/arch/bs4), the
`_nbstyle` preamble (`apply_style()`, `palette(n)`), deterministic seeds.
