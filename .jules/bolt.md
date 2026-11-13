## 2026-08-31 - Memory reallocation in numpy simulation loops
**Learning:** Calling `np.concatenate` to manage rolling history buffers in tight simulation loops (like for Generalized IRF trajectories across `H` steps and `M` parallel histories) is a significant bottleneck due to constant reallocation and copying of the whole buffer, even though the arrays are relatively small per iteration. Pre-allocating the full future length works but increases peak memory. Simply slicing and re-assigning inplace (`buf[:, :-1] = buf[:, 1:]` and `buf[:, -1] = y`) gives ~15% speedup vs `np.concatenate` without the overhead and memory jump of full pre-allocation.
**Action:** When shifting time buffers inplace in high-throughput hot loops with Numpy, use slicing and inplace assignment instead of `np.concatenate` to minimize reallocation.

## 2026-10-21 - Regex overlap masking in single-pattern combining
**Learning:** Combining multiple string terms into a single regex using alternation `|` (e.g. `\b(?:term1|term2)\b`) or even lookaheads `(?=\b(term1|term2)\b)` inherently changes the semantics of overlapping matches if terms share the same starting word/prefix. The regex engine will consume the first valid match at an index and skip other valid shorter/longer overlapping terms starting at the same character, unlike iterating independent regex checks. This breaks exact keyword counts (like for narrative sentiment scoring).
**Action:** When matching multiple static terms against text (e.g., scoring lexicons), do not combine terms into a single regex with `|` (OR) if overlapping matches must be preserved. Instead, iterate through the terms and cache individual `re.compile` calls using `@functools.lru_cache` to bypass repetitive escape and cache-lookup overhead while guaranteeing identical matching logic. The gain is modest (measured 1.05-1.3x), because `re` already caches compiled patterns.

## 2023-10-25 - Row duplication in Pandas DataFrames
**Learning:** Using `.iterrows()` with list comprehensions to duplicate rows in a Pandas DataFrame (e.g. for expanding annual to quarterly periods) is significantly slower than vectorized Pandas/NumPy operations. Expanding rows iteratively creates high object allocation overhead.
**Action:** When uniformly expanding or duplicating pandas DataFrame rows N times (e.g., converting annual to quarterly data), use `df.iloc[np.repeat(np.arange(len(df)), N)].reset_index(drop=True)` for a highly performant vectorized approach instead of `iterrows()`.
## 2026-11-20 - Vectorized lag gathering via sliding_window_view
**Learning:** Constructing short-run lagged matrices (e.g. for VECM models) using a python loop, list comprehensions, and `np.concatenate` is extremely slow and memory inefficient for large matrices because of repeated allocations. Even slicing and combining arrays involves heavy overhead compared to native strided views.
**Action:** Use `numpy.lib.stride_tricks.sliding_window_view` to extract overlapping windows across time dimensions, then sequence slices, transposes, and `.reshape()` to natively collapse the arrays without creating temporary intermediate objects, producing a 10x+ speed up.

## 2023-10-25 - Memory reallocation in pandas concat over groupby
**Learning:** Using `groupby` followed by `sort_values` on each group and then `pd.concat(groups)` is an anti-pattern when trying to sort within groups. It incurs a massive overhead due to the O(N_groups) python loop and the memory allocations required to build intermediate dataframes and concatenate them. It can be fully vectorized by globally sorting the dataframe by the grouping keys and then by the target column.
**Action:** When sorting rows within groups in a Pandas DataFrame, do not use `pd.concat([g.sort_values(c) for _, g in df.groupby(keys)])`. Instead, use `df.sort_values(keys + [c])` for an identical result that is magnitudes faster.

## 2024-11-28 - Vectorized group aggregation for frequency conversion in Pandas
**Learning:** Iterating over large pandas groups and using `.resample().mean()` per variable slice on a wide, long-form dataframe (e.g., in `_aggregate_monthly_to_quarterly`) generates immense overhead due to individual object creation and Python loop execution. This caused a 5.0-second delay for 400k rows.
**Action:** Replace `df.groupby(...).resample(...)` iterations with purely vectorized mappings. Map exact dates to period starts (e.g. `.dt.to_period('Q').dt.start_time`), slice exact aggregation needs first, and use a single `groupby(['code', 'variable', 'q_date']).agg()` for speedups exceeding 10x.

## 2024-11-20 - Vectorized Chained Matrices in Pandas DataFrames
**Learning:** When needing to perform chained matrix multiplication on rows of a Pandas DataFrame grouped by time (e.g. chaining 3 monthly transition matrices to create a quarterly matrix), iterating through `groupby()` and using `iterrows()` to manually construct and multiply arrays incurs massive Python overhead.
**Action:** Instead, assign a sequence index (e.g., `cumcount()`) to the grouped rows, filter out only the required N rows per group, sort appropriately, and extract the required columns into a single flat NumPy array. Reshape this array to `(num_groups, N, rows, cols)` and perform batched matrix multiplication (`data[:, 0] @ data[:, 1] ...`) for >25x performance gains.

## 2024-11-07 - Vectorized Matrix Chaining
**Learning:** Chaining monthly transition matrices into quarterly matrices via `iterrows` over a Pandas DataFrame of transition probabilities is extremely slow.
**Action:** Extract the probabilities into a NumPy array, reshape it to `(num_months, n_states, n_states)`, and use batched matrix multiplication (`@`) for massive speedups (~150x). This fully leverages NumPy's vectorized operations and avoids Python loop overhead.
