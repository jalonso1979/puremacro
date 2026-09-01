## 2024-05-19 - Using itertuples instead of iterrows
**Learning:** `iterrows()` in pandas is extremely slow because it creates a Series object for every row. `itertuples(index=False, name=None)` is significantly faster as it returns plain Python tuples and bypasses the expensive Series construction.
**Action:** Always prefer `itertuples` (or better yet, vectorization) over `iterrows` when iterating over DataFrames.
