## 2026-09-01 - Testing Internal Array Functions
**Learning:** Testing internal numpy array operations like `_within_demean` should cover the happy path and critical edge cases such as missing indexes (gaps), single items, and all items in the same group. Using `np.testing.assert_allclose` is standard for array comparison testing.
**Action:** When adding tests for numerical functions, explicitly cover empty/missing group scenarios since these often trip up zero division or indexing assumptions.
