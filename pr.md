🎯 **What:** The `load_klems_panel` function in `puremacro/klems.py` was overly long and complex. It was refactored by extracting the table loading and the `p_equip_index` computation logic into smaller, dedicated helper functions (`_load_raw_tables` and `_compute_p_equip_index`).

💡 **Why:** Refactoring monolithic functions into smaller, single-responsibility helper functions greatly improves the code's readability and maintainability. This also makes specific pieces of logic easier to test.

✅ **Verification:**
1. Created unit tests for the extracted `_compute_p_equip_index` function to ensure calculations are correct and edge cases (NaNs) are handled.
2. Created a test for the empty cache scenario of `load_klems_panel`.
3. Validated that all test suites pass successfully.
4. Validated that formatting and lint checks (ruff) pass with zero errors.

✨ **Result:** The `load_klems_panel` is now modular and highly readable, while completely preserving the original behavior and output schema.
