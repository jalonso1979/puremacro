🎯 **What:** The code health issue addressed
Refactored the `kalman_filter` function in `puremacro/state_space.py` to reduce its length and complexity. The function contained a very long `for` loop encompassing both exact-diffuse updates and standard non-diffuse updates. These two logical branches have been extracted into separate helper functions: `_kalman_diffuse_update` and `_kalman_standard_update`.

💡 **Why:** How this improves maintainability
The original `kalman_filter` function was overly long, making it difficult to read, understand, and maintain. Extracting the complex logic inside the main loop into dedicated helper functions improves readability by clearly separating concerns and keeping the main control flow concise.

✅ **Verification:** How you confirmed the change is safe
- Verified syntax with `py_compile`.
- Executed core state space tests using `pytest tests/test_state_space_coverage.py`, `tests/test_models_smm_coverage.py`, `tests/test_var_estimate_coverage.py`, and `tests/test_forecast_compare_coverage.py` to ensure no regressions were introduced.
- Requested code review which approved the change and verified its correctness.
- Removed auxiliary scripts used to build the refactored code.

✨ **Result:** The improvement achieved
The `kalman_filter` function is now much shorter and easier to follow, with the detailed mathematical updates safely encapsulated in specialized helper functions without changing behavior.
