submit -b "refactor-harvest-narrative-corpus" -m "🧹 Refactor harvest_narrative_corpus to address long function issue

🎯 **What:** The harvest_narrative_corpus function in puremacro/narrative/harvest.py was refactored. The source resolution logic was extracted into _resolve_target_sources and the per-source extraction logic was extracted into _harvest_source.
💡 **Why:** This improves the readability and maintainability of the codebase by breaking a large orchestration function into smaller, well-defined helper functions.
✅ **Verification:** Ran tests/test_narrative_harvest.py to ensure that narrative harvesting logic is preserved. Additionally, verified type imports to avoid any runtime errors.
✨ **Result:** The harvest_narrative_corpus function is now much shorter and easier to follow, making future modifications less error-prone."
