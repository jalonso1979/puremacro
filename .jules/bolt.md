## 2026-09-01 - Code Health Improvement: Preserving comments in refactoring
**Learning:** When extracting helper functions from a massive block of code during a refactor, it's very easy to accidentally delete original inline comments that explain the business logic (the "why"). These comments are crucial for code health and readability.
**Action:** Always meticulously copy over original business-logic comments into the docstrings or bodies of the newly created helper functions to ensure no domain knowledge is lost during structural refactorings.
