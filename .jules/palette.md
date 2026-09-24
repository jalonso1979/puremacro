## 2024-05-18 - Accessibility in Jupyter HTML outputs
**Learning:** Decorative emojis in inline HTML outputs for Jupyter notebooks can be read awkwardly by screen readers, and external links without proper aria-labels and security attributes can cause issues for accessibility and security.
**Action:** Always wrap decorative emojis with `<span aria-hidden="true">` and add `rel="noopener noreferrer"` along with an explicit `aria-label` to `<a target="_blank">` tags when generating HTML for Jupyter outputs.
