
## 2024-05-18 - Improve accessibility of inline HTML outputs in Jupyter Notebooks
**Learning:** Decorative emojis in HTML strings (such as Jupyter or Colab notebook HTML representations via `IPython.display.HTML`) can cause accessibility issues for screen readers. Similarly, external links with `target="_blank"` lack screen reader clarity and pose a security/performance risk if missing `rel="noopener noreferrer"`.
**Action:** Always append `aria-hidden="true"` to spans containing decorative emojis/icons, and strictly pair `target="_blank"` with `rel="noopener noreferrer"` and an explicit `aria-label` detailing that the link opens in a new tab.
