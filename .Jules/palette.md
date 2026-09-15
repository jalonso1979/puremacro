## 2024-09-15 - Improve HTML Dialog Accessibility
**Learning:** IPython.display.HTML is used to render inline HTML dialogs. Ensure inline emojis, decorative texts like arrows are hidden to screen readers with `aria-hidden="true"`, and external links with `target="_blank"` include `rel="noopener noreferrer"` and `aria-label` for screen readers.
**Action:** Always check IPython/Jupyter inline HTML rendering for accessibility best practices.
