## 2024-10-24 - Inline HTML Accessibility in Jupyter
**Learning:** When injecting raw HTML into Jupyter notebooks using `IPython.display.HTML`, standard web accessibility principles (like `aria-hidden` on emojis and `aria-label`/`rel` on external links) are often overlooked because the output is transient, but they remain critical for screen reader users consuming the notebook.
**Action:** Always scan inline HTML blocks in `IPython.display.HTML` for decorative emojis and missing link labels, applying standard ARIA attributes.
