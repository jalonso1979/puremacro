
## 2024-05-24 - Jupyter Notebook Inline HTML Output Accessibility
**Learning:** Inline HTML generated for Jupyter notebook outputs (e.g., using `IPython.display.HTML`) requires the same accessibility considerations as regular web pages, such as wrapping decorative emojis in `<span aria-hidden="true">` and adding `rel="noopener noreferrer"` and `aria-label`s to external links.
**Action:** Always verify that inline HTML outputs in Jupyter notebooks include necessary ARIA attributes and security practices for external links.
