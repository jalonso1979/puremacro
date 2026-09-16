## 2026-08-31 - Accessible JupyterLite HTML Outputs
**Learning:** Inline HTML rendered in Jupyter notebooks (e.g., via `IPython.display.HTML`) is read by screen readers. Decorative emojis must be hidden (`aria-hidden="true"`), and `target="_blank"` links need `rel="noopener noreferrer"` along with descriptive `aria-label`s.
**Action:** When generating rich HTML displays for Jupyter/Pyodide, explicitly add accessibility attributes to spans and links.
