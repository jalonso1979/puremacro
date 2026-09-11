
## 2024-05-18 - Making inline HTML for Jupyter notebooks accessible
**Learning:** When using tools like `IPython.display.HTML` to generate UI within Jupyter notebooks, those elements must meet accessibility standards just like a normal web page. Screen readers can trip on decorative emojis or symbols (like `🚀` or `↗`), and external links opening in a new window (`target="_blank"`) need security contexts (`rel="noopener noreferrer"`) and explicit warnings (`aria-label="..."`) so users know the link will take them out of their current context.
**Action:** When writing inline HTML strings, always wrap decorative symbols in `<span aria-hidden="true">` and include `rel="noopener noreferrer"` and `aria-label` on links utilizing `target="_blank"`.
