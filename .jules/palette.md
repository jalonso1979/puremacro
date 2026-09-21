## 2024-05-19 - Ensure A11y for dynamically generated rich HTML elements

**Learning:** Puremacro dynamically generates rich HTML payload code when sending interactive notebooks to users or executing locally via Jupyter/Juno. These are not standard web application elements, so we have to ensure proper accessibility practices (like `aria-hidden` and `aria-label`) are baked right into the payload's strings in python.

**Action:** Look for instances of `IPython.display.HTML` or custom HTML strings being generated inside python methods (like `puremacro.runtime.colab`). Ensure emojis are wrapped in `<span aria-hidden="true">` to skip screen-reader announcement and `target="_blank"` links include `rel="noopener noreferrer"` for security/perf, and `aria-label` to warn screen readers about the context switch.
