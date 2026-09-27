## 2024-05-14 - Accessibility improvements for Jupyter HTML outputs
**Learning:** Decorative emojis in generated HTML need `aria-hidden="true"`. Also, links opening in new tabs (`target="_blank"`) need `rel="noopener noreferrer"` and an explicit `aria-label` (like "Open Google Colab in a new tab").
**Action:** Next time I modify or add `IPython.display.HTML` strings, I will verify these elements to ensure they are screen reader friendly.
