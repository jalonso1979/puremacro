## 2024-11-20 - [Add Accessibility to Inline HTML Jupyter Output]
**Learning:** Inline HTML in Jupyter environments needs explicitly added ARIA attributes (`aria-hidden` for emojis/icons and `aria-label` for links) to remain accessible for users relying on screen readers.
**Action:** Ensure `IPython.display.HTML` usages include screen reader accessible tags for icons and target="_blank" links (with `rel="noopener noreferrer"`).
