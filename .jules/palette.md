## 2025-02-18 - Accessible Inline HTML in Jupyter Notebooks
**Learning:** Decorative emojis and external links in inline HTML for Jupyter notebook outputs need explicit accessibility attributes (aria-hidden="true" for emojis, rel="noopener noreferrer" and aria-label for links) to be screen reader friendly.
**Action:** Always add aria-hidden="true" to decorative symbols and proper ARIA labels/rel attributes to target="_blank" links when generating HTML outputs using IPython.display.HTML.
