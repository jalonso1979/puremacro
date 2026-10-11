## 2025-02-15 - Injecting Custom CSS in Notebook HTML Outputs
**Learning:** Standard external CSS classes are not available or reliably targeted within inline HTML representations (like those produced by `IPython.display.HTML`) in Jupyter or Colab environments.
**Action:** When adding hover or focus-visible UX enhancements to these components, inject a small inline `<style>` block directly into the HTML string instead of relying on external classes.
