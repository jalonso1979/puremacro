## 2024-10-14 - Inline Jupyter HTML Accessibility
**Learning:** External CSS cannot be relied upon in Jupyter/Juno output cells. When generating rich HTML dialogs (like `show_colab_offload_dialog`), hover states and focus-visible outlines for keyboard accessibility must be injected directly into the HTML string via an inline `<style>` block.
**Action:** Always inject a `<style>` block and scoped classes (e.g., `pm-colab-btn`) alongside inline HTML strings when returning them from Python backend functions meant for notebook display.
