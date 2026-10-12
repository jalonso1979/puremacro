## 2024-05-13 - Focus/Hover Styles for Colab Badge
**Learning:** The Colab Offload dialog uses inline HTML in Jupyter. While it's largely accessible with ARIA labels, the "Open Google Colab" button lacks hover/focus styles, making it hard for keyboard navigation or mouse users to see they are interacting with it. In Jupyter, standard external CSS classes cannot be reliably targeted, so a small `<style>` tag should be injected to handle hover/focus.
**Action:** Adding hover and focus-visible states using an inline `<style>` tag scoped to a unique class or ID.
