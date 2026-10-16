## 2024-06-19 - Add hover and focus styles to Jupyter inline HTML
**Learning:** Jupyter notebooks often use inline HTML for rich outputs. However, inline CSS `style="..."` attributes cannot support pseudo-classes like `:hover` or `:focus-visible` for keyboard accessibility.
**Action:** When adding focus/hover states to Jupyter inline HTML generated in Python, inject a small `<style>` tag within the HTML string, apply a class to the target element, double-escape CSS curly braces (`{{` and `}}`) if inside an f-string, and use `!important` to override existing inline styles.
