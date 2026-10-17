## 2025-01-20 - Adding interactive styles to inline HTML
**Learning:** Standard external CSS classes are unreliable for inline HTML in Jupyter notebooks. Need to inject a small `<style>` tag directly within the HTML string, double-escape CSS curly braces (`{{` and `}}`) in Python f-strings, and use `!important` to override pre-existing inline styles.
**Action:** Use this pattern whenever styling interactive elements (like focus/hover states) in Jupyter outputs.
