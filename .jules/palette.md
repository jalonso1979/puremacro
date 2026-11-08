## 2026-11-08 - Segmented control accessibility
**Learning:** Adding `aria-label` dynamically to a `div` with `role="group"` or to buttons with visible text overrides the visible text for screen readers, which can break multilingual setups and violate WCAG 2.5.3 (Label in Name).
**Action:** Use a unique ID on a `span` label and link it to the `role="group"` container via `aria-labelledby` instead of overriding `aria-label`. For icon buttons that already have visible text or meaningful content, avoid adding an `aria-label`.
