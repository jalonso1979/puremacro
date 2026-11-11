
## 2024-11-11 - Properly labeling visible text buttons and segmented groups
**Learning:** Adding `aria-label` to buttons that already have populated, visible text content overrides the visible text for screen readers, breaking multilingual setups and violating WCAG 2.5.3 (Label in Name). For custom grouped controls (like segmented buttons using `role="group"`), applying `aria-label` statically or inside a paint loop can be brittle or redundant.
**Action:** Avoid `aria-label` on buttons with visible text; rely on the `title` attribute for supplementary tooltips. When grouping controls, use a visually styled `<span>` with a unique dynamically generated ID and link it to the group using `aria-labelledby`.
