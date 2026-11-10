## 2024-03-24 - Segmented Control Accessibility
**Learning:** When building custom grouped UI controls (like segmented buttons with `role="group"`), do not use a wrapping `<label>` for the group title, as `<label>` must only associate with a single valid form input. Instead, use a visually styled `<span>` with a unique ID and associate it with the group using `aria-labelledby`.
**Action:** Always check the element type of the label and add an ID to the label element and `aria-labelledby` to the `group` container.
