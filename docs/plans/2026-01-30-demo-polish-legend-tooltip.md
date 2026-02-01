 # Demo polish: legend + label tooltip
 
 ## Goal
 Improve the tree demo's readability without changing backend behavior:
 - Add a compact legend for node status colors.
 - Truncate long labels with a tooltip showing the full label.
 
 ## Scope
 - Frontend only (`frontend/src/components/TreeViz.tsx`, `frontend/src/App.css`).
 - Small test setup for UI coverage (Vitest + React Testing Library).
 
 ## Non-goals
 - No backend changes.
 - No changes to data schema or streaming protocol.
 - No new layout or component structure outside the tree area.
 
 ## Approach
 - Legend overlay in the tree container: small dot + label for Accepted/Rejected/Pending.
 - Label truncation: fixed character limit with ASCII ellipsis `...`.
 - Tooltip: SVG `<title>` only when label is truncated.
 - Pill width based on displayed (possibly truncated) label.
 
 ## Tests (TDD)
 - Legend renders all three labels.
 - Long labels are truncated and include a `<title>` with the full label.
 
 ## Implementation steps
 1. Add Vitest + RTL setup and test helpers (ResizeObserver + clientWidth stubs).
 2. Write failing tests for legend and label truncation.
 3. Implement legend markup + CSS.
 4. Implement truncation + tooltip logic in TreeViz.
*** End Patch}/>
