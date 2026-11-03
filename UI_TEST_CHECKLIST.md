# UI acceptance checklist

Start full mode using [UI_README.md](UI_README.md), then test at http://127.0.0.1:8503. The full query and UI checks passed during integration. The 8501 address is a separate saved-output preview; do not use it to judge live retrieval. New live searches may take approximately 20-30 seconds; first page preparation also takes time. For an ordered recording workflow, see [VIDEO_TEST_GUIDE.md](VIDEO_TEST_GUIDE.md).

To restart full mode, run `.\.venv\Scripts\python.exe launch_ui.py --mode full --port 8503` from the project folder. Keep that terminal open. No database copying or rebuilding is required for existing compatible data.

| Check | Action | Expected result |
|---|---|---|
| Initial view | Open Discover | English only; readable text and aligned buttons; no developer menu or report metrics |
| Appearance | Switch Day / Night | Text, input, buttons, cards and evidence remain readable |
| Sidebar | Collapse and reopen twice in both themes | Top-left expand button stays visible and clickable; navigation opens again without refreshing |
| Team | Click Team in the footer, then close the dialog | CCDS and five correct names; closing preserves the current page |
| Example search | Click Long-haul listening / Explore options | Query fills in; product options and cited evidence appear |
| Evidence | Open Inspect cited sources, then product Evidence | Every citation has a corresponding passage; source IDs are visible |
| Requirements | Open Requirement evidence | Available, limited or missing evidence shown; missing evidence is not phrased as confirmed failure |
| Risks | Open Why not? | Review findings and concerns readable; any quotes have source IDs |
| Comparison | Add three products; open Compare | Same selected products, scores, coverage and evidence appear side by side |
| Comparison limit | Try adding a fourth product | No fourth product is added; limit is communicated |
| Remove / export | Remove one; download Export comparison | Product disappears; JSON contains remaining products and their evidence |
| New search | Click New search | Returns to Discover with empty input; existing comparison remains |
| Empty input | Submit a blank query | Helpful prompt; no error trace |
| No match | Search zzzznomatch123 in saved mode | Clear no-result message |
| Smaller screen | Narrow the browser; collapse navigation | Content remains usable without clipped buttons or overlapping text |

## Live queries after full mode is working

Try these exact inputs and record the products, evidence and elapsed time:

1. wireless earbuds with long battery life for flights
2. Bluetooth headphones with a clear microphone for work calls
3. portable speaker with strong bass
4. lightweight laptop for travel
5. noise cancelling headphones comfortable with glasses

Check that editing a query updates the displayed brief and results, without showing an old answer. Switching appearance and adding to comparison should not trigger a fresh answer-generation call. Relevance and sentiment accuracy should be reviewed with the relevant module owners; UI correctness alone does not validate those models.

## Recording issues

For each issue capture: mode/address, exact query, action, expected result, actual result and a screenshot. Keep UI issues separate from data/model issues, such as an irrelevant retrieved product or an incorrect sentiment label.

## Current environment

Full-model startup, five live queries and the full-data UI smoke flow passed during integration. A subsequent real API-backed browser query confirmed full retrieval and generated prose with citations. Recent UI changes were checked with saved-output previews. Export download remains a manual acceptance check. Live retrieval, sentiment and PCS work without an API key; generated prose requires OPENAI_API_KEY. API failure may fall back to excerpts. Keep keys out of screenshots and source files. A comfort claim previously exceeded its supporting evidence and remains a generation-quality issue for review.
