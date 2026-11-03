# ShopSmart recording and acceptance test

Allow about 10-15 minutes. Use full mode for this recording, not a saved-output
preview. This is a functional test; it does not replace formal model evaluation.
For installation and data-path configuration, follow [UI_README.md](UI_README.md).

## Before recording

1. Start from the repository root:

   ```powershell
   .\.venv\Scripts\python.exe launch_ui.py --mode full --port 8503
   ```

2. Keep the terminal running. Open http://127.0.0.1:8503/ in a fresh Chrome
   session, such as an Incognito window, so the comparison starts empty.
3. Wait until the homepage is ready. Maximize the browser and use 100% zoom.
   Keep Results per search at 3. Record only the website; do not expose `.env`,
   API keys or private notifications.
4. Do not submit the recording queries in advance in this session. Reusing
   existing results can demonstrate the session cache rather than a new call.

## Recording sequence

| Step | Action | Expected result / check |
|---|---|---|
| 1. Homepage | Show Discover for a few seconds. | Clear English text, aligned controls and three shopping scenarios. |
| 2. Team | Scroll to the footer, click Team, then close the dialog. | Group 26 / CCDS and all five names; closing returns to the same page. |
| 3. Sidebar | Collapse and reopen twice. Switch to Night and repeat; return to Day. | The top-left expand button stays visible and works in both themes. |
| 4. Empty input | Submit the blank search form. | Helpful input prompt, no exception and no search. |
| 5. Live query A | Submit Query A below once. Preserve the real loading period. | Search/generation indicators, then recommendations and up to three products. Record a delay over a minute rather than repeatedly clicking. |
| 6. Citation check | Expand Inspect cited sources. Compare two concrete claims with their cited passages. | IDs match; each passage actually supports its claim and refers to the correct product. |
| 7. Product details | Open Evidence, Requirement evidence, Score breakdown and Why not? on one product. | Readable passages, source IDs, coverage states, score explanation and available risks. Full scores such as 26.5/100 must not be clipped. |
| 8. Compare | Add three products, then open Compare. | The same products, ASINs, scores and evidence appear; added buttons prevent duplicates. |
| 9. State retention | Switch Night/Day, return to Discover, open/close Team. | Selection and query remain; the existing answer does not re-enter generation. |
| 10. Live query B | Click New search, confirm the input is empty, then submit Query B. | Brief, answer and evidence update for work calls. The previous comparison remains. |
| 11. Capacity | Try adding an unselected product while three are saved. | A fourth product cannot be added. If all returned products are already saved, use another query. |
| 12. Remove / replace | Remove one in Compare, return to Discover and add a new product. | Two remain after removal; adding a replacement restores three without duplicates. |
| 13. Export | Remove one again, export the remaining two and manually open the downloaded JSON. | shopsmart-comparison.json contains exactly those two products, with matching ASINs, queries and evidence. |
| 14. Smaller window | Narrow the window; collapse/reopen navigation and inspect the layout. | Buttons, scores and dialogs remain usable. Restore the window before finishing. |

### Query A: earbuds for long flights

```text
I need wireless earbuds for long flights, with long battery life and a comfortable fit.
```

Check battery life and comfort separately. A reviewer's desire for comfort is
not evidence that a product is comfortable. A prior live answer overstated this;
retain any recurrence as a generation-quality issue for the module owner.

### Query B: headphones for work calls

```text
I need Bluetooth headphones for work calls, with a clear microphone and a stable connection.
```

Check microphone and connectivity evidence. Products may overlap across queries;
that alone does not mean the answer is stale.

## Optional additional checks

- New search -> Long-haul listening -> Explore options: the scenario should fill
  its query and display results. This is an additional live query.
- Change Results per search and confirm the results update. This changes the
  search/answer cache key and can trigger another paid request.
- Try an unrelated query. Dense search normally returns nearest neighbours even
  for unrelated inputs; returned products do not establish relevance. Record
  inappropriate recommendations for the retrieval/generation owners.
- Test Find alternatives only if that button appears in a Why not? finding.

## Acceptance and issue reporting

Mark each step Pass, Fail or Not tested. For a failure, record the exact query,
action, expected result, actual result and video timestamp. Keep UI problems
(layout, buttons or lost state) separate from retrieval, sentiment and unsupported
answer claims.

Browser export download has not yet been confirmed in the agent's UI tests;
step 13 is an explicit manual acceptance check.

The page can fall back to review excerpts when the API is unavailable. Confirm
live model calls using your own API usage records or a suitable backend call
record; do not infer success from the absence of a visible error. The main
sequence plans two new query submissions. Optional queries can add API usage.

Do not describe the score as a purchase-success probability, evidence availability
as requirement satisfaction, or a citation ID as proof of faithfulness. The
system searches the full knowledge base and sends selected evidence to the
model; it does not send all reviews in one prompt.

If restarting is necessary, press Ctrl+C in the original server terminal, then
run the startup command again. Restart after changing `.env` as well.
