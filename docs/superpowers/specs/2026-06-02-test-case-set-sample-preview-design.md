# Test Case Set Sample Preview Design

## Context

The evaluation wizard can already generate reusable test case sets, and the workbench has a test case set detail page. The current flow does not make the generated set easy to inspect immediately, the selectable test case set list does not link names to detail pages, and the test case set detail page renders its sample list as a full table without per-sample curve inspection.

This design keeps test case set inspection independent from benchmark runs and reports. The first version shows only the real sample window: history/context values plus future/ground-truth values. Model forecasts and metrics remain part of report sample pages.

## Goals

- After generating a test case set in the wizard, users can immediately open the generated set for inspection without losing the guided flow.
- In the test case set selection step, users can open any listed set by clicking its name.
- The test case set detail page paginates sample rows instead of rendering every sample at once.
- Each sample row can open a curve view for the real sample window.
- The sample curve view has a clear return path to the test case set detail page.

## Non-Goals

- Do not show model forecasts, per-model metrics, or report-specific state in the test case set sample preview.
- Do not couple test case set pages to benchmarking runs.
- Do not add inline expandable charts inside the sample table in this iteration.

## UX Design

### Wizard Generated Set Access

When `wizardState.shardId` exists in the test case set selection step, show a compact success block above the selectable list. It displays the generated test case set name if available, otherwise the fallback test case set title. The block includes a secondary action linking to `#/shards/{shardId}`.

The wizard remains on the selection step. Opening the detail page is optional and does not change the wizard state; users can return through browser history or the existing resume artifact links.

### Selectable Test Case Set List

Rows keep the existing checkbox behavior for selecting sets. The set title becomes a text link to `#/shards/{shard_id}`. The checkbox remains the control for selection, so clicking the title means inspection and clicking the checkbox means selection.

### Test Case Set Detail Samples

The sample table on the test case set detail page uses backend pagination from `GET /shards/{shardId}/samples?limit=10&offset=n`. It shows:

- Name: `Window #n` when `sample_index` exists, otherwise a short sample ID fallback.
- Context rows: `context_start -> context_end`.
- Forecast rows: `horizon_start -> horizon_end`.
- Action: `Open curve`, linking to `#/shards/{shardId}/samples/{sampleId}`.

The pager follows the report sample list pattern: range caption, previous/next buttons, and a page status label. Empty and loading states use the existing `StateBlock` and table styles.

### Sample Window Preview Page

Add a dedicated route for test case sample preview:

`#/shards/{shardId}/samples/{sampleId}`

The page calls `GET /samples/{sampleId}/preview` and renders a chart of the real sequence only:

- History/context line from `target_history`.
- Future/ground-truth line from `target_future`.
- Boundary marker between history and future, consistent with the report forecast chart.
- Header action: `Back to test case set`, linking to `#/shards/{shardId}`.

For the MVP, the page can reuse the visual conventions from `ForecastChart` but should not pretend there are model forecasts. If reuse becomes awkward, create a small `SampleWindowChart` component that accepts sample preview data and labels history/future explicitly.

## Data Flow

- `ShardPage` loads the shard metadata and the current sample page separately.
- `getShardSamples(shardId, { limit, offset })` becomes parameterized on the frontend while preserving existing defaults.
- `SampleWindowPreviewPage` loads the full sample content via `getSamplePreview(sampleId)`.
- No backend contract change is required for pagination because `/shards/{shardId}/samples` already accepts `limit` and `offset`.
- The existing `/samples/{sampleId}/preview` endpoint provides the values needed for the real-sequence chart.

## Error Handling

- If sample-page loading fails on the test case set detail page, show a localized alert in the samples section and keep shard metadata visible when possible.
- If sample preview loading fails, show the existing `StateBlock` error and retry control.
- If a sample has missing ranges or values, show `N/A` in table cells and render the chart empty state rather than a broken SVG.

## Testing

- Add or update frontend tests for:
  - Generated-set success block and link on the test case selection step.
  - Test case set list title linking to `#/shards/{id}` while checkbox selection still works.
  - Shard detail sample pagination calls with `limit` and `offset`.
  - Shard detail sample action link to the sample preview route.
  - Sample window preview page rendering history and future curves with a back link.
- Run frontend unit tests and Vue type checking before implementation handoff.

## Open Decisions

- First version uses page navigation for sample curves instead of inline table expansion.
- First version shows real sample windows only; report pages remain responsible for forecast comparisons.
