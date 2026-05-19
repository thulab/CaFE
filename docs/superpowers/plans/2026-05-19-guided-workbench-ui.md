# Guided Workbench UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guided workbench UI for the TSBenchmark Vue frontend without changing backend contracts.

**Architecture:** Add a global stylesheet for shared shell, card, table, form, and status styles. Let `EvaluationWizardPage.vue` own the workflow layout and progress state while keeping each wizard step responsible for its existing API behavior.

**Tech Stack:** Vue 3 Composition API, TypeScript, Vite, Vitest, Vue Testing Library.

---

## File Structure

- Modify `frontend/src/main.ts` to import global CSS.
- Create `frontend/src/styles.css` for shared UI primitives and responsive layout.
- Modify `frontend/src/pages/EvaluationWizardPage.vue` for workbench shell, progress rail, and step cards.
- Modify wizard components in `frontend/src/components/wizard/` for semantic markup and status classes.
- Modify result pages/components in `frontend/src/pages/` and `frontend/src/components/results/` for consistent shell, loading states, and responsive tables.
- Modify `frontend/src/tests/e2e-smoke.test.ts` and focused component tests as needed.

### Task 1: Global Shell and Wizard Layout

**Files:**
- Create: `frontend/src/styles.css`
- Modify: `frontend/src/main.ts`
- Modify: `frontend/src/pages/EvaluationWizardPage.vue`
- Test: `frontend/src/tests/e2e-smoke.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
it('renders the guided workbench shell and model-backed run controls', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));

  render(App);

  expect(screen.getByRole('heading', { name: 'Guided benchmark workbench' })).toBeTruthy();
  expect(screen.getByText('Upload CSV')).toBeTruthy();
  expect(screen.getByLabelText('CSV file')).toBeTruthy();
  expect(await screen.findByText('Timer 3.5')).toBeTruthy();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/tests/e2e-smoke.test.ts`

Expected: FAIL because `Guided benchmark workbench` is not rendered.

- [ ] **Step 3: Implement minimal layout**

Add `import './styles.css';` in `frontend/src/main.ts`. Create a global stylesheet with shell, cards, form controls, tables, status badges, and responsive rules. Replace the wizard page template with a header, progress rail, and six step cards while preserving all existing child components.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- src/tests/e2e-smoke.test.ts`

Expected: PASS.

### Task 2: Wizard Step Polish

**Files:**
- Modify: `frontend/src/components/wizard/UploadStep.vue`
- Modify: `frontend/src/components/wizard/ColumnAndSplitStep.vue`
- Modify: `frontend/src/components/wizard/LoadShardStep.vue`
- Modify: `frontend/src/components/wizard/TrackStep.vue`
- Modify: `frontend/src/components/wizard/RunStep.vue`
- Modify: `frontend/src/components/wizard/ResultStep.vue`
- Test: existing `frontend/src/tests/*Step.test.ts`

- [ ] **Step 1: Run focused tests before editing**

Run: `cd frontend && npm test -- src/tests/UploadStep.test.ts src/tests/ColumnAndSplitStep.test.ts src/tests/LoadShardStep.test.ts src/tests/RunStep.test.ts`

Expected: PASS before markup changes.

- [ ] **Step 2: Implement semantic controls**

Add concise helper copy, field groups, `form-grid` classes, responsive table wrappers, alert/status classes, disabled button styling, and model option cards. Preserve accessible names: `CSV file`, `Next`, `Load shard`, `Create track`, and `Run`.

- [ ] **Step 3: Run focused tests after editing**

Run: `cd frontend && npm test -- src/tests/UploadStep.test.ts src/tests/ColumnAndSplitStep.test.ts src/tests/LoadShardStep.test.ts src/tests/RunStep.test.ts`

Expected: PASS.

### Task 3: Result Page Baseline

**Files:**
- Modify: `frontend/src/pages/RankingPage.vue`
- Modify: `frontend/src/pages/ReportPage.vue`
- Modify: `frontend/src/pages/SampleForecastPage.vue`
- Modify: `frontend/src/components/results/RankingTable.vue`
- Modify: `frontend/src/components/results/ReportSummary.vue`
- Modify: `frontend/src/components/results/SampleMetricTable.vue`
- Modify: `frontend/src/components/results/ForecastChart.vue`
- Test: result page tests under `frontend/src/tests/`

- [ ] **Step 1: Run focused tests before editing**

Run: `cd frontend && npm test -- src/tests/RankingPage.test.ts src/tests/ReportPage.test.ts src/tests/SampleForecastPage.test.ts`

Expected: PASS before markup changes.

- [ ] **Step 2: Implement result styling**

Add page headers, loading messages, filter row styling, responsive table wrappers, metric formatting containers, chart card styling, and failure alert styling. Keep existing text and links used by tests.

- [ ] **Step 3: Run focused tests after editing**

Run: `cd frontend && npm test -- src/tests/RankingPage.test.ts src/tests/ReportPage.test.ts src/tests/SampleForecastPage.test.ts`

Expected: PASS.

### Task 4: Full Frontend Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run full Vitest suite**

Run: `cd frontend && npm test`

Expected: all frontend tests pass.

- [ ] **Step 2: Run e2e smoke script**

Run: `cd frontend && npm run test:e2e`

Expected: e2e smoke test passes.
