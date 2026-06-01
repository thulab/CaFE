# Test Case Set Sample Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated and existing test case sets inspectable from the wizard, paginate test case samples, and add a real-sequence sample curve page.

**Architecture:** Keep test case sample inspection independent from runs and reports. Reuse existing hash routing, API client patterns, `StateBlock`, table styles, and SVG chart conventions while adding a focused `SampleWindowChart` for real history/future values only.

**Tech Stack:** Vue 3 Composition API, TypeScript, Vitest with Vue Testing Library, FastAPI-backed REST endpoints already exposed by the backend.

---

## File Structure

- Modify `frontend/src/api/datasets.ts`: parameterize `getShardSamples`.
- Modify `frontend/src/api/types.ts`: add `SamplePreviewDTO`.
- Create `frontend/src/api/samples.ts`: client for `/samples/{sampleId}/preview`.
- Modify `frontend/src/components/ui/SelectablePagedList.vue`: support optional title links per item.
- Modify `frontend/src/components/wizard/TestCaseSetStep.vue`: generated-set success block and shard links in selectable rows.
- Modify `frontend/src/pages/ShardPage.vue`: sample pagination and sample curve links.
- Create `frontend/src/components/results/SampleWindowChart.vue`: real history/future sample chart.
- Create `frontend/src/pages/SampleWindowPreviewPage.vue`: sample preview route page with back link.
- Modify `frontend/src/App.vue`: route `#/shards/{shardId}/samples/{sampleId}`.
- Modify `frontend/src/i18n/locales/en-US.ts` and `frontend/src/i18n/locales/zh-CN.ts`: bilingual labels.
- Update tests in `frontend/src/tests/TestCaseSetStep.test.ts`, `frontend/src/tests/AppRoutes.test.ts`, and add `frontend/src/tests/SampleWindowPreviewPage.test.ts`.

---

### Task 1: API Contracts

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/datasets.ts`
- Create: `frontend/src/api/samples.ts`
- Test: `frontend/src/tests/api-client.test.ts`

- [ ] **Step 1: Write the failing API client test**

Add assertions to `frontend/src/tests/api-client.test.ts`:

```ts
import { getShardSamples } from '../api/datasets';
import { getSamplePreview } from '../api/samples';

it('builds paginated shard sample and sample preview requests', async () => {
  const fetchSpy = vi.spyOn(globalThis, 'fetch')
    .mockResolvedValueOnce(jsonResponse({ items: [], total: 0, limit: 10, offset: 20 }))
    .mockResolvedValueOnce(jsonResponse({ sample_id: 'sample-1', target_history: [[1]], target_future: [[2]] }));

  await getShardSamples('shard-1', { limit: 10, offset: 20 });
  await getSamplePreview('sample-1');

  expect(String(fetchSpy.mock.calls[0]![0])).toBe('/api/shards/shard-1/samples?limit=10&offset=20');
  expect(String(fetchSpy.mock.calls[1]![0])).toBe('/api/samples/sample-1/preview');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run src/tests/api-client.test.ts`

Expected: fails because `getSamplePreview` does not exist and `getShardSamples` does not accept pagination parameters.

- [ ] **Step 3: Implement API clients**

Add `SamplePreviewDTO` to `frontend/src/api/types.ts`:

```ts
export interface SamplePreviewDTO extends SampleWindowMeta {
  sample_id: string;
  shard_id?: string;
  target_column_names?: string[];
  history_timestamps?: string[];
  future_timestamps?: string[];
  target_history: number[][];
  target_future: number[][];
}
```

Change `getShardSamples` in `frontend/src/api/datasets.ts`:

```ts
export function getShardSamples(shardId: string, params: Pick<ListParams, 'limit' | 'offset'> = {}): Promise<ShardSamplesDTO> {
  return apiRequest<ShardSamplesDTO>(`/shards/${shardId}/samples${buildListQuery(params)}`);
}
```

Create `frontend/src/api/samples.ts`:

```ts
import { apiRequest } from './client';
import type { SamplePreviewDTO } from './types';

export function getSamplePreview(sampleId: string): Promise<SamplePreviewDTO> {
  return apiRequest<SamplePreviewDTO>(`/samples/${encodeURIComponent(sampleId)}/preview`);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run src/tests/api-client.test.ts`

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add frontend/src/api/types.ts frontend/src/api/datasets.ts frontend/src/api/samples.ts frontend/src/tests/api-client.test.ts
git commit -m "add sample preview api client"
```

---

### Task 2: Wizard Test Case Set Links

**Files:**
- Modify: `frontend/src/components/ui/SelectablePagedList.vue`
- Modify: `frontend/src/components/wizard/TestCaseSetStep.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Test: `frontend/src/tests/TestCaseSetStep.test.ts`

- [ ] **Step 1: Write failing wizard tests**

Add test coverage to `frontend/src/tests/TestCaseSetStep.test.ts`:

```ts
it('links the generated and listed test case sets to their detail pages', async () => {
  wizardState.shardId = 'shard-generated';
  wizardState.shardName = 'Generated validation cases';
  wizardState.selectedShardIds = ['shard-generated'];
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    if (url.startsWith('/api/shards')) {
      return jsonResponse({
        items: [{
          shard_id: 'shard-generated',
          name: 'Generated validation cases',
          dataset_name: 'Energy',
          source_uri: '/tmp/energy.csv',
          target_columns: ['load'],
          context_length: 60,
          horizon: 16,
          stride: 16,
          sample_count: 20,
          row_count: 96,
          status: 'ready'
        }],
        total: 1,
        limit: 10,
        offset: 0
      });
    }
    return jsonResponse({});
  });

  render(TestCaseSetStep, { global: { plugins: [i18n] } });

  expect(await screen.findByText('Generated validation cases')).toBeTruthy();
  expect(screen.getByRole('link', { name: 'Open generated test case set' }).getAttribute('href')).toBe('#/shards/shard-generated');
  expect(screen.getByRole('link', { name: 'Generated validation cases' }).getAttribute('href')).toBe('#/shards/shard-generated');
  expect((screen.getByLabelText('Select Generated validation cases') as HTMLInputElement).checked).toBe(true);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run src/tests/TestCaseSetStep.test.ts`

Expected: fails because list titles are not links and the generated-set success block does not exist.

- [ ] **Step 3: Implement title links and success block**

Extend `SelectablePagedListItem` in `frontend/src/components/ui/SelectablePagedList.vue` and `frontend/src/components/wizard/TestCaseSetStep.vue` with `href?: string`. Render item titles as `<a class="text-link">` when `href` exists.

In `TestCaseSetStep.vue`, add a note block above the list when `wizardState.shardId` exists:

```vue
<div v-if="generatedShardLink" class="note-success" style="justify-content:space-between;gap:12px;flex-wrap:wrap">
  <span><Icon name="checkCircle" :size="16" />{{ t('wizard.testCaseSetStep.generatedReady', { name: generatedShardName }) }}</span>
  <a class="btn secondary sm" :href="generatedShardLink" :aria-label="t('wizard.testCaseSetStep.openGeneratedSet')">
    <Icon name="layers" :size="15" /> {{ t('wizard.testCaseSetStep.openGeneratedSet') }}
  </a>
</div>
```

Add computed values:

```ts
const generatedShardName = computed(() => wizardState.shardName || wizardState.shardId);
const generatedShardLink = computed(() => wizardState.shardId ? `#/shards/${wizardState.shardId}` : '');
```

Set each list item `href: \`#/shards/${shard.shard_id}\``.

Add locale keys:

```ts
generatedReady: 'Generated test case set: {name}',
openGeneratedSet: 'Open generated test case set',
```

Chinese:

```ts
generatedReady: '已生成测试用例集：{name}',
openGeneratedSet: '查看生成的测试用例集',
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run src/tests/TestCaseSetStep.test.ts`

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add frontend/src/components/ui/SelectablePagedList.vue frontend/src/components/wizard/TestCaseSetStep.vue frontend/src/i18n/locales/en-US.ts frontend/src/i18n/locales/zh-CN.ts frontend/src/tests/TestCaseSetStep.test.ts
git commit -m "link test case sets from wizard"
```

---

### Task 3: Paginated Test Case Sample List

**Files:**
- Modify: `frontend/src/pages/ShardPage.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Test: `frontend/src/tests/AppRoutes.test.ts`

- [ ] **Step 1: Write failing shard detail tests**

Update the shard route test in `frontend/src/tests/AppRoutes.test.ts` so `/api/shards/shard-1/samples?limit=10&offset=0` returns `total: 12`, and assert:

```ts
expect(await screen.findByText('Showing 1-10 of 12 samples')).toBeTruthy();
expect(screen.getByRole('link', { name: /Open curve/ }).getAttribute('href')).toBe('#/shards/shard-1/samples/sample-1');
await fireEvent.click(screen.getByRole('button', { name: 'Next page' }));
await waitFor(() => expect(requests.some((url) => url === '/api/shards/shard-1/samples?limit=10&offset=10')).toBe(true));
expect(await screen.findByText('Showing 11-12 of 12 samples')).toBeTruthy();
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run src/tests/AppRoutes.test.ts`

Expected: fails because `ShardPage` loads all samples once and does not render curve links.

- [ ] **Step 3: Implement pagination**

In `ShardPage.vue`, split shard loading from sample-page loading. Add:

```ts
const SAMPLE_PAGE_SIZE = 10;
const sampleOffset = ref(0);
const sampleLoading = ref(false);
const sampleError = ref('');
const sampleTotal = computed(() => samples.value?.total ?? samples.value?.items.length ?? 0);
const samplePage = computed(() => Math.floor(sampleOffset.value / SAMPLE_PAGE_SIZE) + 1);
const samplePageCount = computed(() => Math.max(1, Math.ceil(sampleTotal.value / SAMPLE_PAGE_SIZE)));
```

Call `getShardSamples(props.shardId, { limit: SAMPLE_PAGE_SIZE, offset: sampleOffset.value })`. Render `caption` as `t('shard.samplePageRange', { start, end, total })`, add previous/next buttons, and link each row to `#/shards/${props.shardId}/samples/${s.sample_id}`.

Add locale keys:

```ts
openCurve: 'Open curve',
samplePageRange: 'Showing {start}-{end} of {total} samples',
samplePagination: 'Test case sample pagination',
```

Chinese:

```ts
openCurve: '查看曲线',
samplePageRange: '显示第 {start}-{end} 个，共 {total} 个样本',
samplePagination: '测试用例样本分页',
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run src/tests/AppRoutes.test.ts`

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add frontend/src/pages/ShardPage.vue frontend/src/i18n/locales/en-US.ts frontend/src/i18n/locales/zh-CN.ts frontend/src/tests/AppRoutes.test.ts
git commit -m "paginate test case samples"
```

---

### Task 4: Sample Window Preview Page

**Files:**
- Create: `frontend/src/components/results/SampleWindowChart.vue`
- Create: `frontend/src/pages/SampleWindowPreviewPage.vue`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Test: `frontend/src/tests/SampleWindowPreviewPage.test.ts`
- Test: `frontend/src/tests/AppRoutes.test.ts`

- [ ] **Step 1: Write failing preview page tests**

Create `frontend/src/tests/SampleWindowPreviewPage.test.ts`:

```ts
import { render, screen } from '@testing-library/vue';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import SampleWindowPreviewPage from '../pages/SampleWindowPreviewPage.vue';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

describe('SampleWindowPreviewPage', () => {
  afterEach(() => vi.restoreAllMocks());

  it('renders a real sample window chart with a back link', async () => {
    setLocale('en-US');
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      sample_id: 'sample-1',
      shard_id: 'shard-1',
      sample_index: 0,
      history_timestamps: ['2026-01-01T00:00:00', '2026-01-01T01:00:00'],
      future_timestamps: ['2026-01-01T02:00:00'],
      target_column_names: ['load'],
      target_history: [[1], [2]],
      target_future: [[3]]
    }));

    render(SampleWindowPreviewPage, { props: { shardId: 'shard-1', sampleId: 'sample-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByRole('img', { name: 'Sample window chart with 2 history steps and 1 future step.' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Back to test case set' }).getAttribute('href')).toBe('#/shards/shard-1');
    expect(screen.getByText('Window #1')).toBeTruthy();
  });
});
```

Add an App route assertion:

```ts
window.location.hash = '#/shards/shard-1/samples/sample-1';
render(App, { global: { plugins: [i18n] } });
expect(await screen.findByRole('heading', { name: 'Sample window preview' })).toBeTruthy();
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run src/tests/SampleWindowPreviewPage.test.ts src/tests/AppRoutes.test.ts`

Expected: fails because the page, chart, and route do not exist.

- [ ] **Step 3: Implement page, chart, and route**

Create `SampleWindowChart.vue` that accepts `{ sample: SamplePreviewDTO }`, flattens the first target dimension from `target_history` and `target_future`, and draws an SVG polyline for history plus another for future. Give the SVG `role="img"` and localized aria label.

Create `SampleWindowPreviewPage.vue` with props `{ shardId: string; sampleId: string }`, `useAsyncData(() => getSamplePreview(sampleId))`, a header back link to `#/shards/{shardId}`, a `StateBlock`, and `SampleWindowChart`.

Add App route before the generic `shards/:id` route:

```ts
if (parts[0] === 'shards' && parts[2] === 'samples' && id && parts[3]) {
  return {
    component: SampleWindowPreviewPage,
    props: { shardId: id, sampleId: parts[3] },
    navKey: 'datasets',
    tier: 'authed',
    crumbs: [HOME_CRUMB.value, { label: t('nav.datasets'), href: '#/datasets' }, { label: t('artifacts.shard'), href: `#/shards/${id}` }, { label: shortId(parts[3]) }]
  };
}
```

Add locale keys under `sampleWindow`:

```ts
eyebrow: 'Test case sample',
title: 'Sample window preview',
subtitle: 'Inspect the real history and future values used by this test case.',
backToShard: 'Back to test case set',
chartTitle: 'Real sample window',
history: 'History',
future: 'Future',
emptyChart: 'No sample values available.',
chartAria: 'Sample window chart with {history} and {future}.',
historyStepsOne: '1 history step',
historyStepsOther: '{count} history steps',
futureStepsOne: '1 future step',
futureStepsOther: '{count} future steps',
windowLabel: 'Window #{index}',
```

Add matching Chinese translations.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run src/tests/SampleWindowPreviewPage.test.ts src/tests/AppRoutes.test.ts`

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add frontend/src/components/results/SampleWindowChart.vue frontend/src/pages/SampleWindowPreviewPage.vue frontend/src/App.vue frontend/src/i18n/locales/en-US.ts frontend/src/i18n/locales/zh-CN.ts frontend/src/tests/SampleWindowPreviewPage.test.ts frontend/src/tests/AppRoutes.test.ts
git commit -m "add sample window preview page"
```

---

### Task 5: Final Verification

**Files:**
- All files changed above.

- [ ] **Step 1: Run focused frontend tests**

Run:

```bash
cd frontend && npm test -- --run src/tests/api-client.test.ts src/tests/TestCaseSetStep.test.ts src/tests/AppRoutes.test.ts src/tests/SampleWindowPreviewPage.test.ts
```

Expected: pass.

- [ ] **Step 2: Run full frontend test suite**

Run: `cd frontend && npm test`

Expected: pass.

- [ ] **Step 3: Run Vue type checking**

Run: `cd frontend && npx vue-tsc --noEmit`

Expected: exit code 0.

- [ ] **Step 4: Run diff hygiene check**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 5: Inspect changed files and status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only intended committed changes remain, or a final verification/doc commit is prepared if needed.
