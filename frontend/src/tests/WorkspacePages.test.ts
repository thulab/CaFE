import { render, screen } from '@testing-library/vue';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DatasetsPage from '../pages/DatasetsPage.vue';
import HomePage from '../pages/HomePage.vue';
import RunsPage from '../pages/RunsPage.vue';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

/**
 * Pages now render whatever the backend list endpoints return — there's no localStorage middleman.
 * We stub fetch by URL prefix so each page's parallel list calls all resolve from the same map.
 */
function stubBackend(byPath: Record<string, unknown>) {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    // Strip the /api prefix and any querystring to match against byPath keys.
    const path = url.replace(/^\/api/, '').split('?')[0];
    const body = byPath[path] ?? { items: [], total: 0, limit: 50, offset: 0 };
    return jsonResponse(body);
  });
}

describe('workspace pages', () => {
  afterEach(() => vi.restoreAllMocks());

  it('HomePage renders the empty activity state when backend has nothing', async () => {
    stubBackend({});
    render(HomePage);
    expect(screen.getByRole('heading', { name: 'Workbench overview' })).toBeTruthy();
    expect(await screen.findByText('No activity yet')).toBeTruthy();
  });

  it('DatasetsPage lists manifests and shards from the backend', async () => {
    stubBackend({
      '/dataset-manifests': {
        items: [{ dataset_manifest_id: 'manifest-9', name: 'Uploaded dataset', domain: 'energy', created_at: '2026-05-26T12:00:00Z' }],
        total: 1, limit: 200, offset: 0
      },
      '/shards': {
        items: [{ shard_id: 'shard-9', dataset_manifest_id: 'manifest-9', target_columns: ['target'], row_count: 20, created_at: '2026-05-26T12:01:00Z' }],
        total: 1, limit: 200, offset: 0
      }
    });
    render(DatasetsPage);
    expect(screen.getByRole('heading', { name: 'Datasets' })).toBeTruthy();
    expect(await screen.findByText('Uploaded dataset')).toBeTruthy();
    expect(await screen.findByText('Shard · target')).toBeTruthy();
  });

  it('RunsPage starts empty then surfaces server-returned runs', async () => {
    stubBackend({});
    const empty = render(RunsPage);
    expect(screen.getByRole('heading', { name: 'Runs' })).toBeTruthy();
    expect(await screen.findByText('No runs yet')).toBeTruthy();
    empty.unmount();
    vi.restoreAllMocks();

    stubBackend({
      '/benchmarking-runs': {
        items: [{ benchmarking_run_id: 'run-9', track_id: 't1', model_ids: ['m1', 'm2'], status: 'running', model_count: 2, task_count: 2, sample_count: 4, created_at: '2026-05-26T13:00:00Z' }],
        total: 1, limit: 200, offset: 0
      }
    });
    render(RunsPage);
    expect((await screen.findAllByText('Run · 2 models')).length).toBeGreaterThan(0);
  });
});
