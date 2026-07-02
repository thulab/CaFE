import { fireEvent, render, screen, waitFor, within } from '@testing-library/vue';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import TrackPage from '../pages/TrackPage.vue';
import TracksPage from '../pages/TracksPage.vue';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

describe('resource lifecycle UI', () => {
  beforeEach(() => setLocale('en-US'));
  afterEach(() => vi.restoreAllMocks());

  it('loads archived tracks when the workspace toggle is enabled', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/shards?limit=200') return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      if (url === '/api/tracks?include_archived=true') {
        return jsonResponse({
          items: [
            { track_id: 'track-1', name: 'Active track', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4 },
            { track_id: 'track-2', name: 'Archived track', primary_metric_id: 'mse', status: 'ready', shard_count: 1, sample_count: 4, archived_at: '2026-05-28T00:00:00Z' }
          ],
          total: 2
        });
      }
      if (url === '/api/tracks') {
        return jsonResponse({
          items: [{ track_id: 'track-1', name: 'Active track', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4 }],
          total: 1
        });
      }
      return jsonResponse({});
    });

    render(TracksPage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Active track')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Create track' })).toBeTruthy();
    expect(screen.queryByLabelText('Track name')).toBeNull();
    expect(screen.queryByText('Archived track')).toBeNull();

    await fireEvent.click(screen.getByLabelText('Show archived'));

    expect(await screen.findByText('Archived track')).toBeTruthy();
    expect(fetchSpy).toHaveBeenCalledWith('/api/tracks?include_archived=true', expect.any(Object));
  });

  it('archives and restores a track through the confirmation dialog', async () => {
    const calls: string[] = [];
    let archived = false;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push(`${init?.method ?? 'GET'} ${url}`);
      if (url === '/api/shards?limit=200') return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      if (url === '/api/tracks/track-1/deletion-impact') {
        return jsonResponse({
          resource_type: 'track',
          resource_id: 'track-1',
          archive_available: true,
          purge_available: true,
          cascade_required: true,
          affected: { tracks: 1, benchmarking_runs: 1, reports: 1 },
          warnings: []
        });
      }
      if (url === '/api/tracks/track-1/archive' && init?.method === 'POST') {
        archived = true;
        return jsonResponse({ resource_type: 'track', resource_id: 'track-1', archived_at: '2026-05-28T00:00:00Z' });
      }
      if (url === '/api/tracks/track-1/restore' && init?.method === 'POST') {
        archived = false;
        return jsonResponse({ resource_type: 'track', resource_id: 'track-1', archived_at: null });
      }
      if (url === '/api/tracks?include_archived=true') {
        return jsonResponse({
          items: [{ track_id: 'track-1', name: 'Track one', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4, archived_at: archived ? '2026-05-28T00:00:00Z' : null }],
          total: 1
        });
      }
      if (url === '/api/tracks') {
        return jsonResponse({
          items: archived ? [] : [{ track_id: 'track-1', name: 'Track one', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4 }],
          total: archived ? 0 : 1
        });
      }
      return jsonResponse({});
    });

    render(TracksPage, { global: { plugins: [i18n] } });
    expect(await screen.findByText('Track one')).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: 'Archive' }));
    const archiveDialog = await screen.findByRole('dialog', { name: 'Archive resource' });
    expect(await within(archiveDialog).findByText('1 run')).toBeTruthy();
    await fireEvent.click(within(archiveDialog).getByRole('button', { name: 'Archive resource' }));
    await waitFor(() => expect(calls).toContain('POST /api/tracks/track-1/archive'));

    await fireEvent.click(screen.getByLabelText('Show archived'));
    expect(await screen.findByText('Archived')).toBeTruthy();
    await fireEvent.click(screen.getByRole('button', { name: 'Restore' }));
    const restoreDialog = await screen.findByRole('dialog', { name: 'Restore resource' });
    await fireEvent.click(within(restoreDialog).getByRole('button', { name: 'Restore resource' }));
    await waitFor(() => expect(calls).toContain('POST /api/tracks/track-1/restore'));
  });

  it('shows purge impact before permanent deletion', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push(`${init?.method ?? 'GET'} ${url}`);
      if (url === '/api/shards?limit=200') return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      if (url === '/api/tracks/track-1/deletion-impact') {
        return jsonResponse({
          resource_type: 'track',
          resource_id: 'track-1',
          archive_available: true,
          purge_available: true,
          cascade_required: true,
          affected: { tracks: 1, benchmarking_runs: 1, reports: 1, metric_results: 4 },
          warnings: []
        });
      }
      if (url === '/api/tracks/track-1?cascade=true' && init?.method === 'DELETE') {
        return jsonResponse({ ok: true });
      }
      if (url === '/api/tracks') {
        return jsonResponse({
          items: [{ track_id: 'track-1', name: 'Track one', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4 }],
          total: 1
        });
      }
      return jsonResponse({});
    });

    render(TracksPage, { global: { plugins: [i18n] } });
    expect(await screen.findByText('Track one')).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: 'Permanent delete' }));
    const dialog = await screen.findByRole('dialog', { name: 'Permanent delete' });
    expect(await within(dialog).findByText('1 run')).toBeTruthy();
    expect(await within(dialog).findByText('4 metric results')).toBeTruthy();

    await fireEvent.click(within(dialog).getByRole('button', { name: 'Permanent delete forever' }));

    await waitFor(() => expect(calls).toContain('DELETE /api/tracks/track-1?cascade=true'));
  });

  it('shows archived track detail without start-run controls', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/tracks/track-1') {
        return jsonResponse({ track_id: 'track-1', name: 'Archived track', primary_metric_id: 'mase', status: 'ready', shard_count: 1, sample_count: 4, archived_at: '2026-05-28T00:00:00Z' });
      }
      if (url === '/api/benchmarking-runs?limit=200&track_id=track-1') return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      if (url === '/api/tracks/track-1/ranking?metric=mase&policy=latest_valid_result') return jsonResponse({ items: [] });
      return jsonResponse({});
    });

    render(TrackPage, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Archived track')).toBeTruthy();
    expect((await screen.findAllByText('Archived')).length).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: 'Start run' })).toBeNull();
  });

  it('shows the test case sets that compose a track', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/tracks/track-1') {
        return jsonResponse({ track_id: 'track-1', name: 'Energy track', primary_metric_id: 'mase', status: 'ready', shard_ids: ['shard-1'], shard_count: 1, sample_count: 20 });
      }
      if (url === '/api/shards/shard-1') {
        return jsonResponse({
          shard_id: 'shard-1',
          name: 'Energy validation cases',
          dataset_name: 'Energy dataset',
          dataset_manifest_id: 'manifest-1',
          source_uri: '/tmp/energy.csv',
          status: 'ready',
          row_count: 96,
          sample_count: 20,
          target_columns: ['load'],
          target_dim: 1,
          context_length: 60,
          horizon: 16,
          stride: 16
        });
      }
      if (url === '/api/models') return jsonResponse({ items: [] });
      if (url === '/api/benchmarking-runs?limit=200&track_id=track-1') return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      if (url === '/api/tracks/track-1/ranking?metric=mase&policy=latest_valid_result') return jsonResponse({ items: [] });
      return jsonResponse({});
    });

    render(TrackPage, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Energy track')).toBeTruthy();
    const link = await screen.findByRole('link', { name: 'Energy validation cases' });
    expect(link.getAttribute('href')).toBe('#/shards/shard-1');
    expect(screen.getByText('Energy dataset')).toBeTruthy();
    expect(screen.getByText('Context 60 · Horizon 16 · Stride 16')).toBeTruthy();
    expect(screen.queryByText('Covariates')).toBeNull();
  });

  it('hides track run history until the start-run card toggle is opened', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/tracks/track-1') {
        return jsonResponse({ track_id: 'track-1', name: 'Energy track', primary_metric_id: 'mse', status: 'ready', shard_ids: [], shard_count: 0, sample_count: 0 });
      }
      if (url === '/api/models') return jsonResponse({ items: [] });
      if (url === '/api/benchmarking-runs?limit=200&track_id=track-1') {
        return jsonResponse({
          items: [{
            benchmarking_run_id: 'run-1',
            track_id: 'track-1',
            model_ids: ['m1'],
            status: 'succeeded',
            activity_status: 'succeeded',
            model_count: 1,
            task_count: 1,
            sample_count: 4,
            report_id: 'rep-1',
            created_at: '2026-07-02T08:00:00Z'
          }],
          total: 1,
          limit: 200,
          offset: 0
        });
      }
      if (url === '/api/tracks/track-1/ranking?metric=mse&policy=latest_valid_result') return jsonResponse({ items: [] });
      if (url === '/api/tracks/track-1/results?sample_link_limit=10&sample_link_offset=0&sample_link_sort=sample_index') {
        return jsonResponse({
          track_id: 'track-1',
          metric: 'mse',
          model_statuses: [],
          model_metrics: [],
          capability_blocks: [],
          capability_metrics: [],
          sample_forecast_links: [],
          sample_forecast_links_total: 0,
          sample_forecast_links_limit: 10,
          sample_forecast_links_offset: 0,
          sample_forecast_links_sort: 'sample_index'
        });
      }
      return jsonResponse({});
    });

    render(TrackPage, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Energy track')).toBeTruthy();
    expect(screen.queryByText('Run · 1 model')).toBeNull();

    await fireEvent.click(await screen.findByRole('button', { name: 'Show run history' }));

    expect(await screen.findByText('Run · 1 model')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Hide run history' })).toBeTruthy();
  });
});
