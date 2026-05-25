import { render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.vue';
import EvaluationWizardPage from '../pages/EvaluationWizardPage.vue';
import { resetWizard, wizardState } from '../stores/wizard';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

function mockFetch() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
    const url = String(input);
    if (url === '/api/models') {
      return Promise.resolve(jsonResponse({ items: [{ model_id: 'model-1', name: 'Timer 3.5' }] }));
    }
    if (url === '/api/reports/rep-1') {
      return Promise.resolve(jsonResponse({ report_id: 'rep-1', model_metrics: [], task_summaries: [], sample_forecast_links: [] }));
    }
    if (url === '/api/dataset-manifests/manifest-1') {
      return Promise.resolve(jsonResponse({
        dataset_manifest_id: 'manifest-1',
        name: 'Uploaded dataset',
        domain: 'general',
        source_uri: '/tmp/hourly.csv',
        file_format: 'csv',
        time_column: 'time',
        value_columns: ['target'],
        status: 'ready_to_load'
      }));
    }
    if (url === '/api/dataset-load-jobs/load-1') {
      return Promise.resolve(jsonResponse({
        load_job_id: 'load-1',
        dataset_manifest_id: 'manifest-1',
        status: 'succeeded',
        split_config: { context_length: 6, horizon: 3, stride: 3 },
        output_shard_id: 'shard-1'
      }));
    }
    if (url === '/api/shards/shard-1') {
      return Promise.resolve(jsonResponse({
        shard_id: 'shard-1',
        dataset_manifest_id: 'manifest-1',
        load_job_id: 'load-1',
        source_uri: '/tmp/hourly.csv',
        status: 'ready',
        row_count: 20,
        sample_count: 4,
        target_columns: ['target'],
        context_length: 6,
        horizon: 3,
        stride: 3
      }));
    }
    if (url === '/api/shards/shard-1/samples') {
      return Promise.resolve(jsonResponse({
        items: [{ sample_id: 'sample-1', sample_index: 0, context_start: 0, context_end: 6, horizon_start: 6, horizon_end: 9 }],
        total: 1,
        limit: 20,
        offset: 0
      }));
    }
    if (url === '/api/benchmarking-runs/run-1/progress') {
      return Promise.resolve(jsonResponse({
        benchmarking_run_id: 'run-1',
        status: 'succeeded',
        progress: { total_models: 1, completed_models: 1, total_tasks: 1, completed_tasks: 1, total_samples: 4, completed_samples: 4, failed_samples: 0 },
        units: [{ unit_id: 'unit-1', model_id: 'model-1', model_name: 'Timer 3.5', status: 'succeeded', task_count: 1, completed_task_count: 1, metrics: {} }],
        tasks: [{ task_id: 'task-1', unit_id: 'unit-1', model_id: 'model-1', capability_block_id: 'block-1', capability_block_name: 'Real data', status: 'succeeded', shard_count: 1, sample_count: 4, completed_sample_count: 4, metrics: {} }],
        recent_events: [{ created_at: '2026-01-01T00:00:00Z', message: 'run succeeded', event_type: 'status_changed', level: 'info' }],
        report_id: 'rep-1',
        ranking_list_id: 'rank-1'
      }));
    }
    if (url === '/api/tracks/track-1/ranking?metric=mase&policy=latest_valid_result') {
      return Promise.resolve(jsonResponse({ items: [{ model_id: 'model-1', rank: 1, metric_value: 0.2 }] }));
    }
    return Promise.resolve(jsonResponse({}));
  });
}

describe('App routes and artifact links', () => {
  beforeEach(() => {
    resetWizard();
    vi.restoreAllMocks();
    window.location.hash = '#/';
  });

  it('routes report hashes to the report view instead of the create workflow', async () => {
    const fetchMock = mockFetch();
    window.location.hash = '#/reports/rep-1';

    render(App);

    expect(await screen.findByRole('heading', { name: 'Benchmark report' })).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith('/api/reports/rep-1', expect.any(Object));
  });

  it('routes artifact hashes to dedicated view pages', async () => {
    mockFetch();

    window.location.hash = '#/datasets/manifest-1';
    const rendered = render(App);
    expect(await screen.findByRole('heading', { name: 'Dataset manifest' })).toBeTruthy();
    expect(await screen.findByText('/tmp/hourly.csv')).toBeTruthy();

    window.location.hash = '#/load-jobs/load-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Dataset load job' })).toBeTruthy();
    expect(await screen.findByText('Succeeded')).toBeTruthy();

    window.location.hash = '#/shards/shard-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Shard detail' })).toBeTruthy();
    expect(await screen.findByText('sample-1')).toBeTruthy();

    window.location.hash = '#/runs/run-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Benchmarking run' })).toBeTruthy();
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();

    window.location.hash = '#/tracks/track-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Track detail' })).toBeTruthy();
    expect((await screen.findAllByText('Lower is better')).length).toBeGreaterThan(0);

    rendered.unmount();
  });

  it('exposes view links for created workflow artifacts', async () => {
    mockFetch();
    wizardState.preview = {
      upload_id: 'upload-1',
      source_uri: '/tmp/hourly.csv',
      columns: [{ name: 'time' }, { name: 'target' }],
      preview_rows: []
    };
    wizardState.manifestId = 'manifest-1';
    wizardState.loadJobId = 'load-1';
    wizardState.shardId = 'shard-1';
    wizardState.trackId = 'track-1';
    wizardState.runId = 'run-1';
    wizardState.reportId = 'rep-1';

    render(EvaluationWizardPage);

    // The wizard keeps a persistent "Created artifacts" panel with deep links,
    // independent of which step is currently active.
    await screen.findByRole('link', { name: 'Dataset manifest' });
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Dataset manifest' }).getAttribute('href')).toBe('#/datasets/manifest-1');
      expect(screen.getByRole('link', { name: 'Load job' }).getAttribute('href')).toBe('#/load-jobs/load-1');
      expect(screen.getByRole('link', { name: 'Shard' }).getAttribute('href')).toBe('#/shards/shard-1');
      expect(screen.getByRole('link', { name: 'Track' }).getAttribute('href')).toBe('#/tracks/track-1');
      expect(screen.getByRole('link', { name: 'Run' }).getAttribute('href')).toBe('#/runs/run-1');
      expect(screen.getByRole('link', { name: 'Report' }).getAttribute('href')).toBe('#/reports/rep-1');
    });
  });
});
