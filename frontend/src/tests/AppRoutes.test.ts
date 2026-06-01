import { fireEvent, render, screen, waitFor, within } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.vue';
import { i18n, setLocale } from '../i18n';
import EvaluationWizardPage from '../pages/EvaluationWizardPage.vue';
import LoadJobPage from '../pages/LoadJobPage.vue';
import { authState } from '../stores/auth';
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
    if (url === '/api/tracks') {
      return Promise.resolve(jsonResponse({
        items: [{ track_id: 'track-1', name: 'Hourly energy', primary_metric_id: 'mase', status: 'ready', created_at: '2026-01-01T00:00:00Z' }]
      }));
    }
    if (url === '/api/shards?limit=200') {
      return Promise.resolve(jsonResponse({
        items: [{ shard_id: 'shard-1', dataset_manifest_id: 'manifest-1', source_uri: '/tmp/hourly.csv', status: 'ready', row_count: 20, target_columns: ['target'], target_dim: 1, context_length: 6, horizon: 3, stride: 3, sample_count: 4 }],
        total: 1,
        limit: 200,
        offset: 0
      }));
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
        status: 'ready_to_load'
      }));
    }
    if (url === '/api/dataset-load-jobs/load-1') {
      return Promise.resolve(jsonResponse({
        load_job_id: 'load-1',
        dataset_manifest_id: 'manifest-1',
        status: 'succeeded',
        current_step: 'materializing_samples',
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
    if (url.startsWith('/api/shards/shard-1/samples')) {
      const params = new URLSearchParams(url.split('?')[1] || '');
      const offset = Number(params.get('offset') || 0);
      const limit = Number(params.get('limit') || 20);
      const firstIndex = offset >= 10 ? 10 : 0;
      const count = offset >= 10 ? 2 : Math.min(limit, 10);
      return Promise.resolve(jsonResponse({
        items: Array.from({ length: count }, (_, index) => {
          const sampleIndex = firstIndex + index;
          return {
            sample_id: sampleIndex === 0 ? 'sample-1' : `sample-${sampleIndex + 1}`,
            sample_index: sampleIndex,
            context_start: sampleIndex * 10,
            context_end: sampleIndex * 10 + 59,
            horizon_start: sampleIndex * 10 + 60,
            horizon_end: sampleIndex * 10 + 75
          };
        }),
        total: 12,
        limit,
        offset
      }));
    }
    if (url === '/api/samples/sample-1/preview') {
      return Promise.resolve(jsonResponse({
        sample_id: 'sample-1',
        shard_id: 'shard-1',
        sample_index: 0,
        history_timestamps: ['2026-01-01T00:00:00', '2026-01-01T01:00:00'],
        future_timestamps: ['2026-01-01T02:00:00'],
        target_column_names: ['target'],
        target_history: [[1], [2]],
        target_future: [[3]]
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
    if (url === '/api/benchmarking-runs?limit=200&track_id=track-1') {
      return Promise.resolve(jsonResponse({
        items: [
          {
            benchmarking_run_id: 'run-1',
            track_id: 'track-1',
            model_ids: ['model-1'],
            status: 'succeeded',
            model_count: 1,
            task_count: 1,
            sample_count: 4,
            report_id: 'rep-1',
            created_at: '2026-01-01T00:00:00Z'
          }
        ],
        total: 1,
        limit: 200,
        offset: 0
      }));
    }
    if (url === '/api/tracks/track-1/ranking?metric=mase&policy=latest_valid_result') {
      return Promise.resolve(jsonResponse({ items: [{ model_id: 'model-1', rank: 1, metric_value: 0.2 }] }));
    }
    if (url === '/api/ranking-lists') {
      return Promise.resolve(jsonResponse({ items: [] }));
    }
    return Promise.resolve(jsonResponse({}));
  });
}

describe('App routes and artifact links', () => {
  beforeEach(() => {
    resetWizard();
    window.sessionStorage.clear();
    vi.restoreAllMocks();
    setLocale('en-US');
    authState.user = {
      user_id: 'admin-test',
      username: 'admin',
      email: null,
      is_active: true,
      is_superuser: true,
      roles: ['admin'],
      permissions: []
    };
    window.location.hash = '#/';
  });

  it('routes report hashes to the report view instead of the create workflow', async () => {
    const fetchMock = mockFetch();
    window.location.hash = '#/reports/rep-1';

    render(App, { global: { plugins: [i18n] } });

    expect(await screen.findByRole('heading', { name: 'Benchmark report' })).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith('/api/reports/rep-1', expect.any(Object));
  });

  it('routes artifact hashes to dedicated view pages', async () => {
    mockFetch();

    window.location.hash = '#/datasets/manifest-1';
    const rendered = render(App, { global: { plugins: [i18n] } });
    expect(await screen.findByRole('heading', { name: 'Dataset manifest' })).toBeTruthy();
    expect(await screen.findByText('/tmp/hourly.csv')).toBeTruthy();

    window.location.hash = '#/load-jobs/load-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Dataset load job' })).toBeTruthy();
    expect(await screen.findByText('Succeeded')).toBeTruthy();

    window.location.hash = '#/shards/shard-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Test case set detail' })).toBeTruthy();
    expect(await screen.findByText('sample-1')).toBeTruthy();

    window.location.hash = '#/runs/run-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Benchmarking run' })).toBeTruthy();
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();

    window.location.hash = '#/tracks/track-1';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(await screen.findByRole('heading', { name: 'Track detail' })).toBeTruthy();
    expect(await screen.findByText('Run · 1 model')).toBeTruthy();
    expect(screen.getByRole('link', { name: /Open report/ }).getAttribute('href')).toBe('#/reports/rep-1');
    expect((await screen.findAllByText('Lower is better')).length).toBeGreaterThan(0);

    rendered.unmount();
  });

  it('paginates test case set samples and links to curve previews', async () => {
    const fetchMock = mockFetch();
    window.location.hash = '#/shards/shard-1';

    render(App, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Showing 1-10 of 12 samples')).toBeTruthy();
    expect(screen.getAllByRole('link', { name: /Open curve/ })[0].getAttribute('href')).toBe('#/shards/shard-1/samples/sample-1');

    await fireEvent.click(screen.getByRole('button', { name: 'Next page' }));

    await waitFor(() => expect(fetchMock.mock.calls.some((call) => String(call[0]) === '/api/shards/shard-1/samples?limit=10&offset=10')).toBe(true));
    expect(await screen.findByText('Showing 11-12 of 12 samples')).toBeTruthy();
    expect(screen.getByText('Window #11')).toBeTruthy();
  });

  it('routes test case sample hashes to the real sample preview page', async () => {
    mockFetch();
    window.location.hash = '#/shards/shard-1/samples/sample-1';

    render(App, { global: { plugins: [i18n] } });

    expect(await screen.findByRole('heading', { name: 'Sample window preview' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Back to test case set' }).getAttribute('href')).toBe('#/shards/shard-1');
  });

  it('continues an unfinished wizard when New evaluation is clicked', async () => {
    mockFetch();
    window.location.hash = '#/shards/shard-1';
    wizardState.entryMode = 'new-track';
    wizardState.preview = {
      upload_id: 'upload-1',
      source_uri: '/tmp/hourly.csv',
      columns: [{ name: 'time' }, { name: 'target' }],
      preview_rows: []
    };
    wizardState.shardId = 'shard-1';
    wizardState.step = 2;

    render(App, { global: { plugins: [i18n] } });

    const newEvaluationLinks = await screen.findAllByRole('link', { name: /Continue evaluation/ });
    await fireEvent.click(newEvaluationLinks.at(-1)!);

    await waitFor(() => {
      expect(wizardState.preview?.upload_id).toBe('upload-1');
      expect(wizardState.step).toBe(2);
      expect(window.location.hash).toBe('#/new');
    });
  });

  it('starts a new wizard session when there is no unfinished draft', async () => {
    mockFetch();
    window.location.hash = '#/new';

    render(App, { global: { plugins: [i18n] } });

    const newEvaluationLinks = await screen.findAllByRole('link', { name: /New evaluation/ });
    await fireEvent.click(newEvaluationLinks.at(-1)!);

    await waitFor(() => {
      expect(wizardState.preview).toBeNull();
      expect(wizardState.step).toBe(0);
      expect(window.location.hash).toMatch(/^#\/new\?session=/);
    });
  });

  it('routes the track workspace list to an authenticated Tracks page', async () => {
    mockFetch();
    window.location.hash = '#/tracks';

    render(App, { global: { plugins: [i18n] } });

    expect(await screen.findByRole('heading', { name: 'Tracks' })).toBeTruthy();
    expect(await screen.findByText('All evaluation tracks you created or can access.')).toBeTruthy();
  });

  it('routes track creation to the standalone track wizard', async () => {
    mockFetch();
    window.location.hash = '#/tracks/new';

    render(App, { global: { plugins: [i18n] } });

    expect((await screen.findAllByRole('heading', { name: 'Create track' })).length).toBeGreaterThan(0);
    expect(screen.getByText('Name the benchmark track, add or reuse test case sets, then save it for future runs.')).toBeTruthy();
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
    wizardState.entryMode = 'new-track';

    render(EvaluationWizardPage, { global: { plugins: [i18n] } });

    // The wizard keeps a persistent "Created artifacts" panel with deep links,
    // independent of which step is currently active.
    await screen.findByRole('link', { name: 'Dataset manifest' });
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Dataset manifest' }).getAttribute('href')).toBe('#/datasets/manifest-1');
      expect(screen.getByRole('link', { name: 'Load job' }).getAttribute('href')).toBe('#/load-jobs/load-1');
      expect(screen.getByRole('link', { name: 'Test case set' }).getAttribute('href')).toBe('#/shards/shard-1');
      expect(screen.getByRole('link', { name: 'Track' }).getAttribute('href')).toBe('#/tracks/track-1');
      expect(screen.getByRole('link', { name: 'Run' }).getAttribute('href')).toBe('#/runs/run-1');
      expect(screen.getByRole('link', { name: 'Report' }).getAttribute('href')).toBe('#/reports/rep-1');
    });
  });

  it('offers a resume link on every artifact page created by the current wizard', async () => {
    mockFetch();
    wizardState.entryMode = 'new-track';
    wizardState.step = 4;
    wizardState.manifestId = 'manifest-1';
    wizardState.loadJobId = 'load-1';
    wizardState.shardId = 'shard-1';
    wizardState.trackId = 'track-1';
    wizardState.runId = 'run-1';
    wizardState.reportId = 'rep-1';

    window.location.hash = '#/datasets/manifest-1';
    const rendered = render(App, { global: { plugins: [i18n] } });

    const routes = [
      { hash: '#/datasets/manifest-1', heading: 'Dataset manifest' },
      { hash: '#/load-jobs/load-1', heading: 'Dataset load job' },
      { hash: '#/shards/shard-1', heading: 'Test case set detail' },
      { hash: '#/tracks/track-1', heading: 'Track detail' },
      { hash: '#/tracks/track-1/ranking', heading: 'Track ranking' },
      { hash: '#/runs/run-1', heading: 'Benchmarking run' },
      { hash: '#/reports/rep-1', heading: 'Benchmark report' },
    ];

    for (const route of routes) {
      window.location.hash = route.hash;
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      expect(await screen.findByRole('heading', { name: route.heading })).toBeTruthy();
      expect((await screen.findByRole('link', { name: 'Continue current evaluation' })).getAttribute('href')).toBe('#/new');
    }

    rendered.unmount();
  });

  it('switches app shell navigation language without changing the route', async () => {
    mockFetch();
    window.location.hash = '#/tracks/track-1/ranking';

    render(App, { global: { plugins: [i18n] } });

    expect((await screen.findAllByText('Leaderboards')).length).toBeGreaterThan(0);
    const breadcrumb = screen.getByLabelText('Breadcrumb');
    expect(within(breadcrumb).getByText('Track · Leaderboards')).toBeTruthy();

    const zhButton = await screen.findByTitle('中文');
    await fireEvent.click(zhButton);

    expect((await screen.findAllByText('排行榜')).length).toBeGreaterThan(0);
    expect(screen.getByText('工作台')).toBeTruthy();
    expect(screen.getByLabelText('面包屑导航')).toBeTruthy();
    expect(within(breadcrumb).getByText('赛道 · 排行榜')).toBeTruthy();
    expect(window.location.hash).toBe('#/tracks/track-1/ranking');
  });

  it('localizes load job current step labels without changing backend step values', async () => {
    setLocale('zh-CN');
    mockFetch();

    render(LoadJobPage, { props: { loadJobId: 'load-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('正在生成样本')).toBeTruthy();
    expect(screen.queryByText('Materializing samples')).toBeNull();
  });
});
