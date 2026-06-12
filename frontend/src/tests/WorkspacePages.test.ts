import { fireEvent, render, screen, waitFor, within } from '@testing-library/vue';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import DatasetsPage from '../pages/DatasetsPage.vue';
import HomePage from '../pages/HomePage.vue';
import RunDetailPage from '../pages/RunDetailPage.vue';
import RunsPage from '../pages/RunsPage.vue';
import { authState } from '../stores/auth';

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
  beforeEach(() => {
    setLocale('en-US');
    authState.user = {
      user_id: 'admin-test',
      username: 'admin',
      email: null,
      is_active: true,
      is_superuser: true,
      roles: ['admin'],
      permissions: [],
    };
  });
  afterEach(() => {
    authState.user = null;
    vi.restoreAllMocks();
  });

  it('HomePage renders the empty activity state when backend has nothing', async () => {
    stubBackend({});
    render(HomePage, { global: { plugins: [i18n] } });
    expect(screen.getByRole('heading', { name: 'Workbench overview' })).toBeTruthy();
    expect(await screen.findByText('No activity yet')).toBeTruthy();
  });

  it('HomePage shows the track total in the workspace overview stats', async () => {
    stubBackend({
      '/dataset-manifests': { items: [], total: 3, limit: 1, offset: 0 },
      '/shards': { items: [], total: 4, limit: 1, offset: 0 },
      '/tracks': { items: [], total: 2 },
      '/benchmarking-runs': { items: [], total: 5, limit: 1, offset: 0 },
      '/reports': { items: [], total: 6, limit: 1, offset: 0 },
    });

    render(HomePage, { global: { plugins: [i18n] } });

    const datasetsTile = screen.getByRole('link', { name: /Datasets/ });
    const tracksTile = screen.getByRole('link', { name: /Tracks/ });
    const runsTile = screen.getByRole('link', { name: /Runs/ });
    await waitFor(() => {
      expect(within(datasetsTile).getByText('7')).toBeTruthy();
      expect(within(tracksTile).getByText('2')).toBeTruthy();
      expect(within(runsTile).getByText('5')).toBeTruthy();
      expect(screen.getByText('6')).toBeTruthy();
    });
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
    render(DatasetsPage, { global: { plugins: [i18n] } });
    expect(screen.getByRole('heading', { name: 'Datasets' })).toBeTruthy();
    expect(await screen.findByText('Uploaded dataset')).toBeTruthy();
    expect(await screen.findByText('Test cases · target')).toBeTruthy();
  });

  it('DatasetsPage uploads a data file and generates a reusable test case set', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const path = url.replace(/^\/api/, '').split('?')[0];
      if (path === '/dataset-manifests/upload') {
        return jsonResponse({
          upload_id: 'upload-9',
          source_uri: '/tmp/data.tsfile',
          file_format: 'tsfile',
          columns: [{ name: 'temperature' }, { name: 'pressure' }],
          preview_rows: []
        });
      }
      if (path === '/dataset-manifests' && init?.method === 'POST') {
        return jsonResponse({ dataset_manifest_id: 'manifest-9' });
      }
      if (path === '/dataset-load-jobs' && init?.method === 'POST') {
        return jsonResponse({ load_job_id: 'load-9', status: 'succeeded', output_shard_id: 'shard-9' });
      }
      if (path === '/dataset-manifests') {
        return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      }
      if (path === '/shards') {
        return jsonResponse({ items: [], total: 0, limit: 200, offset: 0 });
      }
      return jsonResponse({});
    });

    render(DatasetsPage, { global: { plugins: [i18n] } });

    await fireEvent.change(await screen.findByLabelText('Data file'), {
      target: { files: [new File(['tsfile'], 'data.tsfile')] }
    });
    expect((await screen.findAllByText('temperature')).length).toBeGreaterThan(0);
    await fireEvent.click(screen.getByLabelText('Select target temperature'));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    await waitFor(() => expect(screen.getByText('Test case set ready: shard-9')).toBeTruthy());
    const manifestBody = JSON.parse(fetchSpy.mock.calls.find((call) => String(call[0]) === '/api/dataset-manifests')![1]!.body as string);
    expect(manifestBody.file_format).toBe('tsfile');
    expect(manifestBody).not.toHaveProperty('value_columns');
    const loadJobBody = JSON.parse(fetchSpy.mock.calls.find((call) => String(call[0]) === '/api/dataset-load-jobs')![1]!.body as string);
    expect(loadJobBody.split_config).toMatchObject({
      context_length: 60,
      horizon: 16,
      stride: 16,
      shard_name: 'data test cases'
    });
  });

  it('DatasetsPage keeps loaded shard labels reactive after locale changes', async () => {
    stubBackend({
      '/dataset-manifests': {
        items: [],
        total: 0, limit: 200, offset: 0
      },
      '/shards': {
        items: [{ shard_id: 'shard-9', dataset_manifest_id: 'manifest-9', target_columns: ['target'], row_count: 20, created_at: '2026-05-26T12:01:00Z' }],
        total: 1, limit: 200, offset: 0
      }
    });
    render(DatasetsPage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Test cases · target')).toBeTruthy();
    expect(screen.getByText('target · 20 rows')).toBeTruthy();

    setLocale('zh-CN');

    expect(await screen.findByText('测试用例 · target')).toBeTruthy();
    expect(screen.getByText('target · 20 行')).toBeTruthy();
  });

  it('DatasetsPage localizes the missing shard target fallback', async () => {
    stubBackend({
      '/dataset-manifests': {
        items: [],
        total: 0, limit: 200, offset: 0
      },
      '/shards': {
        items: [{ shard_id: 'shard-9', dataset_manifest_id: 'manifest-9', target_columns: [], row_count: 20, created_at: '2026-05-26T12:01:00Z' }],
        total: 1, limit: 200, offset: 0
      }
    });
    render(DatasetsPage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Test cases · No target')).toBeTruthy();
    expect(screen.getByText('Test case set · 20 rows')).toBeTruthy();

    setLocale('zh-CN');

    expect(await screen.findByText('测试用例 · 未知目标')).toBeTruthy();
    expect(screen.getByText('测试用例集 · 20 行')).toBeTruthy();
  });

  it('HomePage localizes the missing shard target fallback', async () => {
    stubBackend({
      '/shards': {
        items: [{ shard_id: 'shard-9', target_columns: [], created_at: '2026-05-26T12:01:00Z' }],
        total: 1, limit: 8, offset: 0
      }
    });
    render(HomePage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Test cases · No target')).toBeTruthy();

    setLocale('zh-CN');

    expect(await screen.findByText('测试用例 · 未知目标')).toBeTruthy();
  });

  it('DatasetsPage keeps visible load errors reactive after locale changes', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => jsonResponse({
      error_code: 'forbidden',
      message: 'raw forbidden'
    }, 403));

    render(DatasetsPage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('You do not have permission to perform this action.')).toBeTruthy();

    setLocale('zh-CN');

    expect(await screen.findByText('你没有执行此操作的权限。')).toBeTruthy();
  });

  it('RunsPage starts empty then surfaces server-returned runs', async () => {
    stubBackend({});
    const empty = render(RunsPage, { global: { plugins: [i18n] } });
    expect(screen.getByRole('heading', { name: 'Runs' })).toBeTruthy();
    expect(await screen.findByText('No runs yet')).toBeTruthy();
    empty.unmount();
    vi.restoreAllMocks();

    stubBackend({
      '/benchmarking-runs': {
        items: [
          { benchmarking_run_id: 'run-9', track_id: 't1', model_ids: ['m1', 'm2'], status: 'running', activity_status: 'model_loading', model_count: 2, task_count: 2, sample_count: 4, created_at: '2026-05-26T13:00:00Z' },
          { benchmarking_run_id: 'run-10', track_id: 't1', model_ids: ['m1'], status: 'queued', model_count: 1, task_count: 1, sample_count: 2, created_at: '2026-05-26T12:00:00Z' }
        ],
        total: 2, limit: 200, offset: 0
      }
    });
    render(RunsPage, { global: { plugins: [i18n] } });
    expect((await screen.findAllByText('Run · 2 models')).length).toBeGreaterThan(0);
    expect((await screen.findAllByText('Run · 1 model')).length).toBeGreaterThan(0);
    expect(await screen.findByText('Loading model')).toBeTruthy();
  });

  it('RunDetailPage renders progress details from the backend', async () => {
    stubBackend({
      '/benchmarking-runs/run-9/progress': {
        benchmarking_run_id: 'run-9',
        status: 'running',
        activity_status: 'model_loading',
        progress: { total_models: 1, completed_models: 0, total_tasks: 1, completed_tasks: 0, total_samples: 4, completed_samples: 0, processed_samples: 0, failed_samples: 0 },
        units: [{ unit_id: 'unit-1', model_id: 'm1', model_name: 'Timer 3.5', status: 'running', activity_status: 'forecasting', task_count: 1, completed_task_count: 0, metrics: {} }],
        tasks: [{ task_id: 'task-1', unit_id: 'unit-1', model_id: 'm1', capability_block_id: 'block-1', capability_block_name: 'Real data', status: 'running', shard_count: 1, sample_count: 4, completed_sample_count: 0, processed_sample_count: 0, metrics: {} }],
        recent_events: [{ created_at: '2026-05-26T13:05:00Z', message: 'run active', event_type: 'status_changed', level: 'info' }],
        report_id: 'rep-1',
        ranking_list_id: 'rank-1'
      }
    });
    render(RunDetailPage, { props: { runId: 'run-9' }, global: { plugins: [i18n] } });

    expect(screen.getByRole('heading', { name: 'Benchmarking run' })).toBeTruthy();
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();
    expect(screen.getByText('Forecasting')).toBeTruthy();
    expect(screen.queryByText('Loading model')).toBeNull();
    expect(screen.getByText('run active')).toBeTruthy();
  });

  it('RunDetailPage restores active failed-sample rerun progress after refresh', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const path = url.replace(/^\/api/, '').split('?')[0];
      if (path === '/benchmarking-runs/run-failed/progress') {
        return jsonResponse({
          benchmarking_run_id: 'run-failed',
          status: 'partial_succeeded',
          progress: { total_models: 1, completed_models: 1, total_tasks: 1, completed_tasks: 1, total_samples: 10, completed_samples: 7, processed_samples: 10, failed_samples: 3 },
          units: [{ unit_id: 'unit-1', model_id: 'm1', model_name: 'Timer 3.5', status: 'partial_succeeded', task_count: 1, completed_task_count: 1, metrics: {} }],
          tasks: [{ task_id: 'task-1', unit_id: 'unit-1', model_id: 'm1', capability_block_id: 'block-1', capability_block_name: 'Real data', status: 'partial_succeeded', shard_count: 1, sample_count: 10, completed_sample_count: 7, processed_sample_count: 10, metrics: {} }],
          recent_events: [],
          report_id: 'rep-1',
          ranking_list_id: 'rank-1'
        });
      }
      if (path === '/benchmarking-runs/run-failed/failed-samples') {
        return jsonResponse({
          total: 3,
          limit: 0,
          offset: 0,
          summary: [{ error_code: 'adapter_error', error_message: 'adapter boom', count: 3, model_count: 1, capability_count: 1, sample_count: 3 }],
          items: []
        });
      }
      if (path === '/benchmarking-runs/run-failed/failed-samples/rerun') {
        return jsonResponse({
          active: {
            rerun_job_id: 'job-1',
            benchmarking_run_id: 'run-failed',
            status: 'running',
            activity_status: 'forecasting',
            total_samples: 10,
            processed_samples: 3,
            succeeded_samples: 2,
            failed_samples: 1,
            error_code: null,
            error_message: null,
            created_at: '2026-05-26T13:10:00Z',
            updated_at: '2026-05-26T13:10:05Z'
          }
        });
      }
      return jsonResponse({});
    });

    render(RunDetailPage, { props: { runId: 'run-failed' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Rerunning failed samples')).toBeTruthy();
    expect(await screen.findByText('Sample progress 3 / 10')).toBeTruthy();
    expect(await screen.findByText('Forecasting')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Rerun failed samples' })).toBeNull();
  });

  it('RunDetailPage shows failure reason summary before paginated sample details', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const path = url.replace(/^\/api/, '').split('?')[0];
      const query = new URL(url, 'http://test.local').searchParams;
      if (path === '/benchmarking-runs/run-failed/progress') {
        return jsonResponse({
          benchmarking_run_id: 'run-failed',
          status: 'partial_succeeded',
          progress: { total_models: 1, completed_models: 1, total_tasks: 1, completed_tasks: 1, total_samples: 4, completed_samples: 1, processed_samples: 4, failed_samples: 3 },
          units: [{ unit_id: 'unit-1', model_id: 'm1', model_name: 'Timer 3.5', status: 'partial_succeeded', task_count: 1, completed_task_count: 1, metrics: {} }],
          tasks: [{ task_id: 'task-1', unit_id: 'unit-1', model_id: 'm1', capability_block_id: 'block-1', capability_block_name: 'Real data', status: 'partial_succeeded', shard_count: 1, sample_count: 4, completed_sample_count: 1, processed_sample_count: 4, metrics: {} }],
          recent_events: [],
          report_id: 'rep-1',
          ranking_list_id: 'rank-1'
        });
      }
      if (path === '/benchmarking-runs/run-failed/failed-samples/rerun') {
        return jsonResponse({ active: null });
      }
      if (path === '/benchmarking-runs/run-failed/failed-samples') {
        const wantsDetails = query.get('error_code') === 'adapter_error';
        return jsonResponse({
          total: wantsDetails ? 3 : 3,
          limit: wantsDetails ? 50 : 0,
          offset: 0,
          summary: [{ error_code: 'adapter_error', error_message: 'adapter boom', count: 3, model_count: 1, capability_count: 1, sample_count: 3 }],
          items: wantsDetails
            ? [{
                forecast_artifact_id: 'artifact-1',
                sample_id: 'sample-1',
                sample_index: 2,
                model_id: 'm1',
                model_name: 'Timer 3.5',
                unit_id: 'unit-1',
                task_id: 'task-1',
                capability_block_id: 'block-1',
                capability_block_name: 'Real data',
                shard_id: 'shard-1',
                error_code: 'adapter_error',
                error_message: 'adapter boom'
              }]
            : []
        });
      }
      return jsonResponse({});
    });

    render(RunDetailPage, { props: { runId: 'run-failed' }, global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByRole('button', { name: /Failed samples/ }));

    expect(await screen.findByText('Failure reason summary')).toBeTruthy();
    expect(await screen.findByText('adapter_error')).toBeTruthy();
    expect(await screen.findByText('adapter boom')).toBeTruthy();
    expect(screen.queryByText('Window #3')).toBeNull();

    await fireEvent.click(screen.getByRole('button', { name: 'View samples' }));

    expect(await screen.findByText('Window #3')).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: 'Collapse details' }));

    expect(screen.queryByText('Window #3')).toBeNull();
  });

  it('renders workspace page chrome in Chinese when locale changes', async () => {
    setLocale('zh-CN');
    stubBackend({});
    render(HomePage, { global: { plugins: [i18n] } });

    expect(screen.getByRole('heading', { name: '工作台概览' })).toBeTruthy();
    expect(await screen.findByText('暂无活动')).toBeTruthy();
  });
});
