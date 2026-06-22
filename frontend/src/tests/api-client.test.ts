import { beforeEach, describe, expect, it, vi } from 'vitest';
import { uploadDataset, createLoadJob, getShardSamples } from '../api/datasets';
import { createRun, getRunProgress } from '../api/runs';
import { getRanking, getReport, getSampleForecast } from '../api/results';
import { getSamplePreview } from '../api/samples';
import { ApiError } from '../api/client';

describe('api client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('constructs upload requests with FormData', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ upload_id: 'u1', source_uri: '/tmp/file.csv', columns: [], preview_rows: [] }));

    await uploadDataset(new File(['time,target\n'], 'data.csv'));

    expect(fetchMock).toHaveBeenCalledWith('/api/dataset-manifests/upload', expect.objectContaining({ method: 'POST', body: expect.any(FormData) }));
  });

  it('parses unified API errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ error_code: 'bad_input', message: 'Bad input', details: { field: 'name' } }, 400));

    await expect(createLoadJob({ dataset_manifest_id: 'm1', split_config: { context_length: 6, horizon: 3, target_columns: ['value'] } })).rejects.toMatchObject(
      new ApiError('bad_input', 'Bad input', { field: 'name' }, 400)
    );
  });

  it('reports empty non-json API failures without JSON parse errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 500, headers: { 'content-type': 'text/plain' } }));

    await expect(uploadDataset(new File(['time,target\n'], 'data.csv'))).rejects.toMatchObject(
      new ApiError('api_error', 'Request failed with status 500', { status: 500 }, 500)
    );
  });

  it('passes through ISO datetimes and typed response shapes', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse({ benchmarking_run_id: 'r1', status: 'running', created_at: '2026-01-01T00:00:00Z' }))
      .mockResolvedValueOnce(jsonResponse({ benchmarking_run_id: 'r1', status: 'succeeded', progress: { total_models: 1 }, units: [], tasks: [], recent_events: [{ created_at: '2026-01-01T00:00:01Z' }] }))
      .mockResolvedValueOnce(jsonResponse({ items: [{ model_id: 'm1', rank: 1, metric_value: 0.1 }] }))
      .mockResolvedValueOnce(jsonResponse({ report_id: 'rep1', model_metrics: [], task_summaries: [], sample_forecast_links: [] }))
      .mockResolvedValueOnce(jsonResponse({ sample_id: 's1', target_history: [], target_future: [], models: [] }));

    expect((await createRun({ track_id: 't1', model_ids: ['m1'] })).created_at).toBe('2026-01-01T00:00:00Z');
    expect((await getRunProgress('r1')).recent_events[0].created_at).toBe('2026-01-01T00:00:01Z');
    expect((await getRanking('t1', 'mse', 'latest_valid_result')).items[0].rank).toBe(1);
    expect((await getReport('rep1')).report_id).toBe('rep1');
    expect((await getSampleForecast('s1', 'r1')).sample_id).toBe('s1');
  });

  it('builds paginated shard sample and sample preview requests', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse({ items: [], total: 0, limit: 10, offset: 20 }))
      .mockResolvedValueOnce(jsonResponse({ sample_id: 'sample-1', target_history: [[1]], target_future: [[2]] }));

    await getShardSamples('shard-1', { limit: 10, offset: 20 });
    await getSamplePreview('sample-1');

    expect(String(fetchSpy.mock.calls[0]![0])).toBe('/api/shards/shard-1/samples?limit=10&offset=20');
    expect(String(fetchSpy.mock.calls[1]![0])).toBe('/api/samples/sample-1/preview');
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}
