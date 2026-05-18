import { beforeEach, describe, expect, it, vi } from 'vitest';
import { uploadDataset, createLoadJob } from '../api/datasets';
import { createRun, getRunProgress } from '../api/runs';
import { getRanking, getReport, getSampleForecast } from '../api/results';
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

    await expect(createLoadJob({ dataset_manifest_id: 'm1', split_config: { context_length: 6, horizon: 3 } })).rejects.toMatchObject(
      new ApiError('bad_input', 'Bad input', { field: 'name' })
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
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}
