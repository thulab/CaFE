import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TestCaseSetStep from '../components/wizard/TestCaseSetStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

describe('TestCaseSetStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    wizardState.trackName = 'Hourly energy benchmark';
    wizardState.primaryMetric = 'mse';
    vi.restoreAllMocks();
  });

  it('preselects a generated test case set and creates a real-dataset track', async () => {
    wizardState.shardId = 'shard-generated';
    wizardState.selectedShardIds = ['shard-generated'];
    wizardState.step = 3;
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url.startsWith('/api/shards')) {
        return jsonResponse({
          items: [
            {
              shard_id: 'shard-generated',
              name: 'Hourly generated test cases',
              dataset_name: 'Hourly energy',
              source_uri: '/tmp/hourly.csv',
              target_columns: ['target'],
              context_length: 6,
              horizon: 3,
              stride: 3,
              sample_count: 12,
              row_count: 48,
              status: 'ready'
            }
          ],
          total: 1,
          limit: 10,
          offset: 0
        });
      }
      if (url === '/api/wizard/real-dataset-track') {
        return jsonResponse({ track_id: 'track-1', capability_block_id: 'block-1', ranking_list_id: 'ranking-1' });
      }
      return jsonResponse({});
    });

    render(TestCaseSetStep, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Hourly generated test cases')).toBeTruthy();
    expect((screen.getByLabelText('Select Hourly generated test cases') as HTMLInputElement).checked).toBe(true);

    await fireEvent.click(screen.getByRole('button', { name: 'Create track from selected sets' }));

    await waitFor(() => expect(wizardState.trackId).toBe('track-1'));
    const postCall = fetchSpy.mock.calls.find((call) => String(call[0]) === '/api/wizard/real-dataset-track');
    const body = JSON.parse(postCall![1]!.body as string);
    expect(body).toEqual({ name: 'Hourly energy benchmark', shard_ids: ['shard-generated'], primary_metric_id: 'mse' });
    expect(wizardState.step).toBe(4);
  });

  it('filters and pages available test case sets', async () => {
    const requests: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      requests.push(url);
      if (url.startsWith('/api/shards')) {
        const params = new URLSearchParams(url.split('?')[1] || '');
        return jsonResponse({
          items: [
            {
              shard_id: params.get('offset') === '10' ? 'shard-2' : 'shard-1',
              name: params.get('q') ? 'Filtered energy cases' : 'Energy validation cases',
              dataset_name: 'Energy',
              source_uri: '/tmp/energy.csv',
              target_columns: ['load'],
              context_length: 12,
              horizon: 6,
              stride: 3,
              sample_count: 20,
              row_count: 80,
              status: 'ready'
            }
          ],
          total: 11,
          limit: 10,
          offset: Number(params.get('offset') || 0)
        });
      }
      return jsonResponse({});
    });

    render(TestCaseSetStep, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Energy validation cases')).toBeTruthy();
    await fireEvent.update(screen.getByLabelText('Search test case sets'), 'energy');
    await waitFor(() => expect(requests.some((url) => url.includes('q=energy'))).toBe(true));

    await fireEvent.click(screen.getByRole('button', { name: 'Next page' }));

    await waitFor(() => expect(requests.some((url) => url.includes('offset=10'))).toBe(true));
  });

  it('omits redundant selected range text from the test case picker footer', async () => {
    setLocale('zh-CN');
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url.startsWith('/api/shards')) {
        return jsonResponse({ items: [], total: 0, limit: 10, offset: 0 });
      }
      return jsonResponse({});
    });

    render(TestCaseSetStep, { global: { plugins: [i18n] } });

    expect(await screen.findByText('未找到测试用例集。')).toBeTruthy();
    expect(screen.getByText('已选择 0 个测试用例集')).toBeTruthy();
    expect(screen.queryByText(/已选择\s*个\s*·\s*-\s*\//)).toBeNull();
  });
});
