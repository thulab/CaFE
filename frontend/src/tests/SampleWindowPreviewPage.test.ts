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

    expect(await screen.findByRole('heading', { name: 'Sample window preview' })).toBeTruthy();
    expect(await screen.findByRole('img', { name: 'Sample window chart with 2 history steps and 1 future step.' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Back to test case set' }).getAttribute('href')).toBe('#/shards/shard-1');
    expect(screen.getByText('Window #1')).toBeTruthy();
  });
});
