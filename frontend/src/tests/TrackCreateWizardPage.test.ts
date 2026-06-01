import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrackCreateWizardPage from '../pages/TrackCreateWizardPage.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

describe('TrackCreateWizardPage', () => {
  beforeEach(() => {
    resetWizard();
    window.sessionStorage.clear();
    window.location.hash = '#/tracks/new';
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('creates a track from existing test case sets and returns to the track detail page', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push({ url, body: init?.body ? JSON.parse(init.body as string) : undefined });
      if (url.startsWith('/api/shards')) {
        return jsonResponse({
          items: [
            {
              shard_id: 'shard-1',
              name: 'Energy validation cases',
              dataset_name: 'Energy',
              source_uri: '/tmp/energy.csv',
              target_columns: ['load'],
              context_length: 60,
              horizon: 16,
              stride: 16,
              sample_count: 20,
              row_count: 96,
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

    render(TrackCreateWizardPage, { global: { plugins: [i18n] } });

    expect((await screen.findAllByRole('heading', { name: 'Create track' })).length).toBeGreaterThan(0);
    expect(wizardState.flow).toBe('track');

    await fireEvent.update(screen.getByLabelText('Track name'), 'Energy benchmark');
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Skip upload' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Choose existing test case sets' }));

    expect(await screen.findByText('Energy validation cases')).toBeTruthy();
    await fireEvent.click(screen.getByLabelText('Select Energy validation cases'));
    await fireEvent.click(screen.getByRole('button', { name: 'Create track from selected sets' }));

    await waitFor(() => expect(window.location.hash).toBe('#/tracks/track-1'));
    const post = calls.find((call) => call.url === '/api/wizard/real-dataset-track');
    expect(post?.body).toEqual({ name: 'Energy benchmark', shard_ids: ['shard-1'], primary_metric_id: 'mase' });
    expect(wizardState.flow).toBe('evaluation');
    expect(wizardState.entryMode).toBe('');
  });
});
