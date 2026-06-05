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
      if (url === '/api/wizard/track-from-shards') {
        return jsonResponse({ track_id: 'track-1', capability_block_id: 'block-1', capability_block_ids: ['block-1'], ranking_list_id: 'ranking-1' });
      }
      return jsonResponse({});
    });

    render(TrackCreateWizardPage, { global: { plugins: [i18n] } });

    expect((await screen.findAllByRole('heading', { name: 'Create track' })).length).toBeGreaterThan(0);
    expect(wizardState.flow).toBe('track');

    await fireEvent.update(screen.getByLabelText('Track name'), 'Energy benchmark');
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    await fireEvent.click(screen.getByRole('button', { name: /Reuse existing sets/ }));
    await fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Choose existing test case sets' }));

    expect(await screen.findByText('Energy validation cases')).toBeTruthy();
    await fireEvent.click(screen.getByLabelText('Select Energy validation cases'));
    await fireEvent.click(screen.getByRole('button', { name: 'Create track from selected sets' }));

    await waitFor(() => expect(window.location.hash).toBe('#/tracks/track-1'));
    const post = calls.find((call) => call.url === '/api/wizard/track-from-shards');
    expect(post?.body).toEqual({ name: 'Energy benchmark', shard_ids: ['shard-1'], primary_metric_id: 'mase' });
    expect(wizardState.flow).toBe('evaluation');
    expect(wizardState.entryMode).toBe('');
  });

  it('uses the synthetic generation step when a new track starts from synthetic data', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push({ url, body: init?.body ? JSON.parse(init.body as string) : undefined });
      if (url === '/api/synthetic/capabilities') {
        return jsonResponse({
          items: [
            {
              capability_id: 'trend',
              label: 'Trend',
              description: 'Trend capability',
              task_type: 'univariate_forecast',
              target_dim_mode: 'fixed_1',
              covariate_columns: [],
            },
            {
              capability_id: 'common_factor',
              label: 'Common factor',
              description: 'Shared factors',
              task_type: 'multivariate_forecast',
              target_dim_mode: 'multi',
              covariate_columns: [],
            },
          ]
        });
      }
      if (url === '/api/synthetic/shards') {
        return jsonResponse({ shard_ids: ['shard-trend'], items: [] });
      }
      if (url.startsWith('/api/shards')) {
        return jsonResponse({
          items: [
            {
              shard_id: 'shard-trend',
              name: 'Synthetic track synthetic cases',
              dataset_name: 'Synthetic benchmark',
              source_uri: 'synthetic://fixed-anchor',
              target_columns: ['target'],
              covariate_columns: [],
              context_length: 60,
              horizon: 16,
              stride: 16,
              sample_count: 32,
              row_count: 76,
              status: 'ready',
              shard_type: 'synthetic',
              capability_type: 'trend',
              generation_config: { capability_label: 'Trend', difficulty: 3, seed: 0 },
            }
          ],
          total: 1,
          limit: 10,
          offset: 0
        });
      }
      return jsonResponse({});
    });

    render(TrackCreateWizardPage, { global: { plugins: [i18n] } });

    await fireEvent.update(await screen.findByLabelText('Track name'), 'Synthetic track');
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    await fireEvent.click(screen.getByRole('button', { name: /Generate synthetic data/ }));
    await fireEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByRole('heading', { name: 'Generate synthetic test cases' })).toBeTruthy();
    expect(await screen.findByText('Trend')).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: 'Generate synthetic test cases' }));

    await waitFor(() => expect(wizardState.shardId).toBe('shard-trend'));
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Select test cases' })).toBeTruthy());

    const generateCall = calls.find((call) => call.url === '/api/synthetic/shards');
    expect(generateCall?.body).toMatchObject({
      name: 'Synthetic track synthetic cases',
      capabilities: ['trend'],
      context_length: 60,
      horizon: 16,
      sample_count: 32,
    });
    expect(wizardState.selectedShardIds).toContain('shard-trend');
  });
});
