import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrackRunPanel from '../components/tracks/TrackRunPanel.vue';
import { i18n, setLocale } from '../i18n';

describe('TrackRunPanel', () => {
  beforeEach(() => {
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('starts a run from an existing track without preloading models', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      calls.push(url);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.0', adapter_type: 'timer_service', loaded: false }] }), { status: 200 }));
      }
      if (url === '/api/models/m1/load') {
        return Promise.resolve(new Response(JSON.stringify({ error: 'preload should not be called' }), { status: 500 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });

    render(TrackRunPanel, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.0'));
    await fireEvent.click(screen.getByRole('button', { name: 'Start run' }));

    await waitFor(() => expect(screen.getByText('Run created: r1')).toBeTruthy());
    expect(calls).not.toContain('/api/models/m1/load');
    expect(calls).toContain('/api/benchmarking-runs');
  });

  it('disables single-target models when the existing track is multi-target', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({
          items: [
            { model_id: 'timer', name: 'Timer 3.0', adapter_type: 'timer_service', forecast_limits: { max_target_count: 1 } },
            { model_id: 'toto', name: 'toto2.0', adapter_type: 'timer_service', forecast_limits: { max_target_count: null } }
          ]
        }), { status: 200 }));
      }
      if (url === '/api/tracks/track-1') {
        return Promise.resolve(new Response(JSON.stringify({
          track_id: 'track-1',
          name: 'Multi target',
          primary_metric_id: 'mase',
          status: 'active',
          shard_ids: ['shard-1']
        }), { status: 200 }));
      }
      if (url === '/api/shards/shard-1') {
        return Promise.resolve(new Response(JSON.stringify({
          shard_id: 'shard-1',
          dataset_manifest_id: 'manifest-1',
          source_uri: '/tmp/multi.csv',
          status: 'ready',
          row_count: 20,
          target_columns: ['load', 'temperature'],
          target_dim: 2,
          context_length: 6,
          horizon: 3,
          stride: 3,
          sample_count: 4
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });

    render(TrackRunPanel, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    const timer = await screen.findByLabelText('Timer 3.0') as HTMLInputElement;
    const toto = await screen.findByLabelText('toto2.0') as HTMLInputElement;
    await waitFor(() => expect(timer.disabled).toBe(true));
    expect(toto.disabled).toBe(false);
  });
});
