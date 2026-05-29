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
});
