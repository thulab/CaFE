import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import RunStep from '../components/wizard/RunStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

describe('RunStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    wizardState.trackId = 'track-1';
    vi.useFakeTimers();
    vi.restoreAllMocks();
  });

  it('requires at least one selected model', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
    render(RunStep, { global: { plugins: [i18n] } });

    await screen.findByText('Timer 3.5');
    expect((screen.getByRole('button', { name: 'Run' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('starts five-second polling and stops on terminal status', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs/r1/progress') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'succeeded', progress: {}, units: [], tasks: [], recent_events: [], report_id: 'rep1' }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });
    render(RunStep, { global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.5'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => expect(wizardState.runId).toBe('r1'));
    await vi.advanceTimersByTimeAsync(5000);

    await waitFor(() => expect(wizardState.reportId).toBe('rep1'));
  });

  it('loads unloaded timer-service models before starting a run', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      calls.push(url);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.0', adapter_type: 'timer_service', loaded: false }] }), { status: 200 }));
      }
      if (url === '/api/models/m1/load') {
        return Promise.resolve(new Response(JSON.stringify({ model_id: 'm1', name: 'Timer 3.0', adapter_type: 'timer_service', loaded: true }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });
    render(RunStep, { global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.0'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => expect(wizardState.runId).toBe('r1'));
    expect(calls.indexOf('/api/models/m1/load')).toBeGreaterThan(calls.indexOf('/api/models'));
    expect(calls.indexOf('/api/models/m1/load')).toBeLessThan(calls.indexOf('/api/benchmarking-runs'));
  });
});
