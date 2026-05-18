import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import RunStep from '../components/wizard/RunStep.vue';
import { resetWizard, wizardState } from '../stores/wizard';

describe('RunStep', () => {
  beforeEach(() => {
    resetWizard();
    wizardState.trackId = 'track-1';
    vi.useFakeTimers();
    vi.restoreAllMocks();
  });

  it('requires at least one selected model', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
    render(RunStep);

    await screen.findByText('Timer 3.5');
    expect((screen.getByRole('button', { name: 'Run' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('starts five-second polling and stops on terminal status', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'succeeded', progress: {}, units: [], tasks: [], recent_events: [], report_id: 'rep1' }), { status: 200 }));
    render(RunStep);

    await fireEvent.click(await screen.findByLabelText('Timer 3.5'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await vi.advanceTimersByTimeAsync(5000);

    await waitFor(() => expect(wizardState.reportId).toBe('rep1'));
  });
});
