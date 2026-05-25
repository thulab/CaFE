import { render, screen } from '@testing-library/vue';
import { nextTick } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.vue';
import { resetWizard, wizardState } from '../stores/wizard';

describe('frontend smoke flow', () => {
  beforeEach(() => {
    resetWizard();
    vi.restoreAllMocks();
    window.location.hash = '#/new';
  });

  it('renders the workbench shell, the guided wizard, and model-backed run controls', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));

    render(App);

    // App shell + wizard sub-page
    expect(screen.getByText('TSBenchmark')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'New evaluation' })).toBeTruthy();
    // Upload step is active first
    expect(screen.getByLabelText('CSV file')).toBeTruthy();

    // Model-backed run controls appear at the run step
    wizardState.step = 4;
    await nextTick();
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();
  });
});
