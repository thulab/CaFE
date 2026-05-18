import { render, screen } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.vue';
import { resetWizard } from '../stores/wizard';

describe('frontend smoke flow', () => {
  beforeEach(() => {
    resetWizard();
    vi.restoreAllMocks();
  });

  it('renders the evaluation wizard shell and model-backed run controls', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));

    render(App);

    expect(screen.getByLabelText('CSV file')).toBeTruthy();
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();
  });
});
