import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrackStep from '../components/wizard/TrackStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

describe('TrackStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('captures the track name and primary metric before any data is uploaded', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    render(TrackStep, { global: { plugins: [i18n] } });

    await fireEvent.update(screen.getByLabelText('Track name'), 'Hourly energy benchmark');
    await fireEvent.update(screen.getByLabelText('Primary metric'), 'mse');
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => expect(wizardState.step).toBe(1));
    expect(wizardState.trackName).toBe('Hourly energy benchmark');
    expect(wizardState.primaryMetric).toBe('mse');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('does not overwrite an existing configured track name under Chinese UI', async () => {
    setLocale('zh-CN');
    wizardState.trackName = 'Energy demand track';

    render(TrackStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByRole('button', { name: '继续' }));

    await waitFor(() => expect(wizardState.step).toBe(1));
    expect(wizardState.trackName).toBe('Energy demand track');
    expect(wizardState.primaryMetric).toBe('mase');
  });
});
