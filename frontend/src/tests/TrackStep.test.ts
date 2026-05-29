import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrackStep from '../components/wizard/TrackStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

describe('TrackStep', () => {
  beforeEach(() => {
    resetWizard();
    wizardState.shardId = 'shard-1';
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('keeps the generated track payload name stable under Chinese UI', async () => {
    setLocale('zh-CN');
    wizardState.trackName = 'Energy demand track';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      track_id: 'track-1',
      capability_block_id: 'block-1',
      ranking_list_id: 'ranking-1',
    }), { status: 200 }));

    render(TrackStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByRole('button', { name: '创建赛道' }));

    await waitFor(() => expect(wizardState.trackId).toBe('track-1'));
    const body = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(body.name).toBe('Energy demand track');
    expect(body.primary_metric_id).toBe('mase');
  });

  it('uses the editable track name and selected primary metric', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      track_id: 'track-1',
      capability_block_id: 'block-1',
      ranking_list_id: 'ranking-1',
    }), { status: 200 }));

    render(TrackStep, { global: { plugins: [i18n] } });

    await fireEvent.update(screen.getByLabelText('Track name'), 'Hourly energy MSE track');
    await fireEvent.update(screen.getByLabelText('Primary metric'), 'mse');
    await fireEvent.click(screen.getByRole('button', { name: 'Create track' }));

    await waitFor(() => expect(wizardState.trackId).toBe('track-1'));
    const body = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(body.name).toBe('Hourly energy MSE track');
    expect(body.primary_metric_id).toBe('mse');
  });
});
