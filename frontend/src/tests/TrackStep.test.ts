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

  it('uses the active locale for the generated track name', async () => {
    setLocale('zh-CN');
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      track_id: 'track-1',
      capability_block_id: 'block-1',
      ranking_list_id: 'ranking-1',
    }), { status: 200 }));

    render(TrackStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByRole('button', { name: '创建赛道' }));

    await waitFor(() => expect(wizardState.trackId).toBe('track-1'));
    const body = JSON.parse(fetchSpy.mock.calls[0][1].body as string);
    expect(body.name).toBe('真实数据集赛道');
  });
});
