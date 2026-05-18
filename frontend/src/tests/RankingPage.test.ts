import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import RankingPage from '../pages/RankingPage.vue';

describe('RankingPage', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('queries metric and policy controls and displays lower-is-better ranking', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ model_id: 'm1', rank: 1, metric_value: 0.2 }] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ model_id: 'm2', rank: 1, metric_value: 0.1 }] }), { status: 200 }));

    render(RankingPage, { props: { trackId: 'track-1' } });

    await screen.findByText('m1');
    await fireEvent.update(screen.getByLabelText('Metric'), 'mae');

    await waitFor(() => expect(fetchMock).toHaveBeenLastCalledWith('/api/tracks/track-1/ranking?metric=mae&policy=latest_valid_result', expect.any(Object)));
    expect(screen.getByText('Lower is better')).toBeTruthy();
  });
});
