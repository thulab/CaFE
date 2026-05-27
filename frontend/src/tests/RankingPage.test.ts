import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import RankingPage from '../pages/RankingPage.vue';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

// URL-aware mock: model-name resolution fetches /api/models, so route by URL
// instead of relying on call order.
function mockFetch() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
    const url = String(input);
    if (url === '/api/models') return Promise.resolve(jsonResponse({ items: [] }));
    if (url.includes('metric=mae')) return Promise.resolve(jsonResponse({ items: [{ model_id: 'm2', rank: 1, metric_value: 0.1 }] }));
    return Promise.resolve(jsonResponse({ items: [{ model_id: 'm1', rank: 1, metric_value: 0.2 }] }));
  });
}

describe('RankingPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setLocale('en-US');
  });

  it('queries metric and policy controls and displays lower-is-better ranking', async () => {
    const fetchMock = mockFetch();

    render(RankingPage, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    expect((await screen.findAllByText('m1')).length).toBeGreaterThan(0);
    await fireEvent.update(screen.getByLabelText('Metric'), 'mae');

    await waitFor(() => expect(fetchMock).toHaveBeenLastCalledWith('/api/tracks/track-1/ranking?metric=mae&policy=latest_valid_result', expect.any(Object)));
    expect(await screen.findByText('Lower is better')).toBeTruthy();
  });

  it('renders the ranking page title in Chinese', async () => {
    setLocale('zh-CN');
    mockFetch();

    render(RankingPage, { props: { trackId: 'track-1' }, global: { plugins: [i18n] } });

    expect(await screen.findByRole('heading', { name: '赛道排名' })).toBeTruthy();
  });
});
