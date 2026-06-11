import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import LeaderboardsPage from '../pages/LeaderboardsPage.vue';
import { authState } from '../stores/auth';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

describe('LeaderboardsPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setLocale('en-US');
    authState.user = {
      user_id: 'admin-test',
      username: 'admin',
      email: null,
      is_active: true,
      is_superuser: true,
      roles: ['admin'],
      permissions: [],
    };
  });

  afterEach(() => {
    authState.user = null;
  });

  it('lets an admin hide a leaderboard card from anonymous visitors', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
      const url = String(input);
      if (url === '/api/ranking-lists/ranking-1/visibility' && init?.method === 'PATCH') {
        return Promise.resolve(jsonResponse({
          ranking_list_id: 'ranking-1',
          track_id: 'track-1',
          public_visible: false,
          updated_at: '2026-06-11T00:00:00Z',
        }));
      }
      if (url === '/api/ranking-lists') {
        return Promise.resolve(jsonResponse({
          items: [{
            ranking_list_id: 'ranking-1',
            track_id: 'track-1',
            track_name: 'Hourly energy',
            track_type: 'benchmark',
            primary_metric_id: 'mase',
            default_policy: 'latest_valid_result',
            public_visible: true,
            updated_at: '2026-06-10T00:00:00Z',
            model_count: 0,
            run_count: 0,
            top: [],
          }],
        }));
      }
      return Promise.resolve(jsonResponse({ items: [] }));
    });

    render(LeaderboardsPage, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Hourly energy')).toBeTruthy();
    await fireEvent.click(screen.getByRole('button', { name: 'Hide from public' }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/ranking-lists/ranking-1/visibility', expect.objectContaining({ method: 'PATCH' }));
      expect(screen.getByText('Hidden')).toBeTruthy();
    });
  });
});
