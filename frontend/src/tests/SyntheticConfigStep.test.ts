import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SyntheticConfigStep from '../components/wizard/SyntheticConfigStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

describe('SyntheticConfigStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    wizardState.dataSource = 'synthetic';
    wizardState.trackName = 'Synthetic benchmark';
    wizardState.step = 2;
    vi.restoreAllMocks();
  });

  it('loads capabilities and generates selected synthetic test case sets', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/synthetic/capabilities') {
        return jsonResponse({
          items: [
            {
              capability_id: 'trend',
              label: 'Trend',
              description: 'Trend capability',
              task_type: 'univariate_forecast',
              target_dim_mode: 'fixed_1',
              covariate_columns: [],
            },
            {
              capability_id: 'common_factor',
              label: 'Common factor',
              description: 'Shared factors',
              task_type: 'multivariate_forecast',
              target_dim_mode: 'multi',
              covariate_columns: [],
            },
          ]
        });
      }
      if (url === '/api/synthetic/shards') {
        return jsonResponse({ shard_ids: ['shard-trend', 'shard-common'], items: [] });
      }
      return jsonResponse({});
    });

    render(SyntheticConfigStep, { global: { plugins: [i18n] } });

    expect(await screen.findByText('Trend')).toBeTruthy();
    await fireEvent.click(screen.getByLabelText('Select capability Common factor'));
    await fireEvent.update(screen.getByLabelText('Sample count'), '5');
    await fireEvent.click(screen.getByRole('button', { name: 'Generate synthetic test cases' }));

    await waitFor(() => expect(wizardState.shardId).toBe('shard-trend'));
    expect(wizardState.selectedShardIds).toEqual(['shard-trend', 'shard-common']);
    expect(wizardState.step).toBe(3);

    const generateCall = fetchSpy.mock.calls.find((call) => String(call[0]) === '/api/synthetic/shards');
    const body = JSON.parse(generateCall![1]!.body as string);
    expect(body).toMatchObject({
      name: 'Synthetic benchmark synthetic cases',
      capabilities: ['trend', 'common_factor'],
      sample_count: 5,
      context_length: 60,
      horizon: 16,
    });
  });
});
