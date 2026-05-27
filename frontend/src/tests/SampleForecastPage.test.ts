import { render, screen } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import SampleForecastPage from '../pages/SampleForecastPage.vue';

describe('SampleForecastPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setLocale('en-US');
  });

  it('shows history, ground truth, multiple forecasts, failed status, and metrics', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      sample_id: 's1',
      target_history: [[1], [2]],
      target_future: [[3], [4]],
      models: [
        { model_id: 'm1', model_name: 'Timer', status: 'succeeded', forecast: [[3.1], [3.9]], metrics: { mse: 0.01, mae: 0.1 } },
        { model_id: 'm2', model_name: 'Chronos', status: 'failed', forecast: null, metrics: {}, error_message: 'failed' }
      ]
    }), { status: 200 }));

    render(SampleForecastPage, { props: { sampleId: 's1', runId: 'r1' }, global: { plugins: [i18n] } });

    // "Timer" appears both in the chart legend and the metric table.
    expect((await screen.findAllByText('Timer')).length).toBeGreaterThan(0);
    expect(screen.getByText('Ground truth')).toBeTruthy();
    expect(screen.getByText('History')).toBeTruthy();
    expect(screen.getByText('Chronos')).toBeTruthy();
    expect(screen.getByText('failed')).toBeTruthy();
    expect(screen.getByText('0.01')).toBeTruthy();
  });
});
