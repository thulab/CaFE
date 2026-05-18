import { render, screen } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ReportPage from '../pages/ReportPage.vue';

describe('ReportPage', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('shows model metrics, task errors, and sample forecast links', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      report_id: 'rep1',
      model_metrics: [{ model_id: 'm1', metrics: { mse: 0.2, mae: 0.3 } }],
      task_summaries: [{ task_id: 'task-1', status: 'failed', error_message: 'boom', metrics: {} }],
      sample_forecast_links: [{ sample_id: 's1', run_id: 'r1' }]
    }), { status: 200 }));

    render(ReportPage, { props: { reportId: 'rep1' } });

    await screen.findByText('m1');
    expect(screen.getByText('boom')).toBeTruthy();
    expect(screen.getByText('s1')).toBeTruthy();
  });
});
