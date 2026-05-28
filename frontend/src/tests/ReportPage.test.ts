import { fireEvent, render, screen, within } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import ReportPage from '../pages/ReportPage.vue';

describe('ReportPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setLocale('en-US');
  });

  it('shows model metrics, task errors, and sample forecast links', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      report_id: 'rep1',
      model_metrics: [{ model_id: 'm1', metrics: { mse: 0.2, mae: 0.3 } }],
      task_summaries: [{ task_id: 'task-1', status: 'failed', error_message: 'boom', metrics: {} }],
      sample_forecast_links: [{ sample_id: 's1', run_id: 'r1' }]
    }), { status: 200 }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    await screen.findByText('m1');
    expect(screen.getByText('boom')).toBeTruthy();
    expect(screen.getByText('s1')).toBeTruthy();
  });

  it('sorts model metrics by clicked metric headers', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      report_id: 'rep1',
      model_metrics: [
        { model_id: 'model-a', metrics: { mase: 0.4, mse: 0.1 } },
        { model_id: 'model-b', metrics: { mase: 0.2, mse: 0.3 } },
        { model_id: 'model-c', metrics: { mase: 0.3, mse: 0.2 } }
      ],
      task_summaries: [],
      sample_forecast_links: []
    }), { status: 200 }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    await screen.findByText('model-a');
    const tbody = screen.getByRole('table').querySelector('tbody')!;
    const names = () => within(tbody).getAllByRole('row').map((row) => within(row).getAllByRole('cell')[0].textContent);

    expect(names()).toEqual(['model-a', 'model-b', 'model-c']);

    await fireEvent.click(screen.getByRole('button', { name: 'Sort by MASE' }));
    expect(names()).toEqual(['model-b', 'model-c', 'model-a']);

    await fireEvent.click(screen.getByRole('button', { name: 'Sort by MSE' }));
    expect(names()).toEqual(['model-a', 'model-c', 'model-b']);

    await fireEvent.click(screen.getByRole('button', { name: 'Sort by MSE' }));
    expect(names()).toEqual(['model-b', 'model-c', 'model-a']);
  });
});
