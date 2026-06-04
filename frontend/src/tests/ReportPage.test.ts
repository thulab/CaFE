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
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0') {
        return Promise.resolve(new Response(JSON.stringify({
          report_id: 'rep1',
          track_id: 'track-1',
          model_metrics: [{ model_id: 'm1', metrics: { mse: 0.2, mae: 0.3 } }],
          task_summaries: [{ task_id: 'task-1', status: 'failed', error_message: 'boom', metrics: {} }],
          sample_forecast_links: [{ sample_id: 's1', run_id: 'r1', sample_index: 0, horizon_start: 6, horizon_end: 8, forecast_start_at: '2026-01-01T06:00:00', forecast_end_at: '2026-01-01T08:00:00', model_count: 1 }],
          sample_forecast_links_total: 1,
          sample_forecast_links_limit: 10,
          sample_forecast_links_offset: 0,
        }), { status: 200 }));
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    await screen.findByText('m1');
    expect(screen.getByText('boom')).toBeTruthy();
    expect(screen.getByText('Window #1')).toBeTruthy();
    expect(screen.getByText('Forecast rows 6-8')).toBeTruthy();
    expect(screen.getByRole('link', { name: /Open/ }).getAttribute('href')).toBe('#/samples/s1?run_id=r1&report_id=rep1');
    expect(screen.getByRole('link', { name: 'Back to track' }).getAttribute('href')).toBe('#/tracks/track-1');
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

  it('deduplicates sample forecast links and paginates them by window order', async () => {
    const links = Array.from({ length: 12 }, (_, index) => ({
      sample_id: `s${index + 1}`,
      run_id: 'r1',
      sample_index: index,
      horizon_start: index * 3 + 6,
      horizon_end: index * 3 + 8,
      forecast_start_at: `2026-01-${String(index + 1).padStart(2, '0')}T06:00:00`,
      forecast_end_at: `2026-01-${String(index + 1).padStart(2, '0')}T08:00:00`,
      model_count: 2
    }));
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      report_id: 'rep1',
      model_metrics: [
        { model_id: 'model-a', metrics: { mse: 0.1 } },
        { model_id: 'model-b', metrics: { mse: 0.2 } }
      ],
      task_summaries: [],
      sample_forecast_links: [links[0], { ...links[0], forecast_artifact_id: 'artifact-duplicate' }, ...links.slice(1)]
    }), { status: 200 }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    await screen.findByText('Showing 1-10 of 12 samples');
    expect(screen.getAllByText('Window #1')).toHaveLength(1);
    expect(screen.queryByText('Window #11')).toBeNull();

    await fireEvent.click(screen.getByRole('button', { name: /Next/ }));

    expect(screen.getByText('Showing 11-12 of 12 samples')).toBeTruthy();
    expect(screen.getByText('Window #11')).toBeTruthy();
    expect(screen.getByText('Window #12')).toBeTruthy();
  });
});
