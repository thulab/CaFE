import { fireEvent, render, screen, within } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import ReportPage from '../pages/ReportPage.vue';

describe('ReportPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.location.hash = '#/reports/rep1';
    setLocale('en-US');
  });

  it('shows model metrics and sample forecast links without the task outcomes card', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_sort=sample_index') {
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

    expect((await screen.findAllByText('m1')).length).toBeGreaterThan(0);
    expect(screen.queryByText('Task outcomes')).toBeNull();
    expect(screen.queryByText('boom')).toBeNull();
    expect(screen.getByText('Window #1')).toBeTruthy();
    expect(screen.getByText('Forecast rows 6-8')).toBeTruthy();
    expect(screen.getByRole('link', { name: /Open/ }).getAttribute('href')).toBe('#/samples/s1?run_id=r1&report_id=rep1&sample_link_offset=0&sample_cursor_offset=0');
    expect(screen.getByRole('link', { name: 'Back to track' }).getAttribute('href')).toBe('#/tracks/track-1');
  });

  it('initializes sample forecast controls from the report URL query', async () => {
    window.location.hash = '#/reports/rep1?sample_link_offset=20&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_asc';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      report_id: 'rep1',
      model_metrics: [{ model_id: 'model-a', model_name: 'Timer A', metrics: { mse: 0.2 } }],
      task_summaries: [],
      capability_blocks: [{ capability_block_id: 'block-a', name: 'Trend set', block_type: 'synthetic', capability_type: 'trend', capability_label: 'Trend', sample_count: 1 }],
      sample_forecast_links: [{ sample_id: 's21', run_id: 'r1', sample_index: 20, capability_block_id: 'block-a', metric_value: 0.2, model_count: 1 }],
      sample_forecast_links_total: 30,
      sample_forecast_links_limit: 10,
      sample_forecast_links_offset: 20,
      sample_forecast_links_capability_block_id: 'block-a',
      sample_forecast_links_model_id: 'model-a',
      sample_forecast_links_metric: 'mse',
      sample_forecast_links_sort: 'metric_asc',
    }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Window #21')).toBeTruthy();
    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe('/api/reports/rep1?sample_link_limit=10&sample_link_offset=20&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_asc');
    expect((screen.getByLabelText('Test group') as HTMLSelectElement).value).toBe('block-a');
    expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('model-a');
    expect((screen.getByLabelText('Sort') as HTMLSelectElement).value).toBe('metric_asc');
    expect(screen.getByRole('link', { name: /Open/ }).getAttribute('href')).toBe(
      '#/samples/s21?run_id=r1&report_id=rep1&sample_link_offset=20&sample_cursor_offset=20&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_metric=mse&sample_link_sort=metric_asc'
    );
  });

  it('reloads sample forecasts by test group and metric-error sort', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      const base = {
        report_id: 'rep1',
        model_metrics: [{ model_id: 'model-a', metrics: { mse: 0.2 } }],
        task_summaries: [],
        capability_blocks: [
          { capability_block_id: 'block-a', name: 'Trend set', block_type: 'synthetic', capability_type: 'trend', capability_label: 'Trend', sample_count: 2 },
          { capability_block_id: 'block-b', name: 'Real set', block_type: 'real', capability_type: 'real_data', capability_label: 'Real data', sample_count: 1 },
        ],
        sample_forecast_links_metric: 'mse',
        sample_forecast_links_limit: 10,
        sample_forecast_links_offset: 0,
      };
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_sort=sample_index') {
        return Promise.resolve(jsonResponse({
          ...base,
          sample_forecast_links: [
            { sample_id: 's1', run_id: 'r1', sample_index: 0, capability_block_id: 'block-a', metric_value: 0.3, model_count: 1 },
            { sample_id: 's2', run_id: 'r1', sample_index: 1, capability_block_id: 'block-b', metric_value: 0.8, model_count: 1 },
          ],
          sample_forecast_links_total: 2,
          sample_forecast_links_sort: 'sample_index',
        }));
      }
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_sort=sample_index') {
        return Promise.resolve(jsonResponse({
          ...base,
          sample_forecast_links: [{ sample_id: 's1', run_id: 'r1', sample_index: 0, capability_block_id: 'block-a', metric_value: 0.3, model_count: 1 }],
          sample_forecast_links_total: 1,
          sample_forecast_links_capability_block_id: 'block-a',
          sample_forecast_links_sort: 'sample_index',
        }));
      }
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=sample_index') {
        return Promise.resolve(jsonResponse({
          ...base,
          sample_forecast_links: [{ sample_id: 's1', run_id: 'r1', sample_index: 0, capability_block_id: 'block-a', metric_model_id: 'model-a', metric_value: 0.3, model_count: 1 }],
          sample_forecast_links_total: 1,
          sample_forecast_links_capability_block_id: 'block-a',
          sample_forecast_links_model_id: 'model-a',
          sample_forecast_links_sort: 'sample_index',
        }));
      }
      if (url === '/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_desc') {
        return Promise.resolve(jsonResponse({
          ...base,
          sample_forecast_links: [{ sample_id: 's3', run_id: 'r1', sample_index: 2, capability_block_id: 'block-a', metric_model_id: 'model-a', metric_value: 0.9, model_count: 1 }],
          sample_forecast_links_total: 1,
          sample_forecast_links_capability_block_id: 'block-a',
          sample_forecast_links_model_id: 'model-a',
          sample_forecast_links_sort: 'metric_desc',
        }));
      }
      return Promise.reject(new Error(`unexpected URL ${url}`));
    });

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    await screen.findByText('Window #1');

    await fireEvent.update(screen.getByLabelText('Test group'), 'block-a');
    expect((await screen.findAllByText('Showing 1-1 of 1 samples')).length).toBeGreaterThan(0);
    expect(String(fetchSpy.mock.calls.at(-1)?.[0])).toBe('/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_sort=sample_index');
    expect(window.location.hash).toBe('#/reports/rep1?sample_link_capability_block_id=block-a');

    await fireEvent.update(screen.getByLabelText('Model'), 'model-a');
    expect(String(fetchSpy.mock.calls.at(-1)?.[0])).toBe('/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=sample_index');
    expect(window.location.hash).toBe('#/reports/rep1?sample_link_capability_block_id=block-a&sample_link_model_id=model-a');

    await fireEvent.update(screen.getByLabelText('Sort'), 'metric_desc');
    expect(await screen.findByText('Window #3')).toBeTruthy();
    expect(String(fetchSpy.mock.calls.at(-1)?.[0])).toBe('/api/reports/rep1?sample_link_limit=10&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_desc');
    expect(window.location.hash).toBe('#/reports/rep1?sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_desc');
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

    expect((await screen.findAllByText('model-a')).length).toBeGreaterThan(0);
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

  it('renders capability radar and all-test-group breakdown from report capability metrics', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      report_id: 'rep1',
      model_metrics: [
        { model_id: 'model-a', model_name: 'Timer A', metrics: { mase: 0.4 } },
        { model_id: 'model-b', model_name: 'Timer B', metrics: { mase: 0.3 } },
      ],
      capability_blocks: [
        { capability_block_id: 'block-trend', name: 'Trend set', block_type: 'synthetic', capability_type: 'trend', capability_label: 'Trend', sample_count: 20 },
        { capability_block_id: 'block-season', name: 'Season set', block_type: 'synthetic', capability_type: 'multi_seasonal', capability_label: 'Multi-seasonal', sample_count: 20 },
        { capability_block_id: 'block-cov', name: 'Covariate set', block_type: 'synthetic', capability_type: 'covariate_response', capability_label: 'Covariate response', sample_count: 20 },
        { capability_block_id: 'block-real', name: 'Real validation', block_type: 'real', capability_type: 'real_data', capability_label: 'Real data', sample_count: 12 },
      ],
      capability_metrics: [
        { task_id: 'ta', unit_id: 'ua', model_id: 'model-a', model_name: 'Timer A', capability_block_id: 'block-trend', status: 'succeeded', sample_count: 20, metrics: { mase: 0.2 } },
        { task_id: 'tb', unit_id: 'ub', model_id: 'model-b', model_name: 'Timer B', capability_block_id: 'block-trend', status: 'succeeded', sample_count: 20, metrics: { mase: 0.4 } },
        { task_id: 'tc', unit_id: 'ua', model_id: 'model-a', model_name: 'Timer A', capability_block_id: 'block-season', status: 'succeeded', sample_count: 20, metrics: { mase: 0.5 } },
        { task_id: 'td', unit_id: 'ub', model_id: 'model-b', model_name: 'Timer B', capability_block_id: 'block-season', status: 'succeeded', sample_count: 20, metrics: { mase: 0.3 } },
        { task_id: 'te', unit_id: 'ua', model_id: 'model-a', model_name: 'Timer A', capability_block_id: 'block-cov', status: 'succeeded', sample_count: 20, metrics: { mase: 0.6 } },
        { task_id: 'tf', unit_id: 'ub', model_id: 'model-b', model_name: 'Timer B', capability_block_id: 'block-cov', status: 'succeeded', sample_count: 20, metrics: { mase: 0.4 } },
        { task_id: 'tg', unit_id: 'ua', model_id: 'model-a', model_name: 'Timer A', capability_block_id: 'block-real', status: 'succeeded', sample_count: 12, metrics: { mase: 0.7 } },
        { task_id: 'th', unit_id: 'ub', model_id: 'model-b', model_name: 'Timer B', capability_block_id: 'block-real', status: 'succeeded', sample_count: 12, metrics: { mase: 0.8 } },
      ],
      task_summaries: [],
      sample_forecast_links: []
    }), { status: 200 }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Capability profile')).toBeTruthy();
    expect(screen.getByRole('img', { name: /Capability radar/ })).toBeTruthy();
    expect(screen.getAllByText('Trend').length).toBeGreaterThan(0);
    expect(screen.queryByRole('cell', { name: /Real validation/ })).toBeNull();

    await fireEvent.update(screen.getByLabelText('Scope'), 'all');

    expect(screen.getByRole('cell', { name: /Real validation/ })).toBeTruthy();
    expect(screen.getAllByText('Real data').length).toBeGreaterThan(0);
  });

  it('scores MASE capability values without forcing the largest error to zero', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      report_id: 'rep1',
      model_metrics: [
        { model_id: 'model-best', model_name: 'Timer Best', metrics: { mase: 0.7612 } },
        { model_id: 'model-slow', model_name: 'Timer Slow', metrics: { mase: 1.3543 } },
      ],
      capability_blocks: [
        { capability_block_id: 'block-trend', name: 'Trend set', block_type: 'synthetic', capability_type: 'trend', capability_label: 'Trend', sample_count: 20 },
        { capability_block_id: 'block-season', name: 'Season set', block_type: 'synthetic', capability_type: 'multi_seasonal', capability_label: 'Multi-seasonal', sample_count: 20 },
        { capability_block_id: 'block-cov', name: 'Covariate set', block_type: 'synthetic', capability_type: 'covariate_response', capability_label: 'Covariate response', sample_count: 20 },
      ],
      capability_metrics: [
        { task_id: 't1', unit_id: 'u1', model_id: 'model-best', model_name: 'Timer Best', capability_block_id: 'block-trend', status: 'succeeded', sample_count: 20, metrics: { mase: 0.7612 } },
        { task_id: 't2', unit_id: 'u2', model_id: 'model-slow', model_name: 'Timer Slow', capability_block_id: 'block-trend', status: 'succeeded', sample_count: 20, metrics: { mase: 1.3543 } },
        { task_id: 't3', unit_id: 'u1', model_id: 'model-best', model_name: 'Timer Best', capability_block_id: 'block-season', status: 'succeeded', sample_count: 20, metrics: { mase: 0.7612 } },
        { task_id: 't4', unit_id: 'u2', model_id: 'model-slow', model_name: 'Timer Slow', capability_block_id: 'block-season', status: 'succeeded', sample_count: 20, metrics: { mase: 1.3543 } },
        { task_id: 't5', unit_id: 'u1', model_id: 'model-best', model_name: 'Timer Best', capability_block_id: 'block-cov', status: 'succeeded', sample_count: 20, metrics: { mase: 0.7612 } },
        { task_id: 't6', unit_id: 'u2', model_id: 'model-slow', model_name: 'Timer Slow', capability_block_id: 'block-cov', status: 'succeeded', sample_count: 20, metrics: { mase: 1.3543 } },
      ],
      task_summaries: [],
      sample_forecast_links: [],
    }));

    render(ReportPage, { props: { reportId: 'rep1' }, global: { plugins: [i18n] } });

    expect(await screen.findByText('Capability profile')).toBeTruthy();
    expect(screen.getByText('57 / 100')).toBeTruthy();
    expect(screen.getByText('42 / 100')).toBeTruthy();
    expect(screen.queryByText('0 / 100')).toBeNull();
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

    expect((await screen.findAllByText('Showing 1-10 of 12 samples')).length).toBeGreaterThan(0);
    expect(screen.getAllByText('Window #1')).toHaveLength(1);
    expect(screen.queryByText('Window #11')).toBeNull();

    await fireEvent.click(screen.getByRole('button', { name: /Next/ }));

    expect(screen.getAllByText('Showing 11-12 of 12 samples').length).toBeGreaterThan(0);
    expect(screen.getByText('Window #11')).toBeTruthy();
    expect(screen.getByText('Window #12')).toBeTruthy();
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}
