import { render, screen } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import SampleForecastPage from '../pages/SampleForecastPage.vue';

describe('SampleForecastPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.location.hash = '';
    setLocale('en-US');
  });

  it('shows history, ground truth, multiple forecasts, failed status, and metrics', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      sample_id: 's1',
      sample_index: 0,
      target_history: [[1], [2]],
      target_future: [[3], [4]],
      links: { report: 'rep1', run: 'r1', ranking: 'rank1' },
      models: [
        { model_id: 'm1', model_name: 'Timer', status: 'succeeded', forecast: [[3.1], [3.9]], metrics: { mse: 0.01, mae: 0.1 } },
        { model_id: 'm2', model_name: 'Chronos', status: 'failed', forecast: null, metrics: {}, error_message: 'failed' }
      ]
    }), { status: 200 }));

    render(SampleForecastPage, { props: { sampleId: 's1', runId: 'r1' }, global: { plugins: [i18n] } });

    // "Timer" appears both in the chart legend and the metric table.
    expect((await screen.findAllByText('Timer')).length).toBeGreaterThan(0);
    expect(screen.getByRole('img', { name: 'Forecast chart with 2 history steps, 2 ground-truth steps and 1 model forecast.' })).toBeTruthy();
    expect(screen.getByText('Ground truth')).toBeTruthy();
    expect(screen.getByText('History')).toBeTruthy();
    expect(screen.getByText('Chronos')).toBeTruthy();
    expect(screen.getByText('failed')).toBeTruthy();
    expect(screen.getByText('0.01')).toBeTruthy();
    expect(screen.getByRole('link', { name: /Back to report/ }).getAttribute('href')).toBe('#/reports/rep1');
    expect(screen.getByText('#1')).toBeTruthy();
  });

  it('uses singular labels for one history step and one ground-truth step', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      sample_id: 's1',
      target_history: [[1]],
      target_future: [[2]],
      models: [
        { model_id: 'm1', model_name: 'Timer', status: 'succeeded', forecast: [[2.1]], metrics: {} }
      ]
    }), { status: 200 }));

    render(SampleForecastPage, { props: { sampleId: 's1', runId: 'r1' }, global: { plugins: [i18n] } });

    expect(await screen.findByRole('img', { name: 'Forecast chart with 1 history step, 1 ground-truth step and 1 model forecast.' })).toBeTruthy();
  });

  it('renders a separate covariate chart when covariates are present', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      sample_id: 's1',
      target_history: [[1], [2]],
      target_future: [[3]],
      covariate_column_names: ['promo', 'temperature'],
      history_cov: [[0, 21], [1, 22]],
      future_cov: [[1, 23]],
      models: [
        { model_id: 'm1', model_name: 'Chronos', status: 'succeeded', forecast: [[3.1]], metrics: {} }
      ]
    }), { status: 200 }));

    render(SampleForecastPage, { props: { sampleId: 's1', runId: 'r1' }, global: { plugins: [i18n] } });

    expect(await screen.findByRole('img', { name: 'Covariate chart with 2 history steps, 1 future step and 2 covariates.' })).toBeTruthy();
    expect(screen.getByText('Covariates')).toBeTruthy();
    expect(screen.getByText('promo')).toBeTruthy();
  });

  it('links to previous and next forecast samples in the same test case set', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/samples/s2/forecast?run_id=r1') {
        return new Response(JSON.stringify({
          sample_id: 's2',
          shard_id: 'shard-1',
          sample_index: 1,
          target_history: [[1]],
          target_future: [[2]],
          models: [
            { model_id: 'm1', model_name: 'Timer', status: 'succeeded', forecast: [[2.1]], metrics: {} }
          ],
          links: { report: 'rep1' }
        }), { status: 200 });
      }
      if (url === '/api/shards/shard-1/samples?limit=1&offset=0') {
        return new Response(JSON.stringify({ items: [{ sample_id: 's1', sample_index: 0 }], total: 3, limit: 1, offset: 0 }), { status: 200 });
      }
      if (url === '/api/shards/shard-1/samples?limit=1&offset=2') {
        return new Response(JSON.stringify({ items: [{ sample_id: 's3', sample_index: 2 }], total: 3, limit: 1, offset: 2 }), { status: 200 });
      }
      return new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 });
    });

    render(SampleForecastPage, { props: { sampleId: 's2', runId: 'r1', reportId: 'rep1' }, global: { plugins: [i18n] } });

    expect((await screen.findByRole('link', { name: 'Previous sample' })).getAttribute('href')).toBe('#/samples/s1?run_id=r1&report_id=rep1');
    expect(screen.getByRole('link', { name: 'Next sample' }).getAttribute('href')).toBe('#/samples/s3?run_id=r1&report_id=rep1');
  });

  it('uses report sorting context for previous and next forecast samples', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/samples/s-current/forecast?run_id=r1') {
        return new Response(JSON.stringify({
          sample_id: 's-current',
          shard_id: 'shard-1',
          sample_index: 7,
          target_history: [[1]],
          target_future: [[2]],
          models: [
            { model_id: 'model-a', model_name: 'Timer A', status: 'succeeded', forecast: [[2.1]], metrics: { mse: 0.7 } }
          ],
          links: { report: 'rep1' }
        }), { status: 200 });
      }
      if (url === '/api/reports/rep1?sample_link_limit=1&sample_link_offset=1&sample_link_capability_block_id=block-a&sample_link_metric=mse&sample_link_model_id=model-a&sample_link_sort=metric_desc') {
        return new Response(JSON.stringify({
          report_id: 'rep1',
          model_metrics: [],
          task_summaries: [],
          sample_forecast_links: [{ sample_id: 's-prev', run_id: 'r1', sample_index: 4 }],
          sample_forecast_links_total: 4,
          sample_forecast_links_limit: 1,
          sample_forecast_links_offset: 1,
        }), { status: 200 });
      }
      if (url === '/api/reports/rep1?sample_link_limit=1&sample_link_offset=3&sample_link_capability_block_id=block-a&sample_link_metric=mse&sample_link_model_id=model-a&sample_link_sort=metric_desc') {
        return new Response(JSON.stringify({
          report_id: 'rep1',
          model_metrics: [],
          task_summaries: [],
          sample_forecast_links: [{ sample_id: 's-next', run_id: 'r1', sample_index: 9 }],
          sample_forecast_links_total: 4,
          sample_forecast_links_limit: 1,
          sample_forecast_links_offset: 3,
        }), { status: 200 });
      }
      throw new Error(`unexpected URL ${url}`);
    });

    render(SampleForecastPage, {
      props: {
        sampleId: 's-current',
        runId: 'r1',
        reportId: 'rep1',
        sampleLinkOffset: 0,
        sampleCursorOffset: 2,
        sampleCapabilityBlockId: 'block-a',
        sampleModelId: 'model-a',
        sampleMetric: 'mse',
        sampleSort: 'metric_desc',
      },
      global: { plugins: [i18n] },
    });

    expect((await screen.findByRole('link', { name: 'Previous sample' })).getAttribute('href')).toBe(
      '#/samples/s-prev?run_id=r1&report_id=rep1&sample_cursor_offset=1&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_metric=mse&sample_link_sort=metric_desc'
    );
    expect(screen.getByRole('link', { name: 'Next sample' }).getAttribute('href')).toBe(
      '#/samples/s-next?run_id=r1&report_id=rep1&sample_cursor_offset=3&sample_link_offset=0&sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_metric=mse&sample_link_sort=metric_desc'
    );
    expect(screen.getByRole('link', { name: /Back to report/ }).getAttribute('href')).toBe(
      '#/reports/rep1?sample_link_capability_block_id=block-a&sample_link_model_id=model-a&sample_link_sort=metric_desc'
    );
  });
});
