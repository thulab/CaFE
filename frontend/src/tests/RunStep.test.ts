import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import RunStep from '../components/wizard/RunStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

describe('RunStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    wizardState.trackId = 'track-1';
    vi.useFakeTimers();
    vi.restoreAllMocks();
  });

  it('requires at least one selected model', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
    render(RunStep, { global: { plugins: [i18n] } });

    await screen.findByText('Timer 3.5');
    expect((screen.getByRole('button', { name: 'Run' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('polls progress and stops on terminal status', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs/r1/progress') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'succeeded', progress: {}, units: [], tasks: [], recent_events: [], report_id: 'rep1' }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });
    render(RunStep, { global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.5'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => expect(wizardState.runId).toBe('r1'));

    await waitFor(() => expect(wizardState.reportId).toBe('rep1'));
  });

  it('shows activity status from progress immediately after starting a run', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs/r1/progress') {
        return Promise.resolve(new Response(JSON.stringify({
          benchmarking_run_id: 'r1',
          status: 'running',
          activity_status: 'model_loading',
          progress: { total_models: 1, completed_models: 0, total_tasks: 1, completed_tasks: 0, total_samples: 4, processed_samples: 0, completed_samples: 0, failed_samples: 0 },
          units: [],
          tasks: [],
          recent_events: []
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });
    render(RunStep, { global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.5'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    expect(await screen.findByText('Loading model')).toBeTruthy();
  });

  it('starts a run without preloading unloaded timer-service models', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      calls.push(url);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.0', adapter_type: 'timer_service', loaded: false }] }), { status: 200 }));
      }
      if (url === '/api/models/m1/load') {
        return Promise.resolve(new Response(JSON.stringify({ error: 'preload should not be called' }), { status: 500 }));
      }
      if (url === '/api/benchmarking-runs') {
        return Promise.resolve(new Response(JSON.stringify({ benchmarking_run_id: 'r1', status: 'running' }), { status: 200 }));
      }
      if (url === '/api/benchmarking-runs/r1/progress') {
        return Promise.resolve(new Response(JSON.stringify({
          benchmarking_run_id: 'r1',
          status: 'running',
          progress: { total_models: 1, completed_models: 0, total_tasks: 1, completed_tasks: 0, total_samples: 4, processed_samples: 0, completed_samples: 0, failed_samples: 0 },
          units: [],
          tasks: [],
          recent_events: []
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });
    render(RunStep, { global: { plugins: [i18n] } });

    await fireEvent.click(await screen.findByLabelText('Timer 3.0'));
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => expect(wizardState.runId).toBe('r1'));
    expect(calls).not.toContain('/api/models/m1/load');
    expect(calls).toContain('/api/benchmarking-runs');
  });

  it('disables single-target models for a multi-target test case set', async () => {
    wizardState.selectedShardIds = ['shard-1'];
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({
          items: [
            { model_id: 'timer', name: 'Timer 3.0', adapter_type: 'timer_service', forecast_limits: { max_target_count: 1 } },
            { model_id: 'toto', name: 'toto2.0', adapter_type: 'timer_service', forecast_limits: { max_target_count: null } }
          ]
        }), { status: 200 }));
      }
      if (url === '/api/shards/shard-1') {
        return Promise.resolve(new Response(JSON.stringify({
          shard_id: 'shard-1',
          dataset_manifest_id: 'manifest-1',
          source_uri: '/tmp/multi.csv',
          status: 'ready',
          row_count: 20,
          target_columns: ['load', 'temperature'],
          target_dim: 2,
          context_length: 6,
          horizon: 3,
          stride: 3,
          sample_count: 4
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });

    render(RunStep, { global: { plugins: [i18n] } });

    const timer = await screen.findByLabelText('Timer 3.0') as HTMLInputElement;
    const toto = await screen.findByLabelText('toto2.0') as HTMLInputElement;
    await waitFor(() => expect(timer.disabled).toBe(true));
    expect(toto.disabled).toBe(false);
    expect(screen.getByText(/supports up to 1 target/)).toBeTruthy();
  });

  it('disables models without covariate capacity for a covariate test case set', async () => {
    wizardState.selectedShardIds = ['shard-1'];
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({
          items: [
            { model_id: 'timer', name: 'Timer 3.0', adapter_type: 'timer_service', forecast_limits: { max_target_count: 1, max_covariate_count: 0 } },
            { model_id: 'chronos', name: 'Chronos-2', adapter_type: 'timer_service', forecast_limits: { max_target_count: 1, max_covariate_count: 50 } }
          ]
        }), { status: 200 }));
      }
      if (url === '/api/shards/shard-1') {
        return Promise.resolve(new Response(JSON.stringify({
          shard_id: 'shard-1',
          dataset_manifest_id: 'manifest-1',
          source_uri: '/tmp/cov.csv',
          status: 'ready',
          row_count: 20,
          target_columns: ['load'],
          target_dim: 1,
          covariate_columns: ['promo', 'temperature'],
          covariate_dim: 2,
          context_length: 6,
          horizon: 3,
          stride: 3,
          sample_count: 4
        }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, limit: 1, offset: 0 }), { status: 200 }));
    });

    render(RunStep, { global: { plugins: [i18n] } });

    const timer = await screen.findByLabelText('Timer 3.0') as HTMLInputElement;
    const chronos = await screen.findByLabelText('Chronos-2') as HTMLInputElement;
    await waitFor(() => expect(timer.disabled).toBe(true));
    expect(chronos.disabled).toBe(false);
    expect(screen.getByText(/supports 0 covariates/)).toBeTruthy();
  });
});
