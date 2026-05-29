import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import EvaluationWizardPage from '../pages/EvaluationWizardPage.vue';
import UploadStep from '../components/wizard/UploadStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

describe('EvaluationWizardPage entry modes', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('starts with separate upload and existing-track choices', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ items: [] }));

    render(EvaluationWizardPage, { global: { plugins: [i18n] } });

    expect(screen.getByRole('button', { name: /Upload data/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Choose existing track/ })).toBeTruthy();
    expect(screen.queryByText('Drop a CSV or TsFile here or browse')).toBeNull();

    await fireEvent.click(screen.getByRole('button', { name: /Upload data/ }));

    expect(wizardState.entryMode).toBe('upload');
    expect(await screen.findByText('Drop a CSV or TsFile here or browse')).toBeTruthy();
  });

  it('opens the existing-track run panel without rendering the upload wizard', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url === '/api/tracks') {
        return jsonResponse({ items: [{ track_id: 'track-1', name: 'Hourly energy', primary_metric_id: 'mase', status: 'ready' }] });
      }
      if (url === '/api/models') return jsonResponse({ items: [{ model_id: 'model-1', name: 'Timer 3.5' }] });
      return jsonResponse({});
    });

    render(EvaluationWizardPage, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByRole('button', { name: /Choose existing track/ }));

    expect(wizardState.entryMode).toBe('existing-track');
    expect(await screen.findByText(/Hourly energy/)).toBeTruthy();
    expect(screen.queryByText('Drop a CSV or TsFile here or browse')).toBeNull();
  });
});

describe('UploadStep naming defaults', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    vi.restoreAllMocks();
  });

  it('defaults dataset, shard, and track names from the uploaded file name', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      upload_id: 'upload-1',
      source_uri: '/tmp/hourly-energy.csv',
      filename: 'hourly-energy.csv',
      columns: [{ name: 'time' }, { name: 'target' }],
      preview_rows: []
    }));
    render(UploadStep, { global: { plugins: [i18n] } });

    await fireEvent.change(screen.getByLabelText('Data file'), { target: { files: [new File(['x'], 'hourly-energy.csv')] } });

    await waitFor(() => expect(wizardState.datasetName).toBe('hourly-energy'));
    expect(wizardState.shardName).toBe('hourly-energy shard');
    expect(wizardState.trackName).toBe('hourly-energy track');
  });
});
