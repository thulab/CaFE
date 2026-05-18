import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import UploadStep from '../components/wizard/UploadStep.vue';
import { resetWizard } from '../stores/wizard';

describe('UploadStep', () => {
  beforeEach(() => {
    resetWizard();
    vi.restoreAllMocks();
  });

  it('blocks next before preview and displays preview rows after upload', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      upload_id: 'u1',
      source_uri: '/tmp/data.csv',
      columns: [{ name: 'time' }, { name: 'target' }],
      preview_rows: [{ time: '2026-01-01', target: '1' }]
    }), { status: 200 }));
    render(UploadStep);

    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.change(screen.getByLabelText('CSV file'), { target: { files: [new File(['x'], 'data.csv')] } });

    await screen.findByText('target');
    expect(screen.getByText('2026-01-01')).toBeTruthy();
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('displays API errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ error_code: 'bad', message: 'Bad CSV', details: {} }), { status: 400 }));
    render(UploadStep);

    await fireEvent.change(screen.getByLabelText('CSV file'), { target: { files: [new File(['x'], 'bad.csv')] } });

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Bad CSV'));
  });
});
