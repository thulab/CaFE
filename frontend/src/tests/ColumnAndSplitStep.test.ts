import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import { resetWizard, wizardState } from '../stores/wizard';

describe('ColumnAndSplitStep', () => {
  beforeEach(() => {
    resetWizard();
    wizardState.preview = {
      upload_id: 'u1',
      source_uri: '/tmp/data.csv',
      columns: [{ name: 'time' }, { name: 'target' }, { name: 'other' }],
      preview_rows: []
    };
    wizardState.sourceUri = '/tmp/data.csv';
    vi.restoreAllMocks();
  });

  it('enforces single target and positive split values', async () => {
    render(ColumnAndSplitStep);

    await fireEvent.update(screen.getByLabelText('Context'), '0');
    await fireEvent.click(screen.getByLabelText('target'));
    await fireEvent.click(screen.getByLabelText('other'));
    await fireEvent.click(screen.getByRole('button', { name: 'Load shard' }));

    expect(screen.getByRole('alert').textContent).toContain('Select exactly one target');
  });

  it('creates manifest and load job with valid config', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep);

    await fireEvent.click(screen.getByLabelText('target'));
    await fireEvent.click(screen.getByRole('button', { name: 'Load shard' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));
  });
});
