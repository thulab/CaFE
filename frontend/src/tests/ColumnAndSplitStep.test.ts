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

    // Value columns default all checked; context=0 triggers split validation
    await fireEvent.update(screen.getByLabelText('Context'), '0');
    // Target not selected (default is empty) → should trigger target error
    await fireEvent.click(screen.getByRole('button', { name: 'Load shard' }));

    expect(screen.getByRole('alert').textContent).toContain('Select exactly one target');
  });

  it('creates manifest and load job with valid config', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep);

    // Select a target from the single-select dropdown
    const targetSelect = screen.getByLabelText('Target');
    await fireEvent.update(targetSelect, 'target');

    await fireEvent.click(screen.getByRole('button', { name: 'Load shard' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));

    // Assert manifest payload uses value_columns
    const manifestCall = fetchSpy.mock.calls[0];
    const manifestBody = JSON.parse(manifestCall[1].body as string);
    expect(manifestBody.value_columns).toContain('target');
    expect(manifestBody.value_columns).toContain('other');
    expect(manifestBody).not.toHaveProperty('target_columns');

    // Assert load job payload has split_config.target_columns
    const loadJobCall = fetchSpy.mock.calls[1];
    const loadJobBody = JSON.parse(loadJobCall[1].body as string);
    expect(loadJobBody.split_config.target_columns).toEqual(['target']);
  });
});
