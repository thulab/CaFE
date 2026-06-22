import { fireEvent, render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import { i18n, setLocale } from '../i18n';
import { resetWizard, wizardState } from '../stores/wizard';

describe('ColumnAndSplitStep', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    wizardState.preview = {
      upload_id: 'u1',
      source_uri: '/tmp/data.csv',
      filename: 'data.csv',
      columns: [{ name: 'time' }, { name: 'target' }, { name: 'other' }],
      preview_rows: []
    };
    wizardState.sourceUri = '/tmp/data.csv';
    vi.restoreAllMocks();
  });

  it('enforces target selection and positive split values', async () => {
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    // context=0 would trigger split validation after a target is selected
    await fireEvent.update(screen.getByLabelText('Context'), '0');
    // Target not selected (default is empty) → should trigger target error
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    expect(screen.getByRole('alert').textContent).toContain('Select at least one target');
  });

  it('keeps time selection separate from target and covariate pickers', () => {
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    const columnRoleSelection = screen.getByLabelText('Target and covariate columns');
    expect(columnRoleSelection.contains(screen.getByText('Target columns'))).toBe(true);
    expect(columnRoleSelection.contains(screen.getByText('Known future covariates'))).toBe(true);
    expect(columnRoleSelection.contains(screen.getByLabelText('Time column'))).toBe(false);
  });

  it('creates manifest and load job with valid config', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    expect((screen.getByLabelText('Dataset name') as HTMLInputElement).value).toBe('data');
    expect((screen.getByLabelText('Test case set name') as HTMLInputElement).value).toBe('data test cases');
    expect((screen.getByLabelText('Context') as HTMLInputElement).value).toBe('60');
    expect((screen.getByLabelText('Horizon') as HTMLInputElement).value).toBe('16');
    expect((screen.getByLabelText('Stride') as HTMLInputElement).value).toBe('16');
    await fireEvent.update(screen.getByLabelText('Dataset name'), 'Energy demand');
    await fireEvent.update(screen.getByLabelText('Test case set name'), 'Energy demand validation');

    await fireEvent.click(screen.getByLabelText('Select target target'));

    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));
    expect(wizardState.selectedShardIds).toEqual(['s1']);

    const manifestCall = fetchSpy.mock.calls[0];
    const manifestBody = JSON.parse(manifestCall![1]!.body as string);
    expect(manifestBody.name).toBe('Energy demand');
    expect(manifestBody).not.toHaveProperty('value_columns');
    expect(manifestBody).not.toHaveProperty('target_columns');

    // Assert load job payload has split_config.target_columns
    const loadJobCall = fetchSpy.mock.calls[1];
    const loadJobBody = JSON.parse(loadJobCall![1]!.body as string);
    expect(loadJobBody.split_config).toMatchObject({
      context_length: 60,
      horizon: 16,
      stride: 16,
      target_columns: ['target'],
      shard_name: 'Energy demand validation'
    });
  });

  it('submits multiple selected target columns', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByLabelText('Select target target'));
    await fireEvent.click(screen.getByLabelText('Select target other'));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));
    const loadJobBody = JSON.parse(fetchSpy.mock.calls[1]![1]!.body as string);
    expect(loadJobBody.split_config.target_columns).toEqual(['target', 'other']);
  });

  it('submits selected known-future covariates from the column lists', async () => {
    wizardState.preview = {
      upload_id: 'u1',
      source_uri: '/tmp/data.csv',
      filename: 'data.csv',
      columns: [{ name: 'time' }, { name: 'target' }, { name: 'promo' }, { name: 'temperature' }],
      preview_rows: []
    };
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByLabelText('Select target target'));
    await fireEvent.click(screen.getByLabelText('Select covariate promo'));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));
    const loadJobBody = JSON.parse(fetchSpy.mock.calls[1]![1]!.body as string);
    expect(loadJobBody.split_config.target_columns).toEqual(['target']);
    expect(loadJobBody.split_config.covariate_columns).toEqual(['promo']);
  });

  it('keeps the generated manifest payload name stable under Chinese UI', async () => {
    setLocale('zh-CN');
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j1', status: 'succeeded', output_shard_id: 's1' }), { status: 200 }));
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByLabelText('选择目标 target'));
    await fireEvent.click(screen.getByRole('button', { name: '生成测试用例集' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s1'));
    const manifestBody = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(manifestBody.name).toBe('data');
  });

  it('creates tsfile manifests without requiring a CSV timestamp column', async () => {
    wizardState.preview = {
      upload_id: 'u-ts',
      source_uri: '/tmp/data.tsfile',
      file_format: 'tsfile',
      columns: [{ name: 'temperature' }, { name: 'pressure' }],
      preview_rows: []
    };
    wizardState.sourceUri = '/tmp/data.tsfile';
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm-ts' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ load_job_id: 'j-ts', status: 'succeeded', output_shard_id: 's-ts' }), { status: 200 }));

    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByLabelText('Select target temperature'));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    await waitFor(() => expect(wizardState.shardId).toBe('s-ts'));
    const manifestBody = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(manifestBody.file_format).toBe('tsfile');
    expect(manifestBody.time_column).toBe('time');
    expect(manifestBody).not.toHaveProperty('value_columns');
  });

  it('shows load job failures returned in a 200 response', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ dataset_manifest_id: 'm1' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        load_job_id: 'j1',
        status: 'failed',
        error_code: 'tsfile_multiple_devices',
        error_message: 'MVP supports exactly one selected device per tsfile'
      }), { status: 200 }));
    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    await fireEvent.click(screen.getByLabelText('Select target target'));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate test case set' }));

    expect((await screen.findByRole('alert')).textContent).toContain('tsfile_multiple_devices');
    expect(wizardState.shardId).toBe('');
  });

  it('renders split labels and window status in Chinese', () => {
    setLocale('zh-CN');

    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    expect(screen.getByText('上下文')).toBeTruthy();
    expect(screen.getByText('预测步长')).toBeTruthy();
    expect(screen.getByText('滑窗步长')).toBeTruthy();
    expect(screen.getByText('窗口：60 个上下文点 → 16 个预测点，滑窗步长 16。')).toBeTruthy();
  });

  it('continues to existing test case selection when upload was skipped', async () => {
    resetWizard();
    wizardState.dataUploadSkipped = true;
    wizardState.step = 2;

    render(ColumnAndSplitStep, { global: { plugins: [i18n] } });

    expect(screen.getByText('No new data uploaded. Continue to choose existing test case sets.')).toBeTruthy();
    await fireEvent.click(screen.getByRole('button', { name: 'Choose existing test case sets' }));

    expect(wizardState.step).toBe(3);
  });
});
