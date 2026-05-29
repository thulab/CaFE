import { reactive } from 'vue';
import type { UploadPreviewDTO } from '../api/types';
import type { MessageState } from '../lib/errors';

export type WizardEntryMode = '' | 'upload' | 'existing-track';

export const wizardState = reactive({
  entryMode: '' as WizardEntryMode,
  step: 0,
  preview: null as UploadPreviewDTO | null,
  sourceUri: '',
  datasetName: '',
  shardName: '',
  trackName: '',
  primaryMetric: 'mase',
  manifestId: '',
  loadJobId: '',
  shardId: '',
  trackId: '',
  capabilityBlockId: '',
  rankingListId: '',
  runId: '',
  reportId: '',
  error: null as MessageState
});

export const STEP_COUNT = 6;

export function goToStep(index: number) {
  wizardState.step = Math.max(0, Math.min(STEP_COUNT - 1, index));
}

export function goNext() {
  goToStep(wizardState.step + 1);
}

export function goPrev() {
  goToStep(wizardState.step - 1);
}

export function resetWizard() {
  wizardState.entryMode = '';
  wizardState.step = 0;
  wizardState.preview = null;
  wizardState.sourceUri = '';
  wizardState.datasetName = '';
  wizardState.shardName = '';
  wizardState.trackName = '';
  wizardState.primaryMetric = 'mase';
  wizardState.manifestId = '';
  wizardState.loadJobId = '';
  wizardState.shardId = '';
  wizardState.trackId = '';
  wizardState.capabilityBlockId = '';
  wizardState.rankingListId = '';
  wizardState.runId = '';
  wizardState.reportId = '';
  wizardState.error = null;
}

export function defaultNameFromFilename(filename?: string | null) {
  const clean = (filename || '').split(/[\\/]/).pop()?.trim() || '';
  if (!clean) return 'Uploaded dataset';
  const dot = clean.lastIndexOf('.');
  return dot > 0 ? clean.slice(0, dot) : clean;
}

export function applyUploadNameDefaults(filename?: string | null) {
  const base = defaultNameFromFilename(filename);
  wizardState.datasetName = base;
  wizardState.shardName = `${base} shard`;
  wizardState.trackName = `${base} track`;
}
