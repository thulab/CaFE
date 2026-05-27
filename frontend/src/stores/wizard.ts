import { reactive } from 'vue';
import type { UploadPreviewDTO } from '../api/types';
import type { MessageState } from '../lib/errors';

export const wizardState = reactive({
  step: 0,
  preview: null as UploadPreviewDTO | null,
  sourceUri: '',
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
  wizardState.step = 0;
  wizardState.preview = null;
  wizardState.sourceUri = '';
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
