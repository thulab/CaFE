import { reactive } from 'vue';
import type { UploadPreviewDTO } from '../api/types';
import type { ModelDTO } from '../api/models';

export const wizardState = reactive({
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
  selectedModels: [] as ModelDTO[],
  error: ''
});

export function resetWizard() {
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
  wizardState.selectedModels = [];
  wizardState.error = '';
}
