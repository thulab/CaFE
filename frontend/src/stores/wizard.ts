import { reactive } from 'vue';
import type { UploadPreviewDTO } from '../api/types';
import type { ModelDTO } from '../api/models';

export const wizardState = reactive({
  preview: null as UploadPreviewDTO | null,
  sourceUri: '',
  manifestId: '',
  shardId: '',
  trackId: '',
  runId: '',
  reportId: '',
  selectedModels: [] as ModelDTO[],
  error: ''
});

export function resetWizard() {
  wizardState.preview = null;
  wizardState.sourceUri = '';
  wizardState.manifestId = '';
  wizardState.shardId = '';
  wizardState.trackId = '';
  wizardState.runId = '';
  wizardState.reportId = '';
  wizardState.selectedModels = [];
  wizardState.error = '';
}
