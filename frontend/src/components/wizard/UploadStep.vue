<template>
  <section class="step-body">
    <div class="source-mode-grid" :aria-label="t('wizard.uploadStep.sourceMode')">
      <button
        v-for="option in sourceOptions"
        :key="option.value"
        class="source-mode"
        :class="{ 'is-active': wizardState.dataSource === option.value }"
        type="button"
        @click="selectSource(option.value)"
      >
        <span class="entry-icon"><Icon :name="option.icon" :size="20" /></span>
        <span>
          <span class="source-title">{{ option.title }}</span>
          <span class="source-desc">{{ option.description }}</span>
        </span>
      </button>
    </div>

    <template v-if="wizardState.dataSource === 'upload'">
      <label
        class="dropzone"
        :class="{ 'is-drag': dragging }"
        for="data-file"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
      >
        <Icon class="dz-icon" name="upload" :size="28" />
        <div>
          <strong>{{ fileName || t('wizard.uploadStep.dropOrBrowse') }}</strong>
          <p class="field-help" style="margin-top:4px">{{ t('wizard.uploadStep.help') }}</p>
        </div>
        <input id="data-file" :aria-label="t('wizard.uploadStep.dataFile')" type="file" accept=".csv,.tsfile,text/csv,application/octet-stream" style="display:none" @change="onChange" />
        <span class="btn secondary sm" style="justify-self:center;pointer-events:none">
          <Icon name="file" :size="15" /> {{ t('wizard.uploadStep.chooseFile') }}
        </span>
      </label>

      <p v-if="uploading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.uploadStep.parsingFile') }}</p>
      <p v-if="wizardError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ wizardError }}</p>

      <div v-if="preview" class="stack">
        <div class="pill-row">
          <span class="badge primary"><Icon name="table" :size="13" />{{ t('wizard.uploadStep.columns', { count: preview.columns.length }) }}</span>
          <span class="badge"><Icon name="list" :size="13" />{{ t('wizard.uploadStep.previewRows', { count: preview.preview_rows.length }) }}</span>
          <span v-if="preview.file_format" class="badge">{{ preview.file_format }}</span>
          <span v-if="preview.detected_delimiter" class="badge">{{ t('wizard.uploadStep.delimiter', { delimiter: preview.detected_delimiter }) }}</span>
        </div>
        <div class="table-wrap">
          <table class="data">
            <caption>{{ t('wizard.uploadStep.previewFrom', { filename: preview.filename || t('wizard.uploadStep.uploadedFile') }) }}</caption>
            <thead>
              <tr>
                <th v-for="col in preview.columns" :key="col.name">
                  {{ col.name }}
                  <span v-if="col.inferred_type" class="faint" style="font-weight:500;text-transform:none">· {{ col.inferred_type }}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in preview.preview_rows" :key="i">
                <td v-for="col in preview.columns" :key="col.name" class="mono">{{ row[col.name] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>

    <template v-else-if="wizardState.dataSource === 'synthetic'">
      <div class="note-success">
        <Icon name="flask" :size="16" />
        {{ t('wizard.uploadStep.syntheticReady') }}
      </div>
      <p class="field-help">{{ t('wizard.uploadStep.syntheticDescription') }}</p>
    </template>

    <template v-else>
      <div class="note-success">
        <Icon name="layers" :size="16" />
        {{ t('wizard.uploadStep.existingReady') }}
      </div>
      <p class="field-help">{{ t('wizard.uploadStep.existingDescription') }}</p>
    </template>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ statusText }}</span>
      <div class="head-actions">
        <button class="btn" type="button" :disabled="nextDisabled" @click="continueFromSource">
          {{ t('wizard.uploadStep.next') }} <Icon name="arrowRight" :size="16" />
        </button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { uploadDataset } from '../../api/datasets';
import { applyUploadNameDefaults, goNext, wizardState, type WizardDataSource } from '../../stores/wizard';
import { messageFromError, renderMessage } from '../../lib/errors';

const { t, te } = useI18n();
const dragging = ref(false);
const uploading = ref(false);
const fileName = ref('');
const preview = computed(() => wizardState.preview);
const wizardError = computed(() => renderMessage(wizardState.error, t));

const sourceOptions = computed<Array<{ value: WizardDataSource; icon: string; title: string; description: string }>>(() => [
  {
    value: 'upload',
    icon: 'upload',
    title: t('wizard.uploadStep.sourceUpload'),
    description: t('wizard.uploadStep.sourceUploadDesc'),
  },
  {
    value: 'synthetic',
    icon: 'flask',
    title: t('wizard.uploadStep.sourceSynthetic'),
    description: t('wizard.uploadStep.sourceSyntheticDesc'),
  },
  {
    value: 'existing',
    icon: 'layers',
    title: t('wizard.uploadStep.sourceExisting'),
    description: t('wizard.uploadStep.sourceExistingDesc'),
  },
]);

const nextDisabled = computed(() => wizardState.dataSource === 'upload' && !preview.value);
const statusText = computed(() => {
  if (wizardState.dataSource === 'synthetic') return t('wizard.uploadStep.syntheticStatus');
  if (wizardState.dataSource === 'existing') return t('wizard.uploadStep.existingStatus');
  return preview.value ? t('wizard.uploadStep.previewLoaded') : t('wizard.uploadStep.waitingForFile');
});

async function onChange(event: Event) {
  const input = event.target as HTMLInputElement;
  await handle(input.files?.[0]);
}

async function onDrop(event: DragEvent) {
  dragging.value = false;
  await handle(event.dataTransfer?.files?.[0]);
}

async function handle(file?: File) {
  if (!file) return;
  fileName.value = file.name;
  uploading.value = true;
  try {
    wizardState.preview = await uploadDataset(file);
    wizardState.sourceUri = wizardState.preview.source_uri;
    wizardState.dataSource = 'upload';
    wizardState.dataUploadSkipped = false;
    applyUploadNameDefaults(wizardState.preview.filename || file.name);
    wizardState.error = null;
  } catch (error) {
    wizardState.error = messageFromError(error, te, 'wizard.uploadStep.errors.uploadFailed');
  } finally {
    uploading.value = false;
  }
}

function selectSource(source: WizardDataSource) {
  wizardState.dataSource = source;
  wizardState.error = null;
  if (source !== 'upload') {
    wizardState.preview = null;
    wizardState.sourceUri = '';
  }
  if (source === 'existing') {
    wizardState.dataUploadSkipped = true;
    return;
  }
  wizardState.dataUploadSkipped = false;
}

function continueFromSource() {
  if (wizardState.dataSource === 'existing') {
    wizardState.dataUploadSkipped = true;
  }
  goNext();
}
</script>
