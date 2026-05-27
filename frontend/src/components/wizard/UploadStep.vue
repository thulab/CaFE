<template>
  <section class="step-body">
    <label
      class="dropzone"
      :class="{ 'is-drag': dragging }"
      for="csv-file"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
    >
      <Icon class="dz-icon" name="upload" :size="28" />
      <div>
        <strong>{{ fileName || t('wizard.uploadStep.dropOrBrowse') }}</strong>
        <p class="field-help" style="margin-top:4px">{{ t('wizard.uploadStep.help') }}</p>
      </div>
      <input id="csv-file" :aria-label="t('wizard.uploadStep.csvFile')" type="file" accept=".csv,text/csv" style="display:none" @change="onChange" />
      <span class="btn secondary sm" style="justify-self:center;pointer-events:none">
        <Icon name="file" :size="15" /> {{ t('wizard.uploadStep.chooseFile') }}
      </span>
    </label>

    <p v-if="uploading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.uploadStep.parsingCsv') }}</p>
    <p v-if="wizardError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ wizardError }}</p>

    <div v-if="preview" class="stack">
      <div class="pill-row">
        <span class="badge primary"><Icon name="table" :size="13" />{{ t('wizard.uploadStep.columns', { count: preview.columns.length }) }}</span>
        <span class="badge"><Icon name="list" :size="13" />{{ t('wizard.uploadStep.previewRows', { count: preview.preview_rows.length }) }}</span>
        <span v-if="preview.detected_delimiter" class="badge">{{ t('wizard.uploadStep.delimiter', { delimiter: preview.detected_delimiter }) }}</span>
      </div>
      <div class="table-wrap">
        <table class="data">
          <caption>{{ t('wizard.uploadStep.previewFrom', { filename: preview.filename || t('wizard.uploadStep.uploadedCsv') }) }}</caption>
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

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ preview ? t('wizard.uploadStep.previewLoaded') : t('wizard.uploadStep.waitingForCsv') }}</span>
      <button class="btn" type="button" :disabled="!preview" @click="goNext">
        {{ t('wizard.uploadStep.next') }} <Icon name="arrowRight" :size="16" />
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { uploadDataset } from '../../api/datasets';
import { goNext, wizardState } from '../../stores/wizard';
import { messageFromError, renderMessage } from '../../lib/errors';

const { t, te } = useI18n();
const dragging = ref(false);
const uploading = ref(false);
const fileName = ref('');
const preview = computed(() => wizardState.preview);
const wizardError = computed(() => renderMessage(wizardState.error, t));

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
    wizardState.error = null;
  } catch (error) {
    wizardState.error = messageFromError(error, te, 'wizard.uploadStep.errors.uploadFailed');
  } finally {
    uploading.value = false;
  }
}
</script>
