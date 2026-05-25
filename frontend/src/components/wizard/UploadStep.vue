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
        <strong>{{ fileName || 'Drop a CSV here or browse' }}</strong>
        <p class="field-help" style="margin-top:4px">One time column and at least one numeric target column.</p>
      </div>
      <input id="csv-file" aria-label="CSV file" type="file" accept=".csv,text/csv" style="display:none" @change="onChange" />
      <span class="btn secondary sm" style="justify-self:center;pointer-events:none">
        <Icon name="file" :size="15" /> Choose file
      </span>
    </label>

    <p v-if="uploading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />Parsing CSV…</p>
    <p v-if="wizardState.error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ wizardState.error }}</p>

    <div v-if="preview" class="stack">
      <div class="pill-row">
        <span class="badge primary"><Icon name="table" :size="13" />{{ preview.columns.length }} columns</span>
        <span class="badge"><Icon name="list" :size="13" />{{ preview.preview_rows.length }} preview rows</span>
        <span v-if="preview.detected_delimiter" class="badge">Delimiter “{{ preview.detected_delimiter }}”</span>
      </div>
      <div class="table-wrap">
        <table class="data">
          <caption>Preview from {{ preview.filename || 'uploaded CSV' }}</caption>
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
      <span class="status-line">{{ preview ? 'Preview loaded — continue to configure the split.' : 'Waiting for a CSV.' }}</span>
      <button class="btn" type="button" :disabled="!preview" @click="goNext">
        Next <Icon name="arrowRight" :size="16" />
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import Icon from '../ui/Icon.vue';
import { uploadDataset } from '../../api/datasets';
import { goNext, wizardState } from '../../stores/wizard';

const dragging = ref(false);
const uploading = ref(false);
const fileName = ref('');
const preview = computed(() => wizardState.preview);

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
    wizardState.error = '';
  } catch (error) {
    wizardState.error = error instanceof Error ? error.message : 'Upload failed';
  } finally {
    uploading.value = false;
  }
}
</script>
