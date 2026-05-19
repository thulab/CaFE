<template>
  <section class="step-body">
    <div class="file-upload">
      <label class="control-label" for="csv-file">CSV file</label>
      <input id="csv-file" aria-label="CSV file" type="file" accept=".csv,text/csv" @change="onFile" />
      <p class="field-help">Upload a CSV with one time column and at least one numeric target column.</p>
    </div>

    <p v-if="wizardState.error" class="alert" role="alert">{{ wizardState.error }}</p>

    <div v-if="wizardState.preview" class="table-scroll">
      <table class="data-table">
        <caption>Preview rows from {{ wizardState.preview.filename || 'uploaded CSV' }}</caption>
        <thead>
          <tr><th v-for="column in wizardState.preview.columns" :key="column.name">{{ column.name }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in wizardState.preview.preview_rows" :key="index">
            <td v-for="column in wizardState.preview.columns" :key="column.name">{{ row[column.name] }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="action-row">
      <button :disabled="!wizardState.preview">Next</button>
      <p class="status-line">{{ wizardState.preview ? 'Preview loaded' : 'Waiting for CSV preview' }}</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import { uploadDataset } from '../../api/datasets';
import { wizardState } from '../../stores/wizard';

async function onFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    wizardState.preview = await uploadDataset(file);
    wizardState.sourceUri = wizardState.preview.source_uri;
    wizardState.error = '';
  } catch (error) {
    wizardState.error = error instanceof Error ? error.message : 'Upload failed';
  }
}
</script>
