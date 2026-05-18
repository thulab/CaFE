<template>
  <section>
    <input aria-label="CSV file" type="file" accept=".csv,text/csv" @change="onFile" />
    <p v-if="wizardState.error" role="alert">{{ wizardState.error }}</p>
    <table v-if="wizardState.preview">
      <thead>
        <tr><th v-for="column in wizardState.preview.columns" :key="column.name">{{ column.name }}</th></tr>
      </thead>
      <tbody>
        <tr v-for="(row, index) in wizardState.preview.preview_rows" :key="index">
          <td v-for="column in wizardState.preview.columns" :key="column.name">{{ row[column.name] }}</td>
        </tr>
      </tbody>
    </table>
    <button :disabled="!wizardState.preview">Next</button>
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
