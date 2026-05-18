<template>
  <section>
    <label>Time
      <select v-model="timeColumn">
        <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
      </select>
    </label>
    <fieldset>
      <legend>Target</legend>
      <label v-for="column in columns" :key="column">
        <input v-model="targets" type="checkbox" :value="column" />
        {{ column }}
      </label>
    </fieldset>
    <label>Context <input v-model.number="context" aria-label="Context" type="number" min="1" /></label>
    <label>Horizon <input v-model.number="horizon" aria-label="Horizon" type="number" min="1" /></label>
    <label>Stride <input v-model.number="stride" aria-label="Stride" type="number" min="1" /></label>
    <p v-if="error" role="alert">{{ error }}</p>
    <button @click="load">Load shard</button>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { createDatasetManifest, createLoadJob } from '../../api/datasets';
import { wizardState } from '../../stores/wizard';

const columns = computed(() => wizardState.preview?.columns.map((column) => column.name) || []);
const timeColumn = ref('time');
const targets = ref<string[]>([]);
const context = ref(6);
const horizon = ref(3);
const stride = ref(3);
const error = ref('');

async function load() {
  if (targets.value.length !== 1) {
    error.value = 'Select exactly one target';
    return;
  }
  if (context.value <= 0 || horizon.value <= 0 || stride.value <= 0) {
    error.value = 'Split values must be positive';
    return;
  }
  try {
    const manifest = await createDatasetManifest({
      name: 'Uploaded dataset',
      domain: 'general',
      source_uri: wizardState.sourceUri,
      file_format: 'csv',
      time_column: timeColumn.value,
      target_columns: targets.value
    });
    wizardState.manifestId = manifest.dataset_manifest_id;
    const job = await createLoadJob({
      dataset_manifest_id: manifest.dataset_manifest_id,
      split_config: { context_length: context.value, horizon: horizon.value, stride: stride.value }
    });
    wizardState.shardId = job.output_shard_id || '';
    error.value = '';
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : 'Load failed';
  }
}
</script>
