<template>
  <section class="step-body">
    <div class="field-stack">
      <label for="time-column">Time</label>
      <select id="time-column" v-model="timeColumn">
        <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
      </select>
      <p class="field-help">Use the column that orders observations over time.</p>
    </div>

    <fieldset class="field-stack">
      <legend>Target</legend>
      <div class="checkbox-grid">
        <label v-for="column in columns" :key="column" class="checkbox-row">
        <input v-model="targets" type="checkbox" :value="column" />
        {{ column }}
        </label>
      </div>
    </fieldset>

    <div class="form-grid">
      <label>Context <input v-model.number="context" aria-label="Context" type="number" min="1" /></label>
      <label>Horizon <input v-model.number="horizon" aria-label="Horizon" type="number" min="1" /></label>
      <label>Stride <input v-model.number="stride" aria-label="Stride" type="number" min="1" /></label>
    </div>

    <p v-if="error" class="alert" role="alert">{{ error }}</p>
    <div class="action-row">
      <button @click="load">Load shard</button>
      <p class="status-line">{{ wizardState.shardId ? 'Shard configuration saved' : 'Ready to create dataset shard' }}</p>
    </div>
    <div v-if="wizardState.manifestId || wizardState.loadJobId" class="resource-links" aria-label="Dataset view links">
      <a v-if="wizardState.manifestId" class="text-link" :href="`#/datasets/${wizardState.manifestId}`">Dataset manifest</a>
      <a v-if="wizardState.loadJobId" class="text-link" :href="`#/load-jobs/${wizardState.loadJobId}`">Load job</a>
    </div>
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
    wizardState.loadJobId = job.load_job_id;
    wizardState.shardId = job.output_shard_id || '';
    error.value = '';
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : 'Load failed';
  }
}
</script>
