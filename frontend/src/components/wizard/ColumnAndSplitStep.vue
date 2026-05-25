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
      <legend>Value columns</legend>
      <div class="checkbox-grid">
        <label v-for="column in nonTimeColumns" :key="column" class="checkbox-row">
          <input v-model="valueColumns" type="checkbox" :value="column" :aria-label="column" />
          {{ column }}
        </label>
      </div>
    </fieldset>

    <div class="field-stack">
      <label for="target-select">Target</label>
      <select id="target-select" v-model="target" aria-label="Target">
        <option value="">-- select target --</option>
        <option v-for="column in valueColumns" :key="column" :value="column">{{ column }}</option>
      </select>
      <p class="field-help">Select exactly one target from the checked value columns.</p>
    </div>

    <div class="form-grid">
      <label>Context <input v-model.number="context" aria-label="Context" type="number" min="1" /></label>
      <label>Horizon <input v-model.number="horizon" aria-label="Horizon" type="number" min="1" /></label>
      <label>Stride <input v-model.number="stride" aria-label="Stride" type="number" min="1" /></label>
      <label>Max samples <input v-model.number="maxSamples" aria-label="Max samples" type="number" min="1" placeholder="No cap" /></label>
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
import { computed, ref, watch } from 'vue';
import { createDatasetManifest, createLoadJob } from '../../api/datasets';
import { wizardState } from '../../stores/wizard';

const columns = computed(() => wizardState.preview?.columns.map((column) => column.name) || []);
const nonTimeColumns = computed(() => columns.value.filter((c) => c !== timeColumn.value));

const timeColumn = ref('time');
const valueColumns = ref<string[]>([]);
const target = ref('');
const context = ref(6);
const horizon = ref(3);
const stride = ref(3);
const maxSamples = ref<number | undefined>(undefined);
const error = ref('');

// Default all non-time columns as value columns when columns become available
watch(nonTimeColumns, (cols) => {
  if (cols.length > 0 && valueColumns.value.length === 0) {
    valueColumns.value = [...cols];
  }
}, { immediate: true });

async function load() {
  if (!target.value || !valueColumns.value.includes(target.value)) {
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
      value_columns: valueColumns.value
    });
    wizardState.manifestId = manifest.dataset_manifest_id;
    const splitConfig: { context_length: number; horizon: number; stride?: number; target_columns: string[]; max_samples?: number } = {
      context_length: context.value,
      horizon: horizon.value,
      stride: stride.value,
      target_columns: [target.value]
    };
    if (maxSamples.value != null && maxSamples.value > 0) {
      splitConfig.max_samples = maxSamples.value;
    }
    const job = await createLoadJob({
      dataset_manifest_id: manifest.dataset_manifest_id,
      split_config: splitConfig
    });
    wizardState.loadJobId = job.load_job_id;
    wizardState.shardId = job.output_shard_id || '';
    error.value = '';
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : 'Load failed';
  }
}
</script>
