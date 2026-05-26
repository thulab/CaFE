<template>
  <section class="step-body">
    <div class="grid-2">
      <div class="field">
        <label class="label" for="time-column">Time column</label>
        <select id="time-column" v-model="timeColumn">
          <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
        </select>
        <p class="hint">The column that orders observations over time.</p>
      </div>

      <div class="field">
        <label class="label" for="target-select">Target column</label>
        <select id="target-select" v-model="target" aria-label="Target">
          <option value="">— select target —</option>
          <option v-for="column in valueColumns" :key="column" :value="column">{{ column }}</option>
        </select>
        <p class="hint">Exactly one target from the checked value columns.</p>
      </div>
    </div>

    <fieldset class="field" style="border:0;padding:0;margin:0">
      <legend class="label" style="padding:0;margin-bottom:6px">Value columns</legend>
      <div class="choice-grid">
        <label v-for="column in nonTimeColumns" :key="column" class="choice">
          <input v-model="valueColumns" type="checkbox" :value="column" :aria-label="column" />
          {{ column }}
        </label>
      </div>
    </fieldset>

    <div class="grid-auto">
      <div class="field">
        <label class="label">Context</label>
        <input v-model.number="context" aria-label="Context" type="number" min="1" />
        <p class="hint">Look-back window length.</p>
      </div>
      <div class="field">
        <label class="label">Horizon</label>
        <input v-model.number="horizon" aria-label="Horizon" type="number" min="1" />
        <p class="hint">Steps to forecast.</p>
      </div>
      <div class="field">
        <label class="label">Stride</label>
        <input v-model.number="stride" aria-label="Stride" type="number" min="1" />
        <p class="hint">Gap between sample windows.</p>
      </div>
      <div class="field">
        <label class="label">Max samples</label>
        <input v-model.number="maxSamples" aria-label="Max samples" type="number" min="1" placeholder="No cap" />
        <p class="hint">Optional cap on generated samples.</p>
      </div>
    </div>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="wizardState.shardId" class="note-success"><Icon name="checkCircle" :size="16" />Shard ready — continue to confirm it.</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">Window: {{ context }} context → {{ horizon }} horizon, stride {{ stride }}.</span>
      <button class="btn" type="button" :disabled="busy" @click="load">
        <span v-if="busy" class="spinner" /> {{ wizardState.shardId ? 'Re-load shard' : 'Load shard' }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import Icon from '../ui/Icon.vue';
import { createDatasetManifest, createLoadJob } from '../../api/datasets';
import { goNext, wizardState } from '../../stores/wizard';
import { refreshResourceCounts } from '../../composables/useResourceCounts';

const columns = computed(() => wizardState.preview?.columns.map((column) => column.name) || []);
const timeColumn = ref('time');
const nonTimeColumns = computed(() => columns.value.filter((c) => c !== timeColumn.value));

const valueColumns = ref<string[]>([]);
const target = ref('');
const context = ref(6);
const horizon = ref(3);
const stride = ref(3);
const maxSamples = ref<number | undefined>(undefined);
const error = ref('');
const busy = ref(false);

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
  busy.value = true;
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
    const job = await createLoadJob({ dataset_manifest_id: manifest.dataset_manifest_id, split_config: splitConfig });
    wizardState.loadJobId = job.load_job_id;
    wizardState.shardId = job.output_shard_id || '';
    void refreshResourceCounts();
    error.value = '';
    goNext();
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : 'Load failed';
  } finally {
    busy.value = false;
  }
}
</script>
