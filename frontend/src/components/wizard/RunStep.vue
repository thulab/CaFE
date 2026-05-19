<template>
  <section class="step-body">
    <div class="model-list" aria-label="Available models">
      <label v-for="model in models" :key="model.model_id" class="model-row">
        <input v-model="selectedIds" type="checkbox" :value="model.model_id" />
        {{ model.name }}
      </label>
      <p v-if="models.length === 0" class="empty-state">Loading model adapters...</p>
    </div>

    <div class="action-row">
      <button :disabled="selectedIds.length === 0 || !wizardState.trackId" @click="run">Run</button>
      <p class="status-line">{{ status }}</p>
    </div>
    <div v-if="wizardState.runId" class="resource-links" aria-label="Run view links">
      <a class="text-link" :href="`#/runs/${wizardState.runId}`">Run</a>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue';
import { listModels, type ModelDTO } from '../../api/models';
import { createRun, getRunProgress } from '../../api/runs';
import { wizardState } from '../../stores/wizard';

const models = ref<ModelDTO[]>([]);
const selectedIds = ref<string[]>([]);
const status = ref('Idle');
let timer: ReturnType<typeof setInterval> | undefined;

onMounted(async () => {
  models.value = (await listModels()).items;
});

onBeforeUnmount(() => stopPolling());

async function run() {
  stopPolling();
  const created = await createRun({ track_id: wizardState.trackId, model_ids: selectedIds.value });
  wizardState.runId = created.benchmarking_run_id;
  status.value = created.status;
  timer = setInterval(poll, 5000);
}

async function poll() {
  const progress = await getRunProgress(wizardState.runId);
  status.value = progress.status;
  if (progress.report_id) wizardState.reportId = progress.report_id;
  if (['succeeded', 'partial_succeeded', 'failed', 'cancelled'].includes(progress.status)) {
    stopPolling();
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}
</script>
