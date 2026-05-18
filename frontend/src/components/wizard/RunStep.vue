<template>
  <section>
    <label v-for="model in models" :key="model.model_id">
      <input v-model="selectedIds" type="checkbox" :value="model.model_id" />
      {{ model.name }}
    </label>
    <button :disabled="selectedIds.length === 0" @click="run">Run</button>
    <p>{{ status }}</p>
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
