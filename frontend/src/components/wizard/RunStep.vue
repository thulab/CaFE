<template>
  <section class="step-body">
    <div class="field" style="gap:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
        <span class="label">Model adapters</span>
        <button v-if="models.length" class="btn ghost sm" type="button" @click="toggleAll">
          {{ allSelected ? 'Clear all' : 'Select all' }}
        </button>
      </div>
      <div v-if="models.length" class="choice-grid" aria-label="Available models">
        <label v-for="model in models" :key="model.model_id" class="choice">
          <input v-model="selectedIds" type="checkbox" :value="model.model_id" :aria-label="model.name" />
          <span style="display:grid;gap:1px;min-width:0">
            <span class="nowrap" style="overflow:hidden;text-overflow:ellipsis">{{ model.name }}</span>
            <span class="faint" style="font-size:0.74rem">{{ model.adapter_type }}{{ modelStateLabel(model) }}</span>
          </span>
        </label>
      </div>
      <p v-else class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />Loading model adapters…</p>
    </div>
    <p v-if="isPreparingModels" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />Loading selected timer-service models…</p>

    <!-- Live progress -->
    <div v-if="runId" class="card pad" style="display:grid;gap:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
        <StatusBadge :status="status" big />
        <span class="status-line">{{ runningPct }}% complete</span>
      </div>
      <div class="progress" :class="progressVariant"><span :style="{ width: runningPct + '%' }" /></div>
      <div v-if="progress" class="pill-row">
        <span class="badge"><Icon name="layers" :size="13" />{{ progress.progress.completed_models ?? 0 }}/{{ progress.progress.total_models ?? selectedIds.length }} models</span>
        <span class="badge"><Icon name="list" :size="13" />{{ progress.progress.completed_tasks ?? 0 }}/{{ progress.progress.total_tasks ?? 0 }} tasks</span>
        <span class="badge"><Icon name="gauge" :size="13" />{{ progress.progress.completed_samples ?? 0 }}/{{ progress.progress.total_samples ?? 0 }} samples</span>
      </div>
    </div>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <div class="pill-row">
        <a v-if="runId" class="btn secondary sm" :href="`#/runs/${runId}`"><Icon name="external" :size="15" /> Open run</a>
        <button v-if="isRunning" class="btn danger sm" type="button" @click="onCancel"><Icon name="ban" :size="15" /> Cancel</button>
      </div>
      <button class="btn" type="button" :disabled="selectedIds.length === 0 || !wizardState.trackId || isPreparingModels || isRunning" @click="run">
        <span v-if="isPreparingModels || isRunning" class="spinner" /> <Icon v-else name="play" :size="16" /> Run
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import Icon from '../ui/Icon.vue';
import StatusBadge from '../ui/StatusBadge.vue';
import { listModels, loadModel, type ModelDTO } from '../../api/models';
import { cancelRun, createRun, getRunProgress } from '../../api/runs';
import { goNext, wizardState } from '../../stores/wizard';
import { refreshResourceCounts } from '../../composables/useResourceCounts';
import type { RunProgressDTO } from '../../api/types';
import { percent } from '../../lib/format';

const TERMINAL = ['succeeded', 'partial_succeeded', 'failed', 'cancelled'];

const models = ref<ModelDTO[]>([]);
const selectedIds = ref<string[]>([]);
const status = ref('idle');
const error = ref('');
const progress = ref<RunProgressDTO | null>(null);
const isPreparingModels = ref(false);
let timer: ReturnType<typeof setInterval> | undefined;

const runId = computed(() => wizardState.runId);
const isRunning = computed(() => !TERMINAL.includes(status.value) && Boolean(runId.value));
const allSelected = computed(() => models.value.length > 0 && selectedIds.value.length === models.value.length);

const runningPct = computed(() => {
  if (status.value === 'succeeded') return 100;
  const p = progress.value?.progress;
  if (!p) return isRunning.value ? 8 : 0;
  return percent(Number(p.completed_tasks ?? 0), Number(p.total_tasks ?? 0));
});

const progressVariant = computed(() => {
  if (status.value === 'failed' || status.value === 'cancelled') return 'danger';
  if (status.value === 'succeeded') return 'success';
  return isRunning.value ? 'striped' : '';
});

onMounted(async () => {
  try {
    models.value = (await listModels()).items;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load models';
  }
});

onBeforeUnmount(() => stopPolling());

function toggleAll() {
  selectedIds.value = allSelected.value ? [] : models.value.map((m) => m.model_id);
}

async function run() {
  stopPolling();
  error.value = '';
  progress.value = null;
  isPreparingModels.value = true;
  try {
    await loadSelectedModels();
    const created = await createRun({ track_id: wizardState.trackId, model_ids: selectedIds.value });
    wizardState.runId = created.benchmarking_run_id;
    status.value = created.status;
    void refreshResourceCounts();
    timer = setInterval(poll, 5000);
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to start run';
  } finally {
    isPreparingModels.value = false;
  }
}

async function loadSelectedModels() {
  const selected = models.value.filter((model) => selectedIds.value.includes(model.model_id));
  for (const model of selected) {
    if (model.loaded === false) {
      const loaded = await loadModel(model.model_id);
      const index = models.value.findIndex((item) => item.model_id === model.model_id);
      if (index >= 0) models.value[index] = { ...models.value[index], ...loaded };
    }
  }
}

function modelStateLabel(model: ModelDTO) {
  if (model.loaded === true) return ' · loaded';
  if (model.loading === true) return ' · loading';
  if (model.loaded === false) return ' · not loaded';
  return '';
}

async function poll() {
  try {
    const p = await getRunProgress(wizardState.runId);
    progress.value = p;
    status.value = p.status;
    if (p.report_id) {
      wizardState.reportId = p.report_id;
      void refreshResourceCounts();
    }
    if (p.ranking_list_id) wizardState.rankingListId = p.ranking_list_id;
    if (TERMINAL.includes(p.status)) {
      stopPolling();
      if (wizardState.reportId) goNext();
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to read progress';
    stopPolling();
  }
}

async function onCancel() {
  try {
    const res = await cancelRun(wizardState.runId);
    status.value = res.status;
    stopPolling();
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to cancel run';
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}
</script>
