<template>
  <section class="step-body">
    <div class="field" style="gap:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
        <span class="label">{{ t('wizard.runStep.modelAdapters') }}</span>
        <button v-if="models.length" class="btn ghost sm" type="button" @click="toggleAll">
          {{ allSelected ? t('wizard.runStep.clearAll') : t('wizard.runStep.selectAll') }}
        </button>
      </div>
      <div v-if="models.length" class="choice-grid" :aria-label="t('wizard.runStep.availableModels')">
        <label v-for="model in models" :key="model.model_id" class="choice">
          <input v-model="selectedIds" type="checkbox" :value="model.model_id" :aria-label="model.name" />
          <span style="display:grid;gap:1px;min-width:0">
            <span class="nowrap" style="overflow:hidden;text-overflow:ellipsis">{{ model.name }}</span>
            <span class="faint" style="font-size:0.74rem">{{ model.adapter_type }}{{ modelStateLabel(model) }}</span>
          </span>
        </label>
      </div>
      <p v-else class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.runStep.loadingModels') }}</p>
    </div>
    <p v-if="isPreparingModels" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.runStep.loadingSelectedModels') }}</p>

    <!-- Live progress -->
    <div v-if="runId" class="card pad" style="display:grid;gap:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
        <StatusBadge :status="status" big />
        <span class="status-line">{{ t('wizard.runStep.percentComplete', { percent: runningPct }) }}</span>
      </div>
      <div class="progress" :class="progressVariant"><span :style="{ width: runningPct + '%' }" /></div>
      <div v-if="progress" class="pill-row">
        <span class="badge"><Icon name="layers" :size="13" />{{ t('wizard.runStep.modelsProgress', { done: progress.progress.completed_models ?? 0, total: progress.progress.total_models ?? selectedIds.length }) }}</span>
        <span class="badge"><Icon name="list" :size="13" />{{ t('wizard.runStep.tasksProgress', { done: progress.progress.completed_tasks ?? 0, total: progress.progress.total_tasks ?? 0 }) }}</span>
        <span class="badge"><Icon name="gauge" :size="13" />{{ t('wizard.runStep.samplesProgress', { done: progress.progress.completed_samples ?? 0, total: progress.progress.total_samples ?? 0 }) }}</span>
      </div>
    </div>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <div class="pill-row">
        <a v-if="runId" class="btn secondary sm" :href="`#/runs/${runId}`"><Icon name="external" :size="15" /> {{ t('wizard.runStep.openRun') }}</a>
        <button v-if="isRunning" class="btn danger sm" type="button" @click="onCancel"><Icon name="ban" :size="15" /> {{ t('common.cancel') }}</button>
      </div>
      <button class="btn" type="button" :disabled="selectedIds.length === 0 || !wizardState.trackId || isPreparingModels || isRunning" @click="run">
        <span v-if="isPreparingModels || isRunning" class="spinner" /> <Icon v-else name="play" :size="16" /> {{ t('wizard.runStep.run') }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import StatusBadge from '../ui/StatusBadge.vue';
import { listModels, loadModel, type ModelDTO } from '../../api/models';
import { cancelRun, createRun, getRunProgress } from '../../api/runs';
import { goNext, wizardState } from '../../stores/wizard';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { refreshResourceCounts } from '../../composables/useResourceCounts';
import type { RunProgressDTO } from '../../api/types';
import { percent } from '../../lib/format';

const TERMINAL = ['succeeded', 'partial_succeeded', 'failed', 'cancelled'];

const { t } = useI18n();
const models = ref<ModelDTO[]>([]);
const status = ref('idle');
const { text: error, clear: clearError, setError } = useDisplayMessage();
const progress = ref<RunProgressDTO | null>(null);
const isPreparingModels = ref(false);
let timer: ReturnType<typeof setInterval> | undefined;

const runId = computed(() => wizardState.runId);
const selectedIds = computed({
  get: () => wizardState.selectedModelIds,
  set: (ids: string[]) => {
    wizardState.selectedModelIds = ids;
  }
});
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
    if (wizardState.runId) {
      await poll();
      if (!TERMINAL.includes(status.value)) timer = setInterval(poll, 5000);
    }
  } catch (e) {
    setError(e, 'errors.failedToLoadModels');
  }
});

onBeforeUnmount(() => stopPolling());

function toggleAll() {
  selectedIds.value = allSelected.value ? [] : models.value.map((m) => m.model_id);
}

async function run() {
  stopPolling();
  clearError();
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
    setError(e, 'errors.failedToStartRun');
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
  if (model.loaded === true) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.loaded') });
  if (model.loading === true) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.loading') });
  if (model.loaded === false) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.notLoaded') });
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
    setError(e, 'errors.failedToReadProgress');
    stopPolling();
  }
}

async function onCancel() {
  try {
    const res = await cancelRun(wizardState.runId);
    status.value = res.status;
    stopPolling();
  } catch (e) {
    setError(e, 'errors.failedToCancelRun');
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}
</script>
