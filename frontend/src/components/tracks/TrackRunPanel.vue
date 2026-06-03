<template>
  <section class="stack">
    <div class="field" style="gap:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
        <span class="label">{{ t('runPanel.modelAdapters') }}</span>
        <button v-if="models.length" class="btn ghost sm" type="button" @click="toggleAll">
          {{ allSelected ? t('wizard.runStep.clearAll') : t('wizard.runStep.selectAll') }}
        </button>
      </div>
      <div v-if="models.length" class="choice-grid" :aria-label="t('wizard.runStep.availableModels')">
        <label v-for="model in models" :key="model.model_id" class="choice">
          <input v-model="selectedIds" type="checkbox" :value="model.model_id" :aria-label="model.name" :disabled="!isModelCompatible(model)" />
          <span style="display:grid;gap:1px;min-width:0">
            <span class="nowrap" style="overflow:hidden;text-overflow:ellipsis">{{ model.name }}</span>
            <span class="faint" style="font-size:0.74rem">{{ model.adapter_type }}{{ modelStateLabel(model) }}{{ modelTargetLimitLabel(model) }}</span>
          </span>
        </label>
      </div>
      <p v-else class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.runStep.loadingModels') }}</p>
    </div>

    <p v-if="isCreatingRun" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('runPanel.startingRun') }}</p>
    <p v-if="runId" class="note-success">
      <Icon name="checkCircle" :size="16" />{{ t('runPanel.runCreated', { id: runId }) }}
      <a class="text-link" :href="`#/runs/${runId}`">{{ t('wizard.runStep.openRun') }}</a>
    </p>
    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('runPanel.selectedModels', { count: selectedIds.length }) }}</span>
      <button class="btn" type="button" :disabled="selectedIds.length === 0 || isCreatingRun" @click="run">
        <span v-if="isCreatingRun" class="spinner" /> <Icon v-else name="play" :size="16" /> {{ t('runPanel.startRun') }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { listModels, type ModelDTO } from '../../api/models';
import { createRun } from '../../api/runs';
import { getShard } from '../../api/datasets';
import { getTrack } from '../../api/tracks';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { refreshResourceCounts } from '../../composables/useResourceCounts';
import { modelMaxTargetCount, modelSupportsTargetDim } from '../../lib/modelLimits';
import Icon from '../ui/Icon.vue';

const props = defineProps<{ trackId: string }>();
const emit = defineEmits<{ (event: 'run-created', runId: string): void }>();

const { t } = useI18n();
const models = ref<ModelDTO[]>([]);
const selectedIds = ref<string[]>([]);
const runId = ref('');
const isCreatingRun = ref(false);
const targetDim = ref(1);
const { text: error, clear: clearError, setError } = useDisplayMessage();

const compatibleModels = computed(() => models.value.filter((model) => isModelCompatible(model)));
const allSelected = computed(() => compatibleModels.value.length > 0 && selectedIds.value.length === compatibleModels.value.length);

onMounted(async () => {
  await Promise.all([loadModels(), loadTargetDim()]);
});
watch(() => props.trackId, () => {
  runId.value = '';
  clearError();
  void loadTargetDim();
});
watch([models, targetDim], pruneIncompatibleSelections);

async function loadModels() {
  try {
    models.value = (await listModels()).items;
  } catch (caught) {
    setError(caught, 'errors.failedToLoadModels');
  }
}

function toggleAll() {
  selectedIds.value = allSelected.value ? [] : compatibleModels.value.map((model) => model.model_id);
}

async function run() {
  clearError();
  isCreatingRun.value = true;
  runId.value = '';
  try {
    const created = await createRun({ track_id: props.trackId, model_ids: selectedIds.value });
    runId.value = created.benchmarking_run_id;
    emit('run-created', created.benchmarking_run_id);
    void refreshResourceCounts();
  } catch (caught) {
    setError(caught, 'errors.failedToStartRun');
  } finally {
    isCreatingRun.value = false;
  }
}

function modelStateLabel(model: ModelDTO) {
  if (model.loaded === true) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.loaded') });
  if (model.loading === true) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.loading') });
  if (model.loaded === false) return t('wizard.runStep.modelStateSuffix', { state: t('wizard.runStep.notLoaded') });
  return '';
}

function modelTargetLimitLabel(model: ModelDTO) {
  if (targetDim.value <= 1) return '';
  const maxTargetCount = modelMaxTargetCount(model);
  if (maxTargetCount === null) return t('wizard.runStep.targetLimitUnboundedSuffix');
  const count = maxTargetCount ?? 1;
  return t(count === 1 ? 'wizard.runStep.targetLimitSuffixOne' : 'wizard.runStep.targetLimitSuffixOther', { count });
}

function isModelCompatible(model: ModelDTO) {
  return modelSupportsTargetDim(model, targetDim.value);
}

function pruneIncompatibleSelections() {
  const allowed = new Set(compatibleModels.value.map((model) => model.model_id));
  const kept = selectedIds.value.filter((id) => allowed.has(id));
  if (kept.length !== selectedIds.value.length) selectedIds.value = kept;
}

async function loadTargetDim() {
  try {
    const track = await getTrack(props.trackId);
    const shardIds = track.shard_ids ?? [];
    if (!shardIds.length) {
      targetDim.value = 1;
      return;
    }
    const shards = await Promise.all(shardIds.map((id) => getShard(id)));
    targetDim.value = Math.max(1, ...shards.map((shard) => Number(shard.target_dim || shard.target_columns?.length || 1)));
  } catch (caught) {
    setError(caught, 'errors.failedToLoadTracks');
    targetDim.value = 1;
  }
}
</script>
