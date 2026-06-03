<template>
  <section class="step-body">
    <template v-if="wizardState.dataUploadSkipped">
      <p class="field-help">{{ t('wizard.columnAndSplitStep.skipUploadDescription') }}</p>
      <p class="note-success"><Icon name="checkCircle" :size="16" />{{ t('wizard.columnAndSplitStep.skipUploadReady') }}</p>
      <div class="wizard-foot" style="padding:0;border:0">
        <span class="status-line">{{ t('wizard.columnAndSplitStep.skipUploadStatus') }}</span>
        <button class="btn" type="button" @click="goNext">
          {{ t('wizard.columnAndSplitStep.chooseExistingSets') }} <Icon name="arrowRight" :size="16" />
        </button>
      </div>
    </template>

    <template v-else>
    <div class="grid-2">
      <div class="field">
        <label class="label" for="dataset-name">{{ t('wizard.columnAndSplitStep.datasetName') }}</label>
        <input id="dataset-name" v-model.trim="wizardState.datasetName" :aria-label="t('wizard.columnAndSplitStep.datasetName')" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.datasetNameHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="shard-name">{{ t('wizard.columnAndSplitStep.shardName') }}</label>
        <input id="shard-name" v-model.trim="wizardState.shardName" :aria-label="t('wizard.columnAndSplitStep.shardName')" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.shardNameHint') }}</p>
      </div>
    </div>

    <div class="grid-2">
      <div v-if="!isTsFile" class="field">
        <label class="label" for="time-column">{{ t('wizard.columnAndSplitStep.timeColumn') }}</label>
        <select id="time-column" v-model="timeColumn">
          <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
        </select>
        <p class="hint">{{ t('wizard.columnAndSplitStep.timeColumnHint') }}</p>
      </div>
      <div v-else class="field">
        <span class="label">{{ t('wizard.columnAndSplitStep.timeColumn') }}</span>
        <p class="status-line">{{ t('wizard.columnAndSplitStep.tsfileTimeColumn') }}</p>
        <p class="hint">{{ t('wizard.columnAndSplitStep.tsfileHint') }}</p>
      </div>

      <div class="field">
        <span class="label">{{ t('wizard.columnAndSplitStep.targetColumns') }}</span>
        <div class="choice-grid" role="group" :aria-label="t('wizard.columnAndSplitStep.targetColumns')">
          <label v-for="column in nonTimeColumns" :key="column" class="choice">
            <input v-model="targets" type="checkbox" :value="column" :aria-label="column" />
            <span class="nowrap" style="overflow:hidden;text-overflow:ellipsis">{{ column }}</span>
          </label>
        </div>
        <p class="hint">{{ t('wizard.columnAndSplitStep.targetHint') }}</p>
      </div>
    </div>

    <div class="grid-auto">
      <div class="field">
        <label class="label">{{ t('wizard.columnAndSplitStep.context') }}</label>
        <input v-model.number="context" :aria-label="t('wizard.columnAndSplitStep.context')" type="number" min="1" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.contextHint') }}</p>
      </div>
      <div class="field">
        <label class="label">{{ t('wizard.columnAndSplitStep.horizon') }}</label>
        <input v-model.number="horizon" :aria-label="t('wizard.columnAndSplitStep.horizon')" type="number" min="1" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.horizonHint') }}</p>
      </div>
      <div class="field">
        <label class="label">{{ t('wizard.columnAndSplitStep.stride') }}</label>
        <input v-model.number="stride" :aria-label="t('wizard.columnAndSplitStep.stride')" type="number" min="1" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.strideHint') }}</p>
      </div>
      <div class="field">
        <label class="label">{{ t('wizard.columnAndSplitStep.maxSamples') }}</label>
        <input v-model.number="maxSamples" :aria-label="t('wizard.columnAndSplitStep.maxSamples')" type="number" min="1" :placeholder="t('wizard.columnAndSplitStep.noCap')" />
        <p class="hint">{{ t('wizard.columnAndSplitStep.maxSamplesHint') }}</p>
      </div>
    </div>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="wizardState.shardId" class="note-success"><Icon name="checkCircle" :size="16" />{{ t('wizard.columnAndSplitStep.shardReady') }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('wizard.columnAndSplitStep.windowStatus', { context, horizon, stride }) }}</span>
      <button class="btn" type="button" :disabled="busy" @click="load">
        <span v-if="busy" class="spinner" /> {{ wizardState.shardId ? t('wizard.columnAndSplitStep.reloadShard') : t('wizard.columnAndSplitStep.loadShard') }}
      </button>
    </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { createDatasetManifest, createLoadJob } from '../../api/datasets';
import { defaultNameFromFilename, goNext, wizardState } from '../../stores/wizard';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { refreshResourceCounts } from '../../composables/useResourceCounts';

const { t } = useI18n();
const columns = computed(() => wizardState.preview?.columns.map((column) => column.name) || []);
const fileFormat = computed(() => wizardState.preview?.file_format === 'tsfile' ? 'tsfile' : 'csv');
const isTsFile = computed(() => fileFormat.value === 'tsfile');
const timeColumn = ref('time');
const nonTimeColumns = computed(() => isTsFile.value ? columns.value : columns.value.filter((c) => c !== timeColumn.value));

const targets = ref<string[]>([]);
const context = ref(60);
const horizon = ref(16);
const stride = ref(16);
const maxSamples = ref<number | undefined>(undefined);
const { text: error, clear: clearError, setKey: setErrorKey, setRaw: setRawError, setError } = useDisplayMessage();
const busy = ref(false);

watch(nonTimeColumns, (cols) => {
  targets.value = targets.value.filter((target) => cols.includes(target));
}, { immediate: true });

watch(() => wizardState.preview?.filename, ensureNames, { immediate: true });

async function load() {
  if (targets.value.length === 0 || targets.value.some((target) => !nonTimeColumns.value.includes(target))) {
    setErrorKey('wizard.columnAndSplitStep.errors.selectExactlyOneTarget');
    return;
  }
  if (context.value <= 0 || horizon.value <= 0 || stride.value <= 0) {
    setErrorKey('wizard.columnAndSplitStep.errors.positiveSplitValues');
    return;
  }
  busy.value = true;
  try {
    ensureNames();
    const manifest = await createDatasetManifest({
      name: wizardState.datasetName || defaultNameFromFilename(wizardState.preview?.filename),
      domain: 'general',
      source_uri: wizardState.sourceUri,
      file_format: fileFormat.value,
      time_column: isTsFile.value ? 'time' : timeColumn.value
    });
    wizardState.manifestId = manifest.dataset_manifest_id;

    const splitConfig: { context_length: number; horizon: number; stride?: number; target_columns: string[]; shard_name?: string; max_samples?: number } = {
      context_length: context.value,
      horizon: horizon.value,
      stride: stride.value,
      target_columns: [...targets.value],
      shard_name: wizardState.shardName || `${wizardState.datasetName || defaultNameFromFilename(wizardState.preview?.filename)} shard`
    };
    if (maxSamples.value != null && maxSamples.value > 0) {
      splitConfig.max_samples = maxSamples.value;
    }
    const job = await createLoadJob({ dataset_manifest_id: manifest.dataset_manifest_id, split_config: splitConfig });
    wizardState.loadJobId = job.load_job_id;
    wizardState.shardId = job.output_shard_id || '';
    if (job.output_shard_id) selectGeneratedShard(job.output_shard_id);
    void refreshResourceCounts();
    if (job.status !== 'succeeded' || !job.output_shard_id) {
      setRawError(loadJobErrorMessage(job));
      return;
    }
    clearError();
    goNext();
  } catch (caught) {
    setError(caught, 'wizard.columnAndSplitStep.errors.loadFailed');
  } finally {
    busy.value = false;
  }
}

function ensureNames() {
  const base = defaultNameFromFilename(wizardState.preview?.filename);
  if (!wizardState.datasetName) wizardState.datasetName = base;
  if (!wizardState.shardName) wizardState.shardName = `${wizardState.datasetName || base} test cases`;
}

function loadJobErrorMessage(job: { status?: string; error_code?: string | null; error_message?: string | null }) {
  const code = job.error_code || job.status || 'load_failed';
  return job.error_message ? `${code} · ${job.error_message}` : code;
}

function selectGeneratedShard(shardId: string) {
  wizardState.selectedShardIds = [shardId, ...wizardState.selectedShardIds.filter((id) => id !== shardId)];
}
</script>
