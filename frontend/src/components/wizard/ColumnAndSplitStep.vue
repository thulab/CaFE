<template>
  <section class="step-body">
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
        <label class="label" for="target-select">{{ t('wizard.columnAndSplitStep.targetColumn') }}</label>
        <select id="target-select" v-model="target" :aria-label="t('wizard.columnAndSplitStep.target')">
          <option value="">{{ t('wizard.columnAndSplitStep.selectTarget') }}</option>
          <option v-for="column in valueColumns" :key="column" :value="column">{{ column }}</option>
        </select>
        <p class="hint">{{ t('wizard.columnAndSplitStep.targetHint') }}</p>
      </div>
    </div>

    <fieldset class="field" style="border:0;padding:0;margin:0">
      <legend class="label" style="padding:0;margin-bottom:6px">{{ t('wizard.columnAndSplitStep.valueColumns') }}</legend>
      <div class="choice-grid">
        <label v-for="column in nonTimeColumns" :key="column" class="choice">
          <input v-model="valueColumns" type="checkbox" :value="column" :aria-label="column" />
          {{ column }}
        </label>
      </div>
    </fieldset>

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

const valueColumns = ref<string[]>([]);
const target = ref('');
const context = ref(6);
const horizon = ref(3);
const stride = ref(3);
const maxSamples = ref<number | undefined>(undefined);
const { text: error, clear: clearError, setKey: setErrorKey, setError } = useDisplayMessage();
const busy = ref(false);

watch(nonTimeColumns, (cols) => {
  if (cols.length > 0 && valueColumns.value.length === 0) {
    valueColumns.value = [...cols];
  }
}, { immediate: true });

watch(() => wizardState.preview?.filename, ensureNames, { immediate: true });

async function load() {
  if (!target.value || !valueColumns.value.includes(target.value)) {
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
      time_column: isTsFile.value ? 'time' : timeColumn.value,
      value_columns: valueColumns.value
    });
    wizardState.manifestId = manifest.dataset_manifest_id;

    const splitConfig: { context_length: number; horizon: number; stride?: number; target_columns: string[]; shard_name?: string; max_samples?: number } = {
      context_length: context.value,
      horizon: horizon.value,
      stride: stride.value,
      target_columns: [target.value],
      shard_name: wizardState.shardName || `${wizardState.datasetName || defaultNameFromFilename(wizardState.preview?.filename)} shard`
    };
    if (maxSamples.value != null && maxSamples.value > 0) {
      splitConfig.max_samples = maxSamples.value;
    }
    const job = await createLoadJob({ dataset_manifest_id: manifest.dataset_manifest_id, split_config: splitConfig });
    wizardState.loadJobId = job.load_job_id;
    wizardState.shardId = job.output_shard_id || '';
    void refreshResourceCounts();
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
  if (!wizardState.shardName) wizardState.shardName = `${wizardState.datasetName || base} shard`;
  if (!wizardState.trackName) wizardState.trackName = `${wizardState.datasetName || base} track`;
}
</script>
