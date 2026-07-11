<template>
  <section class="step-body">
    <div class="grid-2">
      <div class="field">
        <label class="label" for="synthetic-name">{{ t('wizard.syntheticStep.name') }}</label>
        <input id="synthetic-name" v-model.trim="wizardState.shardName" :aria-label="t('wizard.syntheticStep.name')" />
        <p class="hint">{{ t('wizard.syntheticStep.nameHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-frequency">{{ t('wizard.syntheticStep.frequency') }}</label>
        <select id="synthetic-frequency" v-model="frequency">
          <option value="h">{{ t('wizard.syntheticStep.hourly') }}</option>
          <option value="d">{{ t('wizard.syntheticStep.daily') }}</option>
          <option value="15min">{{ t('wizard.syntheticStep.fifteenMinutes') }}</option>
        </select>
        <p class="hint">{{ t('wizard.syntheticStep.frequencyHint') }}</p>
      </div>
    </div>

    <section class="stack" style="gap:10px" :aria-label="t('wizard.syntheticStep.capabilities')">
      <div class="field">
        <span class="label">{{ t('wizard.syntheticStep.capabilities') }}</span>
        <p class="hint">{{ t('wizard.syntheticStep.capabilitiesHint') }}</p>
      </div>
      <p v-if="loading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.syntheticStep.loadingCapabilities') }}</p>
      <div v-else class="capability-grid">
        <label
          v-for="capability in capabilities"
          :key="capability.capability_id"
          class="capability-card"
          :class="{ 'is-selected': selectedCapabilities.includes(capability.capability_id) }"
        >
          <input
            v-model="selectedCapabilities"
            type="checkbox"
            :value="capability.capability_id"
            :aria-label="t('wizard.syntheticStep.selectCapability', { name: capabilityLabel(capability) })"
          />
          <span class="capability-top">
            <span class="capability-title">{{ capabilityLabel(capability) }}</span>
            <span class="badge">{{ taskTypeLabel(capability.task_type) }}</span>
          </span>
          <span class="capability-desc">{{ capabilityDescription(capability) }}</span>
          <span v-if="capability.covariate_columns.length" class="faint">
            {{ t('wizard.syntheticStep.covariateColumns', { columns: capability.covariate_columns.join(', ') }) }}
          </span>
        </label>
      </div>
    </section>

    <div class="grid-auto">
      <div class="field">
        <label class="label" for="synthetic-sample-count">{{ t('wizard.syntheticStep.sampleCount') }}</label>
        <input id="synthetic-sample-count" v-model.number="sampleCount" type="number" min="1" max="1000" />
        <p class="hint">{{ t('wizard.syntheticStep.sampleCountHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-context">{{ t('wizard.syntheticStep.context') }}</label>
        <input id="synthetic-context" v-model.number="contextLength" type="number" min="16" max="2048" />
        <p class="hint">{{ t('wizard.syntheticStep.contextHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-horizon">{{ t('wizard.syntheticStep.horizon') }}</label>
        <input id="synthetic-horizon" v-model.number="horizon" type="number" min="1" max="512" />
        <p class="hint">{{ t('wizard.syntheticStep.horizonHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-intensity">{{ t('wizard.syntheticStep.intensity') }}</label>
        <select id="synthetic-intensity" v-model.number="intensity">
          <option v-for="level in [1, 2, 3, 4, 5]" :key="level" :value="level">{{ level }}</option>
        </select>
        <p class="hint">{{ t('wizard.syntheticStep.intensityHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-target-dim">{{ t('wizard.syntheticStep.targetDim') }}</label>
        <input id="synthetic-target-dim" v-model.number="targetDim" type="number" min="1" max="16" :disabled="targetDimDisabled" />
        <p class="hint">{{ targetDimDisabled ? t('wizard.syntheticStep.targetDimFixedHint') : t('wizard.syntheticStep.targetDimHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="synthetic-seed">{{ t('wizard.syntheticStep.seed') }}</label>
        <input id="synthetic-seed" v-model.number="seed" type="number" />
        <p class="hint">{{ t('wizard.syntheticStep.seedHint') }}</p>
      </div>
    </div>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="generatedCount" class="note-success">
      <Icon name="checkCircle" :size="16" />{{ t('wizard.syntheticStep.generated', { count: generatedCount }) }}
    </p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('wizard.syntheticStep.status', { count: selectedCapabilities.length, samples: sampleCount }) }}</span>
      <button class="btn" type="button" :disabled="busy || loading" @click="generate">
        <span v-if="busy" class="spinner" />
        {{ wizardState.shardId ? t('wizard.syntheticStep.regenerate') : t('wizard.syntheticStep.generate') }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { createSyntheticShards, listSyntheticCapabilities } from '../../api/synthetic';
import type { SyntheticCapabilityDTO } from '../../api/types';
import { refreshResourceCounts } from '../../composables/useResourceCounts';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { syntheticCapabilityDescription, syntheticCapabilityLabel } from '../../lib/syntheticCapabilities';
import { goNext, wizardState } from '../../stores/wizard';
import Icon from '../ui/Icon.vue';

const { t } = useI18n();
const capabilities = ref<SyntheticCapabilityDTO[]>([]);
const loading = ref(false);
const busy = ref(false);
const generatedCount = ref(0);
const selectedCapabilities = ref<string[]>([...wizardState.syntheticCapabilityIds]);
const sampleCount = ref(32);
const contextLength = ref(168);
const horizon = ref(24);
const intensity = ref(3);
const targetDim = ref(3);
const seed = ref(0);
const frequency = ref('h');
const { text: error, clear: clearError, setKey: setErrorKey, setError } = useDisplayMessage();

const selectedCapabilityDetails = computed(() => capabilities.value.filter((capability) => selectedCapabilities.value.includes(capability.capability_id)));
const hasMultiTargetCapability = computed(() => selectedCapabilityDetails.value.some((capability) => capability.target_dim_mode === 'multi'));
const targetDimDisabled = computed(() => selectedCapabilityDetails.value.length > 0 && !hasMultiTargetCapability.value);

watch(selectedCapabilities, (value) => {
  wizardState.syntheticCapabilityIds = [...value];
});

watch(selectedCapabilityDetails, (details) => {
  if (!details.length) return;
  const onlyHierarchy = details.every((capability) => capability.capability_id === 'hierarchical_coherence');
  if (onlyHierarchy) {
    contextLength.value = 365;
    horizon.value = 28;
    frequency.value = 'd';
  } else if (!details.some((capability) => capability.capability_id === 'hierarchical_coherence')) {
    contextLength.value = 168;
    horizon.value = 24;
    frequency.value = 'h';
  }
  targetDim.value = hasMultiTargetCapability.value ? Math.max(3, Number(targetDim.value) || 3) : 1;
});

onMounted(async () => {
  ensureName();
  await loadCapabilities();
});

async function loadCapabilities() {
  loading.value = true;
  clearError();
  try {
    capabilities.value = (await listSyntheticCapabilities()).items;
    if (!selectedCapabilities.value.length && capabilities.value.length) {
      selectedCapabilities.value = [capabilities.value[0].capability_id];
    }
  } catch (caught) {
    setError(caught, 'wizard.syntheticStep.errors.failedToLoadCapabilities');
  } finally {
    loading.value = false;
  }
}

async function generate() {
  clearError();
  ensureName();
  if (selectedCapabilities.value.length === 0) {
    setErrorKey('wizard.syntheticStep.errors.selectCapability');
    return;
  }
  if (sampleCount.value <= 0 || contextLength.value <= 0 || horizon.value <= 0 || targetDim.value <= 0) {
    setErrorKey('wizard.syntheticStep.errors.positiveValues');
    return;
  }
  busy.value = true;
  try {
    const result = await createSyntheticShards({
      name: wizardState.shardName,
      capabilities: [...selectedCapabilities.value],
      context_length: contextLength.value,
      horizon: horizon.value,
      sample_count: sampleCount.value,
      intensity: intensity.value,
      target_dim: targetDimDisabled.value ? 1 : targetDim.value,
      seed: seed.value,
      frequency: frequency.value,
    });
    const ids = result.shard_ids;
    wizardState.preview = null;
    wizardState.dataUploadSkipped = false;
    wizardState.manifestId = '';
    wizardState.loadJobId = '';
    wizardState.shardId = ids[0] || '';
    wizardState.selectedShardIds = [...ids, ...wizardState.selectedShardIds.filter((id) => !ids.includes(id))];
    generatedCount.value = ids.length;
    void refreshResourceCounts();
    goNext();
  } catch (caught) {
    setError(caught, 'wizard.syntheticStep.errors.failedToGenerate');
  } finally {
    busy.value = false;
  }
}

function ensureName() {
  if (!wizardState.shardName) {
    wizardState.shardName = wizardState.trackName ? `${wizardState.trackName} synthetic cases` : 'Synthetic test cases';
  }
}

function taskTypeLabel(taskType: string) {
  if (taskType === 'multivariate_forecast') return t('wizard.syntheticStep.multivariate');
  if (taskType === 'covariate_forecast') return t('wizard.syntheticStep.covariate');
  return t('wizard.syntheticStep.univariate');
}

function capabilityLabel(capability: SyntheticCapabilityDTO) {
  return syntheticCapabilityLabel(capability.capability_id, capability.label, t);
}

function capabilityDescription(capability: SyntheticCapabilityDTO) {
  return syntheticCapabilityDescription(capability.capability_id, capability.description, t);
}
</script>
