<template>
  <section class="step-body">
    <p class="field-help">
      {{ t('wizard.trackStep.description', { metric: wizardState.primaryMetric.toUpperCase() }) }}
    </p>

    <div class="grid-2">
      <div class="field">
        <label class="label" for="track-name">{{ t('wizard.trackStep.trackName') }}</label>
        <input id="track-name" v-model.trim="wizardState.trackName" :aria-label="t('wizard.trackStep.trackName')" />
        <p class="hint">{{ t('wizard.trackStep.trackNameHint') }}</p>
      </div>
      <div class="field">
        <label class="label" for="primary-metric">{{ t('wizard.trackStep.primaryMetricLabel') }}</label>
        <select id="primary-metric" v-model="wizardState.primaryMetric" :aria-label="t('wizard.trackStep.primaryMetricLabel')">
          <option value="mase">MASE</option>
          <option value="mse">MSE</option>
          <option value="mae">MAE</option>
        </select>
        <p class="hint">{{ t('wizard.trackStep.primaryMetricHint') }}</p>
      </div>
    </div>

    <p v-if="!wizardState.trackName" class="status-line">{{ t('wizard.trackStep.nameRequired') }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('wizard.trackStep.primaryMetric', { metric: wizardState.primaryMetric.toUpperCase() }) }}</span>
      <button class="btn" type="button" :disabled="!wizardState.trackName" @click="continueToData">
        {{ t('wizard.trackStep.continue') }} <Icon name="arrowRight" :size="16" />
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { goNext, wizardState } from '../../stores/wizard';

const { t } = useI18n();

onMounted(ensurePrimaryMetric);

function continueToData() {
  ensurePrimaryMetric();
  goNext();
}

function ensurePrimaryMetric() {
  if (!wizardState.primaryMetric) {
    wizardState.primaryMetric = 'mase';
  }
}
</script>
