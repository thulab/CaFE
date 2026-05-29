<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('sampleForecast.eyebrow') }}</p>
        <h1>{{ t('sampleForecast.title') }}</h1>
        <p class="page-sub">{{ t('sampleForecast.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a v-if="reportLink" class="btn secondary sm" :href="reportLink">
          <Icon name="chevronLeft" :size="15" /> {{ t('sampleForecast.backToReport') }}
        </a>
        <ResumeWizardButton resource-type="sample_forecast" :resource-id="runId" />
        <span class="badge mono">{{ sampleBadge }}</span>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="sample" class="stack">
        <div class="card pad">
          <ForecastChart :sample="sample" />
        </div>
        <div class="card">
          <header class="card-head"><h2 class="card-title">{{ t('sampleForecast.perSampleMetrics') }}</h2></header>
          <div class="card-body">
            <SampleMetricTable :models="sample.models" />
          </div>
        </div>
        <div v-if="failedModels.length" class="stack">
          <p v-for="model in failedModels" :key="model.model_id" class="alert" role="alert">
            <Icon class="alert-ico" name="alert" :size="16" />
            <span><strong>{{ model.model_name || model.model_id }}:</strong> {{ model.error_message || model.status }}</span>
          </p>
        </div>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import ForecastChart from '../components/results/ForecastChart.vue';
import SampleMetricTable from '../components/results/SampleMetricTable.vue';
import { getSampleForecast } from '../api/results';
import type { SampleForecastDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ sampleId: string; runId: string; reportId?: string }>();
const { data: sample, loading, error, run } = useAsyncData<SampleForecastDTO>(() => getSampleForecast(props.sampleId, props.runId));
const failedModels = computed(() => sample.value?.models.filter((m) => m.status !== 'succeeded') || []);
const reportLink = computed(() => {
  const reportId = props.reportId || sample.value?.links?.report;
  return reportId ? `#/reports/${reportId}` : '';
});
const sampleBadge = computed(() => {
  if (typeof sample.value?.sample_index === 'number') return `#${sample.value.sample_index + 1}`;
  return shortId(props.sampleId);
});
const { t } = useI18n();

onMounted(run);
</script>
