<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('sampleForecast.eyebrow') }}</p>
        <h1>{{ t('sampleForecast.title') }}</h1>
        <p class="page-sub">{{ t('sampleForecast.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a v-if="backLink" class="btn secondary sm" :href="backLink">
          <Icon name="chevronLeft" :size="15" /> {{ backLabel }}
        </a>
        <ResumeWizardButton v-if="runId" resource-type="sample_forecast" :resource-id="runId" />
        <SampleNeighborNav
          v-if="sample"
          :shard-id="sample.shard_id"
          :sample-index="sample.sample_index"
          :report-id="reportContext.reportId"
          :track-id="reportContext.trackId"
          :sample-cursor-offset="reportContext.cursorOffset"
          :sample-capability-block-id="reportContext.capabilityBlockId"
          :sample-model-id="reportContext.modelId"
          :sample-metric="reportContext.metric"
          :sample-sort="reportContext.sort"
          :href-for-sample="sampleForecastHref"
        />
        <span class="badge mono">{{ sampleBadge }}</span>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="sample" class="stack">
        <div class="card pad">
          <ForecastChart :sample="sample" />
        </div>
        <div v-if="hasCovariates" class="card pad">
          <CovariateChart :sample="sample" />
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
import CovariateChart from '../components/results/CovariateChart.vue';
import SampleNeighborNav from '../components/results/SampleNeighborNav.vue';
import SampleMetricTable from '../components/results/SampleMetricTable.vue';
import { getSampleForecast } from '../api/results';
import type { SampleForecastDTO, SampleForecastSort } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{
  sampleId: string;
  runId?: string;
  trackId?: string;
  reportId?: string;
  sampleLinkOffset?: number;
  sampleCursorOffset?: number;
  sampleCapabilityBlockId?: string;
  sampleModelId?: string;
  sampleMetric?: string;
  sampleSort?: SampleForecastSort;
}>();
const { data: sample, loading, error, run } = useAsyncData<SampleForecastDTO>(() => getSampleForecast(props.sampleId, { runId: props.runId, trackId: props.trackId }));
const failedModels = computed(() => sample.value?.models.filter((m) => m.status !== 'succeeded') || []);
const hasCovariates = computed(() => Boolean(sample.value?.covariate_column_names?.length));
const reportContext = computed(() => ({
  reportId: props.reportId || sample.value?.links?.report || '',
  trackId: props.trackId || sample.value?.links?.track || '',
  listOffset: Math.max(0, Number(props.sampleLinkOffset || 0)),
  cursorOffset: typeof props.sampleCursorOffset === 'number' && Number.isFinite(props.sampleCursorOffset) ? Math.max(0, props.sampleCursorOffset) : null,
  capabilityBlockId: props.sampleCapabilityBlockId || '',
  modelId: props.sampleModelId || '',
  metric: props.sampleMetric || '',
  sort: props.sampleSort || 'sample_index',
}));
const reportLink = computed(() => {
  const reportId = reportContext.value.reportId;
  if (!reportId) return '';
  const params = new URLSearchParams();
  if (reportContext.value.listOffset > 0) params.set('sample_link_offset', String(reportContext.value.listOffset));
  if (reportContext.value.capabilityBlockId) params.set('sample_link_capability_block_id', reportContext.value.capabilityBlockId);
  if (reportContext.value.modelId) params.set('sample_link_model_id', reportContext.value.modelId);
  if (reportContext.value.sort !== 'sample_index') params.set('sample_link_sort', reportContext.value.sort);
  return `#/reports/${reportId}${params.toString() ? `?${params.toString()}` : ''}`;
});
const trackLink = computed(() => {
  const trackId = reportContext.value.trackId;
  return trackId ? `#/tracks/${trackId}` : '';
});
const backLink = computed(() => trackLink.value || reportLink.value);
const backLabel = computed(() => trackLink.value ? t('sampleForecast.backToTrack') : t('sampleForecast.backToReport'));
const sampleBadge = computed(() => {
  if (typeof sample.value?.sample_index === 'number') return `#${sample.value.sample_index + 1}`;
  return shortId(props.sampleId);
});
const { t } = useI18n();

function sampleForecastHref(sampleId: string, cursorOffset?: number | null) {
  const params = new URLSearchParams();
  if (reportContext.value.trackId) {
    params.set('track_id', reportContext.value.trackId);
  } else if (props.runId) {
    params.set('run_id', props.runId);
  }
  const reportId = reportContext.value.reportId;
  if (reportId && !reportContext.value.trackId) params.set('report_id', reportId);
  const safeCursor = typeof cursorOffset === 'number' && Number.isFinite(cursorOffset) ? Math.max(0, cursorOffset) : reportContext.value.cursorOffset;
  if (safeCursor !== null) {
    params.set('sample_cursor_offset', String(safeCursor));
    params.set('sample_link_offset', String(Math.floor(safeCursor / 10) * 10));
  } else if (reportContext.value.listOffset > 0) {
    params.set('sample_link_offset', String(reportContext.value.listOffset));
  }
  if (reportContext.value.capabilityBlockId) params.set('sample_link_capability_block_id', reportContext.value.capabilityBlockId);
  if (reportContext.value.modelId) params.set('sample_link_model_id', reportContext.value.modelId);
  if (reportContext.value.metric) params.set('sample_link_metric', reportContext.value.metric);
  if (reportContext.value.sort !== 'sample_index') params.set('sample_link_sort', reportContext.value.sort);
  return `#/samples/${encodeURIComponent(sampleId)}?${params.toString()}`;
}

onMounted(run);
</script>
