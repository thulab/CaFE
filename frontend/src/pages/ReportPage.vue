<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('report.eyebrow') }}</p>
        <h1>{{ t('report.title') }}</h1>
        <p class="page-sub">{{ t('report.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a v-if="report?.track_id" class="btn secondary sm" :href="`#/tracks/${report.track_id}`">
          <Icon name="chevronLeft" :size="15" /> {{ t('report.backToTrack') }}
        </a>
        <ResumeWizardButton resource-type="report" :resource-id="reportId" />
        <span class="badge primary mono">{{ shortId(reportId) }}</span>
      </div>
    </header>

    <StateBlock :loading="loading && !report" :error="error" @retry="run">
      <ReportSummary
        v-if="report"
        :report="report"
        :sample-capability-block-id="sampleLinkCapabilityBlockId"
        :sample-model-id="sampleLinkModelId"
        :sample-sort="sampleLinkSort"
        @sample-page-change="loadSamplePage"
        @sample-query-change="updateSampleQuery"
      />
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import ReportSummary from '../components/results/ReportSummary.vue';
import { getReport } from '../api/results';
import type { ReportDTO, SampleForecastSort } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ reportId: string }>();
const SAMPLE_LINK_PAGE_SIZE = 10;
const initialSampleQuery = readSampleQueryFromHash();
const sampleLinkOffset = ref(initialSampleQuery.offset);
const sampleLinkCapabilityBlockId = ref(initialSampleQuery.capabilityBlockId);
const sampleLinkModelId = ref(initialSampleQuery.modelId);
const sampleLinkSort = ref<SampleForecastSort>(initialSampleQuery.sort);
const { data: report, loading, error, run } = useAsyncData<ReportDTO>(() =>
  getReport(props.reportId, {
    sampleLinkLimit: SAMPLE_LINK_PAGE_SIZE,
    sampleLinkOffset: sampleLinkOffset.value,
    sampleLinkCapabilityBlockId: sampleLinkCapabilityBlockId.value,
    sampleLinkModelId: sampleLinkModelId.value,
    sampleLinkSort: sampleLinkSort.value
  })
);
const { t } = useI18n();

function loadSamplePage(page: number) {
  sampleLinkOffset.value = Math.max(0, page - 1) * SAMPLE_LINK_PAGE_SIZE;
  syncSampleQueryToHash();
  void run();
}

function updateSampleQuery(query: { capabilityBlockId: string; modelId: string; sort: SampleForecastSort }) {
  sampleLinkCapabilityBlockId.value = query.capabilityBlockId;
  sampleLinkModelId.value = query.modelId;
  sampleLinkSort.value = query.sort;
  sampleLinkOffset.value = 0;
  syncSampleQueryToHash();
  void run();
}

function readSampleQueryFromHash() {
  const query = window.location.hash.split('?')[1] || '';
  const params = new URLSearchParams(query);
  return {
    offset: Math.max(0, Number(params.get('sample_link_offset') || 0) || 0),
    capabilityBlockId: params.get('sample_link_capability_block_id') || '',
    modelId: params.get('sample_link_model_id') || '',
    sort: parseSampleSort(params.get('sample_link_sort')),
  };
}

function parseSampleSort(value: string | null): SampleForecastSort {
  return value === 'metric_desc' || value === 'metric_asc' || value === 'sample_index' ? value : 'sample_index';
}

function syncSampleQueryToHash() {
  const [path, query = ''] = window.location.hash.slice(1).split('?');
  if (path !== `/reports/${props.reportId}`) return;
  const params = new URLSearchParams(query);
  setQueryParam(params, 'sample_link_offset', sampleLinkOffset.value > 0 ? String(sampleLinkOffset.value) : '');
  setQueryParam(params, 'sample_link_capability_block_id', sampleLinkCapabilityBlockId.value);
  setQueryParam(params, 'sample_link_model_id', sampleLinkModelId.value);
  setQueryParam(params, 'sample_link_sort', sampleLinkSort.value === 'sample_index' ? '' : sampleLinkSort.value);
  const next = `#${path}${params.toString() ? `?${params.toString()}` : ''}`;
  window.history.replaceState(window.history.state, '', next);
}

function setQueryParam(params: URLSearchParams, key: string, value: string) {
  if (value) params.set(key, value);
  else params.delete(key);
}

onMounted(run);
</script>
