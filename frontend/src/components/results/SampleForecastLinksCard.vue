<template>
  <article class="card">
    <header class="card-head sample-forecast-head">
      <div>
        <h2 class="card-title">{{ t('results.sampleForecastLinks') }}</h2>
        <p class="faint" style="margin:4px 0 0;font-size:0.82rem">{{ samplePageLabel || t('results.noSampleForecasts') }}</p>
      </div>
      <div class="sample-forecast-controls">
        <label class="field compact-field" for="sample-forecast-group">
          <span class="field-label">{{ t('results.sampleTestGroup') }}</span>
          <select id="sample-forecast-group" :value="selectedSampleCapabilityBlockId" :aria-label="t('results.sampleTestGroup')" @change="onSampleCapabilityChange">
            <option value="">{{ t('results.allTestGroups') }}</option>
            <option v-for="option in sampleCapabilityOptions" :key="option.id" :value="option.id">{{ option.label }}</option>
          </select>
        </label>
        <label class="field compact-field" for="sample-forecast-sort">
          <span class="field-label">{{ t('results.sampleSort') }}</span>
          <select id="sample-forecast-sort" :value="selectedSampleSort" :aria-label="t('results.sampleSort')" @change="onSampleSortChange">
            <option value="sample_index">{{ t('results.sampleSortWindow') }}</option>
            <option value="metric_desc">{{ t('results.sampleSortMetricDesc', { metric: sampleMetricLabel }) }}</option>
            <option value="metric_asc">{{ t('results.sampleSortMetricAsc', { metric: sampleMetricLabel }) }}</option>
          </select>
        </label>
        <label class="field compact-field" for="sample-forecast-model">
          <span class="field-label">{{ t('results.sampleModel') }}</span>
          <select id="sample-forecast-model" :value="selectedSampleModelId" :aria-label="t('results.sampleModel')" @change="onSampleModelChange">
            <option value="">{{ t('results.allModels') }}</option>
            <option v-for="option in sampleModelOptions" :key="option.id" :value="option.id">{{ option.label }}</option>
          </select>
        </label>
        <span class="badge">{{ formatInt(serverSampleTotal ?? sampleLinks.length) }}</span>
      </div>
    </header>
    <div class="card-body">
      <p v-if="!sampleLinks.length" class="empty-state">{{ t('results.noSampleForecasts') }}</p>
      <div v-else class="stack">
        <div class="table-wrap">
          <table class="data">
            <caption>{{ samplePageLabel }}</caption>
            <thead>
              <tr>
                <th>{{ t('results.sample') }}</th>
                <th>{{ t('results.window') }}</th>
                <th class="num">{{ t('results.sampleMetricError', { metric: sampleMetricLabel }) }}</th>
                <th>{{ t('results.models') }}</th>
                <th>{{ t('common.actions') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(link, index) in visibleSampleLinks" :key="`${link.run_id}:${link.sample_id}`">
                <td>
                  <span style="font-weight:650">{{ sampleLabel(link) }}</span>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(link.sample_id) }}</div>
                  <div v-if="sampleGroupLabel(link)" class="faint" style="font-size:0.74rem">{{ sampleGroupLabel(link) }}</div>
                </td>
                <td>
                  <span>{{ sampleWindowSummary(link) }}</span>
                  <div class="faint" style="font-size:0.74rem">{{ sampleTimeSummary(link) }}</div>
                </td>
                <td class="num"><span class="mono">{{ sampleMetricValue(link) }}</span></td>
                <td><span class="badge neutral">{{ modelCountText(link.model_count ?? source.model_metrics.length) }}</span></td>
                <td>
                  <a class="btn secondary sm" :href="sampleHref(link, index)">
                    <Icon name="lineChart" :size="14" /> {{ t('common.open') }}
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="sample-pager" :aria-label="t('results.samplePagination')">
          <button class="btn ghost sm" type="button" :disabled="activeSamplePage <= 1" @click="goSamplePage(activeSamplePage - 1)">
            <Icon name="chevronLeft" :size="14" /> {{ t('results.previousPage') }}
          </button>
          <span class="status-line">{{ t('results.pageStatus', { page: activeSamplePage, pages: samplePageCount }) }}</span>
          <button class="btn ghost sm" type="button" :disabled="activeSamplePage >= samplePageCount" @click="goSamplePage(activeSamplePage + 1)">
            {{ t('results.nextPage') }} <Icon name="chevronRight" :size="14" />
          </button>
        </div>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import type { CapabilityBlockReportDTO, SampleForecastLinkDTO, SampleForecastSort } from '../../api/types';
import { useFormat } from '../../composables/useFormat';
import { useModels } from '../../composables/useModels';
import { parseDateTime, shortId } from '../../lib/format';
import { syntheticCapabilityLabel } from '../../lib/syntheticCapabilities';
import Icon from '../ui/Icon.vue';

export interface SampleForecastLinkSource {
  track_id?: string | null;
  report_id?: string | null;
  model_metrics: Array<Record<string, unknown>>;
  capability_blocks?: CapabilityBlockReportDTO[];
  sample_forecast_links: SampleForecastLinkDTO[];
  sample_forecast_links_total?: number;
  sample_forecast_links_limit?: number;
  sample_forecast_links_offset?: number;
  sample_forecast_links_capability_block_id?: string | null;
  sample_forecast_links_metric?: string | null;
  sample_forecast_links_model_id?: string | null;
  sample_forecast_links_sort?: SampleForecastSort;
}

const props = defineProps<{
  source: SampleForecastLinkSource;
  sampleCapabilityBlockId?: string;
  sampleModelId?: string;
  sampleSort?: SampleForecastSort;
}>();
const emit = defineEmits<{
  (event: 'sample-page-change', page: number): void;
  (event: 'sample-query-change', query: { capabilityBlockId: string; modelId: string; sort: SampleForecastSort }): void;
}>();
const { t } = useI18n();
const { modelName } = useModels();
const { formatNumber, formatInt, locale } = useFormat();
const samplePage = ref(1);
const SAMPLE_PAGE_SIZE = 10;

const metricKeys = computed(() => {
  const keys = new Set<string>();
  props.source.model_metrics.forEach((item) => {
    const metrics = (item.metrics || {}) as Record<string, unknown>;
    Object.entries(metrics).forEach(([key, value]) => {
      const numeric = typeof value === 'number' ? value : Number(value);
      if (Number.isFinite(numeric)) keys.add(key);
    });
  });
  return [...keys];
});
const sampleMetricLabel = computed(() => (props.source.sample_forecast_links_metric || metricKeys.value[0] || 'mse').toUpperCase());
const selectedSampleCapabilityBlockId = computed(() => props.sampleCapabilityBlockId ?? props.source.sample_forecast_links_capability_block_id ?? '');
const selectedSampleModelId = computed(() => props.sampleModelId ?? props.source.sample_forecast_links_model_id ?? '');
const selectedSampleSort = computed<SampleForecastSort>(() => props.sampleSort ?? props.source.sample_forecast_links_sort ?? 'sample_index');
const sampleCapabilityOptions = computed(() =>
  (props.source.capability_blocks || []).map((block) => ({
    id: block.capability_block_id,
    label: capabilityBlockLabel(block),
  }))
);
const sampleModelOptions = computed(() =>
  props.source.model_metrics.map((item) => {
    const id = String(item.model_id || '');
    return {
      id,
      label: String(item.model_name || modelName(id) || id),
    };
  }).filter((item) => item.id)
);

const sampleLinks = computed(() => {
  const bySample = new Map<string, SampleForecastLinkDTO>();
  for (const link of props.source.sample_forecast_links || []) {
    const key = `${link.run_id}:${link.sample_id}`;
    const current = bySample.get(key);
    if (!current) {
      bySample.set(key, { ...link });
      continue;
    }
    bySample.set(key, {
      ...current,
      ...Object.fromEntries(Object.entries(link).filter(([, value]) => value !== undefined && value !== null)),
      model_count: Math.max(Number(current.model_count ?? 0), Number(link.model_count ?? 0), props.source.model_metrics.length),
    });
  }
  const links = [...bySample.values()];
  if (serverPagedSamples.value) return links;
  return links.sort((a, b) => {
    const ai = typeof a.sample_index === 'number' ? a.sample_index : Number.POSITIVE_INFINITY;
    const bi = typeof b.sample_index === 'number' ? b.sample_index : Number.POSITIVE_INFINITY;
    if (ai !== bi) return ai - bi;
    return a.sample_id.localeCompare(b.sample_id);
  });
});

const serverSampleTotal = computed(() => {
  const total = props.source.sample_forecast_links_total;
  return typeof total === 'number' && Number.isFinite(total) ? Math.max(0, total) : null;
});
const serverSampleLimit = computed(() => {
  const limit = props.source.sample_forecast_links_limit;
  return typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? limit : SAMPLE_PAGE_SIZE;
});
const serverSampleOffset = computed(() => {
  const offset = props.source.sample_forecast_links_offset;
  return typeof offset === 'number' && Number.isFinite(offset) ? Math.max(0, offset) : 0;
});
const serverPagedSamples = computed(() => serverSampleTotal.value !== null);
const samplePageCount = computed(() => {
  const total = serverSampleTotal.value ?? sampleLinks.value.length;
  const pageSize = serverPagedSamples.value ? serverSampleLimit.value : SAMPLE_PAGE_SIZE;
  return Math.max(1, Math.ceil(total / pageSize));
});
const activeSamplePage = computed(() => {
  if (!serverPagedSamples.value) return samplePage.value;
  return Math.floor(serverSampleOffset.value / serverSampleLimit.value) + 1;
});
const visibleSampleLinks = computed(() => {
  if (serverPagedSamples.value) return sampleLinks.value;
  const start = (samplePage.value - 1) * SAMPLE_PAGE_SIZE;
  return sampleLinks.value.slice(start, start + SAMPLE_PAGE_SIZE);
});
const samplePageLabel = computed(() => {
  const total = serverSampleTotal.value ?? sampleLinks.value.length;
  if (!total) return '';
  const start = serverPagedSamples.value ? serverSampleOffset.value + 1 : (samplePage.value - 1) * SAMPLE_PAGE_SIZE + 1;
  const end = serverPagedSamples.value
    ? Math.min(serverSampleOffset.value + sampleLinks.value.length, total)
    : Math.min(samplePage.value * SAMPLE_PAGE_SIZE, total);
  return t('results.samplePageRange', { start, end, total });
});

watch(samplePageCount, (count) => {
  if (serverPagedSamples.value) return;
  if (samplePage.value > count) samplePage.value = count;
});

function goSamplePage(page: number) {
  const next = Math.max(1, Math.min(page, samplePageCount.value));
  if (serverPagedSamples.value) {
    emit('sample-page-change', next);
    return;
  }
  samplePage.value = next;
}

function emitSampleQuery(patch: Partial<{ capabilityBlockId: string; modelId: string; sort: SampleForecastSort }>) {
  emit('sample-query-change', {
    capabilityBlockId: patch.capabilityBlockId ?? selectedSampleCapabilityBlockId.value,
    modelId: patch.modelId ?? selectedSampleModelId.value,
    sort: patch.sort ?? selectedSampleSort.value,
  });
}

function onSampleCapabilityChange(event: Event) {
  emitSampleQuery({ capabilityBlockId: (event.target as HTMLSelectElement).value });
}

function onSampleSortChange(event: Event) {
  emitSampleQuery({ sort: (event.target as HTMLSelectElement).value as SampleForecastSort });
}

function onSampleModelChange(event: Event) {
  emitSampleQuery({ modelId: (event.target as HTMLSelectElement).value });
}

function sampleLabel(link: SampleForecastLinkDTO) {
  if (typeof link.sample_index === 'number') return t('results.sampleWindowLabel', { index: link.sample_index + 1 });
  return t('results.sampleFallbackLabel', { id: shortId(link.sample_id) });
}

function sampleWindowSummary(link: SampleForecastLinkDTO) {
  if (typeof link.horizon_start === 'number' && typeof link.horizon_end === 'number') {
    return t('results.sampleHorizonRows', { start: link.horizon_start, end: link.horizon_end });
  }
  if (typeof link.context_start === 'number' && typeof link.context_end === 'number') {
    return t('results.sampleContextRows', { start: link.context_start, end: link.context_end });
  }
  return t('common.notAvailable');
}

function sampleTimeSummary(link: SampleForecastLinkDTO) {
  if (link.forecast_start_at && link.forecast_end_at) {
    return t('results.sampleForecastTimeRange', { start: compactDateTime(link.forecast_start_at), end: compactDateTime(link.forecast_end_at) });
  }
  return t('results.sampleNoTimestamp');
}

function sampleGroupLabel(link: SampleForecastLinkDTO) {
  const block = (props.source.capability_blocks || []).find((item) => item.capability_block_id === link.capability_block_id);
  if (block) return capabilityBlockLabel(block);
  const type = link.capability_type;
  const fallback = link.capability_label || link.capability_block_name || '';
  return syntheticCapabilityLabel(type, fallback, t);
}

function capabilityBlockLabel(block: CapabilityBlockReportDTO) {
  if (block.block_type === 'synthetic') {
    return syntheticCapabilityLabel(block.capability_type, block.capability_label || block.name, t);
  }
  return block.name || block.capability_label || block.capability_type || block.capability_block_id;
}

function sampleMetricValue(link: SampleForecastLinkDTO) {
  return typeof link.metric_value === 'number' && Number.isFinite(link.metric_value) ? formatNumber(link.metric_value) : t('common.notAvailable');
}

function compactDateTime(value: string) {
  const date = parseDateTime(value);
  if (!date) return value;
  return date.toLocaleString(locale.value, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function modelCountText(count: number | null | undefined) {
  const safe = Math.max(0, Number(count || 0));
  return t(safe === 1 ? 'results.modelCountInlineOne' : 'results.modelCountInlineOther', { count: safe });
}

function sampleHref(link: SampleForecastLinkDTO, index: number) {
  const params = new URLSearchParams();
  if (props.source.track_id) {
    params.set('track_id', props.source.track_id);
  } else {
    params.set('run_id', link.run_id);
    if (props.source.report_id) params.set('report_id', props.source.report_id);
  }
  const cursorOffset = (serverPagedSamples.value ? serverSampleOffset.value : (samplePage.value - 1) * SAMPLE_PAGE_SIZE) + index;
  params.set('sample_link_offset', String(serverPagedSamples.value ? serverSampleOffset.value : (samplePage.value - 1) * SAMPLE_PAGE_SIZE));
  params.set('sample_cursor_offset', String(cursorOffset));
  if (selectedSampleCapabilityBlockId.value) params.set('sample_link_capability_block_id', selectedSampleCapabilityBlockId.value);
  if (selectedSampleModelId.value) params.set('sample_link_model_id', selectedSampleModelId.value);
  if (props.source.sample_forecast_links_metric) params.set('sample_link_metric', props.source.sample_forecast_links_metric);
  if (selectedSampleSort.value !== 'sample_index') params.set('sample_link_sort', selectedSampleSort.value);
  return `#/samples/${encodeURIComponent(link.sample_id)}?${params.toString()}`;
}
</script>
