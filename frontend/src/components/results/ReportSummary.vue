<template>
  <div class="stack">
    <article class="card">
      <header class="card-head">
        <h2 class="card-title"><Icon name="barChart" :size="18" style="vertical-align:-3px;margin-right:6px" />{{ t('results.modelMetrics') }}</h2>
        <span class="badge">{{ modelCountLabel }}</span>
      </header>
      <div class="card-body">
        <div v-if="!report.model_metrics.length" class="empty-state">{{ t('results.noModelMetrics') }}</div>
        <div v-else class="table-wrap">
          <table class="data">
            <caption>{{ t('results.lowerIsBetterBest') }}</caption>
            <thead>
              <tr>
                <th>{{ t('results.model') }}</th>
                <th
                  v-for="key in metricKeys"
                  :key="key"
                  class="num"
                  :aria-sort="sortKey === key ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'"
                >
                  <button
                    class="btn ghost sm"
                    type="button"
                    :aria-label="t('results.sortByMetric', { metric: key.toUpperCase() })"
                    @click="sortBy(key)"
                  >
                    {{ key.toUpperCase() }}
                    <span v-if="sortKey === key" class="faint">{{ sortDir === 'asc' ? t('results.sortAscShort') : t('results.sortDescShort') }}</span>
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="m in rows" :key="m.id">
                <td>
                  <span style="font-weight:600">{{ m.name }}</span>
                  <div v-if="m.name !== m.id" class="faint mono" style="font-size:0.74rem">{{ shortId(m.id) }}</div>
                </td>
                <td v-for="key in metricKeys" :key="key" class="num">
                  <span :style="m.best[key] ? 'font-weight:700;color:var(--success-text)' : ''">{{ formatNumber(m.metrics[key]) }}</span>
                  <Icon v-if="m.best[key]" name="check" :size="12" style="margin-left:4px;color:var(--success)" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </article>

    <div class="grid-2">
      <article class="card">
        <header class="card-head"><h2 class="card-title">{{ t('results.taskOutcomes') }}</h2><span class="badge">{{ formatInt(report.task_summaries.length) }}</span></header>
        <div class="card-body stack">
          <div v-if="!report.task_summaries.length" class="empty-state">{{ t('results.noTasks') }}</div>
          <div v-for="task in report.task_summaries" :key="String(task.task_id)" class="detail-item" style="display:grid;gap:8px">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
              <span class="mono faint" style="font-size:0.78rem">{{ shortId(String(task.task_id)) }}</span>
              <StatusBadge :status="String(task.status)" />
            </div>
            <p v-if="task.error_message" class="alert" style="padding:8px 10px">{{ task.error_message }}</p>
          </div>
        </div>
      </article>

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('results.sampleForecasts') }}</h2>
          <span class="badge">{{ formatInt(sampleLinks.length) }}</span>
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
                    <th>{{ t('results.models') }}</th>
                    <th>{{ t('common.actions') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="link in visibleSampleLinks" :key="`${link.run_id}:${link.sample_id}`">
                    <td>
                      <span style="font-weight:650">{{ sampleLabel(link) }}</span>
                      <div class="faint mono" style="font-size:0.74rem">{{ shortId(link.sample_id) }}</div>
                    </td>
                    <td>
                      <span>{{ sampleWindowSummary(link) }}</span>
                      <div class="faint" style="font-size:0.74rem">{{ sampleTimeSummary(link) }}</div>
                    </td>
                    <td><span class="badge neutral">{{ modelCountText(link.model_count ?? report.model_metrics.length) }}</span></td>
                    <td>
                      <a class="btn secondary sm" :href="sampleHref(link)">
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import StatusBadge from '../ui/StatusBadge.vue';
import type { ReportDTO, SampleForecastLinkDTO } from '../../api/types';
import { useModels } from '../../composables/useModels';
import { useFormat } from '../../composables/useFormat';
import { shortId } from '../../lib/format';

const props = defineProps<{ report: ReportDTO }>();
const emit = defineEmits<{ (event: 'sample-page-change', page: number): void }>();
const { modelName } = useModels();
const { t } = useI18n();
const { formatNumber, formatInt, locale } = useFormat();
const sortKey = ref('');
const sortDir = ref<'asc' | 'desc'>('asc');
const samplePage = ref(1);
const SAMPLE_PAGE_SIZE = 10;
const modelCountLabel = computed(() =>
  t(props.report.model_metrics.length === 1 ? 'results.modelCountOne' : 'results.modelCountOther', { count: props.report.model_metrics.length })
);

const KNOWN_ORDER = ['mase', 'mse', 'rmse', 'mae', 'smape', 'mape'];

function metricsOf(m: Record<string, unknown>): Record<string, number> {
  const raw = (m.metrics ?? {}) as Record<string, unknown>;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    const n = typeof v === 'number' ? v : Number(v);
    if (Number.isFinite(n)) out[k] = n;
  }
  return out;
}

const metricKeys = computed(() => {
  const keys = new Set<string>();
  props.report.model_metrics.forEach((m) => Object.keys(metricsOf(m)).forEach((k) => keys.add(k)));
  return [...keys].sort((a, b) => {
    const ia = KNOWN_ORDER.indexOf(a);
    const ib = KNOWN_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
});

const bestByMetric = computed(() => {
  const best: Record<string, number> = {};
  props.report.model_metrics.forEach((m) => {
    const mm = metricsOf(m);
    metricKeys.value.forEach((k) => {
      if (k in mm && (!(k in best) || mm[k] < best[k])) best[k] = mm[k];
    });
  });
  return best;
});

const unsortedRows = computed(() =>
  props.report.model_metrics.map((m) => {
    const id = String(m.model_id ?? '');
    const metrics = metricsOf(m);
    const best: Record<string, boolean> = {};
    metricKeys.value.forEach((k) => {
      best[k] = props.report.model_metrics.length > 1 && k in metrics && metrics[k] === bestByMetric.value[k];
    });
    return { id, name: modelName(id) || id, metrics, best };
  })
);

const rows = computed(() => {
  if (!sortKey.value) return unsortedRows.value;
  const direction = sortDir.value === 'asc' ? 1 : -1;
  return [...unsortedRows.value].sort((a, b) => {
    const av = a.metrics[sortKey.value];
    const bv = b.metrics[sortKey.value];
    const aMissing = av == null || !Number.isFinite(av);
    const bMissing = bv == null || !Number.isFinite(bv);
    if (aMissing && bMissing) return a.name.localeCompare(b.name);
    if (aMissing) return 1;
    if (bMissing) return -1;
    if (av === bv) return a.name.localeCompare(b.name);
    return (av - bv) * direction;
  });
});

const sampleLinks = computed(() => {
  const bySample = new Map<string, SampleForecastLinkDTO>();
  for (const link of props.report.sample_forecast_links || []) {
    const key = `${link.run_id}:${link.sample_id}`;
    const current = bySample.get(key);
    if (!current) {
      bySample.set(key, { ...link });
      continue;
    }
    bySample.set(key, {
      ...current,
      ...Object.fromEntries(Object.entries(link).filter(([, value]) => value !== undefined && value !== null)),
      model_count: Math.max(Number(current.model_count ?? 0), Number(link.model_count ?? 0), props.report.model_metrics.length),
    });
  }
  return [...bySample.values()].sort((a, b) => {
    const ai = typeof a.sample_index === 'number' ? a.sample_index : Number.POSITIVE_INFINITY;
    const bi = typeof b.sample_index === 'number' ? b.sample_index : Number.POSITIVE_INFINITY;
    if (ai !== bi) return ai - bi;
    return a.sample_id.localeCompare(b.sample_id);
  });
});

const serverSampleTotal = computed(() => {
  const total = props.report.sample_forecast_links_total;
  return typeof total === 'number' && Number.isFinite(total) ? Math.max(0, total) : null;
});
const serverSampleLimit = computed(() => {
  const limit = props.report.sample_forecast_links_limit;
  return typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? limit : SAMPLE_PAGE_SIZE;
});
const serverSampleOffset = computed(() => {
  const offset = props.report.sample_forecast_links_offset;
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

function sortBy(key: string) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc';
    return;
  }
  sortKey.value = key;
  sortDir.value = 'asc';
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

function compactDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
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

function sampleHref(link: SampleForecastLinkDTO) {
  const params = new URLSearchParams({ run_id: link.run_id, report_id: props.report.report_id });
  return `#/samples/${encodeURIComponent(link.sample_id)}?${params.toString()}`;
}
</script>
