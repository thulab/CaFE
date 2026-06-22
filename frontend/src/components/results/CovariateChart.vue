<template>
  <div v-if="valid" class="chart-card">
    <div class="chart-toolbar">
      <strong>{{ t('results.covariateTitle') }}</strong>
    </div>

    <svg
      class="chart-svg"
      :viewBox="`0 0 ${W} ${H}`"
      preserveAspectRatio="none"
      role="img"
      :aria-label="ariaLabel"
    >
      <g class="grid">
        <line v-for="tick in yTicks" :key="`grid-${tick.v}`" :x1="M.left" :y1="tick.y" :x2="W - M.right" :y2="tick.y" />
      </g>
      <g class="axis tick">
        <text v-for="tick in yTicks" :key="`y-${tick.v}`" :x="M.left - 8" :y="tick.y + 4" text-anchor="end">{{ tick.label }}</text>
      </g>

      <rect class="band" :x="boundaryX" :y="M.top" :width="(W - M.right) - boundaryX" :height="innerH" />
      <line class="chart-hover-x" :x1="boundaryX" :y1="M.top" :x2="boundaryX" :y2="M.top + innerH" />
      <text class="tick" :x="boundaryX + 6" :y="M.top + 14" fill="var(--text-faint)">{{ t('results.covariateFutureRegion') }}</text>

      <g class="axis tick">
        <text v-for="tick in xTicks" :key="`x-${tick.i}`" :x="tick.x" :y="H - M.bottom + 18" text-anchor="middle">{{ tick.label }}</text>
      </g>

      <polyline
        v-for="series in seriesList"
        :key="series.key"
        :points="pathFor(series.points)"
        fill="none"
        :stroke="series.color"
        stroke-width="2"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    </svg>

    <div class="legend" role="group" :aria-label="t('results.seriesLegend')">
      <span v-for="series in seriesList" :key="series.key" class="legend-chip" style="cursor:default">
        <span class="swatch" :style="{ background: series.color }" />
        {{ series.label }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import type { SampleForecastDTO, SamplePreviewDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';

const props = defineProps<{ sample: Pick<SampleForecastDTO | SamplePreviewDTO, 'covariate_column_names' | 'history_cov' | 'future_cov'> }>();
const { t } = useI18n();
const { formatNumber } = useFormat();

const W = 760;
const H = 260;
const M = { top: 18, right: 20, bottom: 30, left: 52 };
const innerW = W - M.left - M.right;
const innerH = H - M.top - M.bottom;

const names = computed(() => props.sample.covariate_column_names || []);
const history = computed(() => props.sample.history_cov || []);
const future = computed(() => props.sample.future_cov || []);
const histLen = computed(() => history.value.length);
const futureLen = computed(() => future.value.length);
const totalLen = computed(() => histLen.value + futureLen.value);
const valid = computed(() => names.value.length > 0 && totalLen.value > 0);

interface Point { i: number; v: number }
interface Series { key: string; label: string; color: string; points: Point[] }

const seriesList = computed<Series[]>(() => names.value.map((name, dim) => {
  const points: Point[] = [];
  history.value.forEach((row, index) => addPoint(points, index, row?.[dim]));
  future.value.forEach((row, index) => addPoint(points, histLen.value + index, row?.[dim]));
  return { key: name, label: name, color: `var(--series-${(dim % 8) + 1})`, points };
}).filter((series) => series.points.length));

const yDomain = computed(() => {
  const values = seriesList.value.flatMap((series) => series.points.map((point) => point.v));
  if (!values.length) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
});

function addPoint(points: Point[], index: number, value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) points.push({ i: index, v: value });
}

function xScale(index: number): number {
  if (totalLen.value <= 1) return M.left + innerW / 2;
  return M.left + (index / (totalLen.value - 1)) * innerW;
}

function yScale(value: number): number {
  const { min, max } = yDomain.value;
  return M.top + innerH - ((value - min) / (max - min)) * innerH;
}

function pathFor(points: Point[]): string {
  return points.map((point) => `${xScale(point.i).toFixed(1)},${yScale(point.v).toFixed(1)}`).join(' ');
}

const boundaryX = computed(() => xScale(Math.max(0, histLen.value - 1)));

const yTicks = computed(() => {
  const { min, max } = yDomain.value;
  return Array.from({ length: 5 }, (_, index) => {
    const value = min + ((max - min) * index) / 4;
    return { v: value, y: yScale(value), label: formatNumber(value, 2) };
  });
});

const xTicks = computed(() => {
  const last = totalLen.value - 1;
  const indices = [0, Math.max(0, histLen.value - 1), last].filter((value, index, array) => array.indexOf(value) === index);
  return indices.map((i) => ({ i, x: xScale(i), label: String(i) }));
});

const ariaLabel = computed(() => t('results.covariateAria', {
  history: t(histLen.value === 1 ? 'results.covariateAriaHistoryOne' : 'results.covariateAriaHistoryOther', { count: histLen.value }),
  future: t(futureLen.value === 1 ? 'results.covariateAriaFutureOne' : 'results.covariateAriaFutureOther', { count: futureLen.value }),
  covariates: t(names.value.length === 1 ? 'results.covariateAriaCovariateOne' : 'results.covariateAriaCovariateOther', { count: names.value.length }),
}));
</script>
