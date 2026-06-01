<template>
  <div class="chart-card">
    <div class="chart-toolbar">
      <strong>{{ t('sampleWindow.chartTitle') }}</strong>
      <span class="spacer" />
      <label v-if="dimCount > 1" class="field" style="flex-direction:row;align-items:center;gap:8px">
        <span class="faint" style="font-size:0.82rem">{{ t('results.dimension') }}</span>
        <select v-model.number="dim" :aria-label="t('results.targetDimension')" style="min-height:32px;width:auto">
          <option v-for="d in dimCount" :key="d" :value="d - 1">{{ t('results.dim', { index: d - 1 }) }}</option>
        </select>
      </label>
    </div>

    <div v-if="!valid" class="state-block">
      <span class="state-icon"><Icon name="lineChart" :size="24" /></span>
      <p class="state-desc">{{ t('sampleWindow.emptyChart') }}</p>
    </div>

    <svg
      v-else
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

      <rect v-if="futureLen" class="band" :x="boundaryX" :y="M.top" :width="(W - M.right) - boundaryX" :height="innerH" />
      <line v-if="futureLen" class="chart-hover-x" :x1="boundaryX" :y1="M.top" :x2="boundaryX" :y2="M.top + innerH" />
      <text v-if="futureLen" class="tick" :x="boundaryX + 6" :y="M.top + 14" fill="var(--text-faint)">{{ t('sampleWindow.future') }}</text>

      <g class="axis tick">
        <text v-for="tick in xTicks" :key="`x-${tick.i}`" :x="tick.x" :y="H - M.bottom + 18" text-anchor="middle">{{ tick.label }}</text>
      </g>

      <polyline
        v-for="series in seriesList"
        :key="series.key"
        :points="pathFor(series.points)"
        fill="none"
        :stroke="series.color"
        :stroke-width="series.width"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    </svg>

    <div v-if="valid" class="legend" role="group" :aria-label="t('results.seriesLegend')">
      <span v-for="series in seriesList" :key="series.key" class="legend-chip" style="cursor:default">
        <span class="swatch" :style="{ background: series.color }" />
        {{ series.label }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import type { SamplePreviewDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';

const props = defineProps<{ sample: SamplePreviewDTO }>();
const { t } = useI18n();
const { formatNumber } = useFormat();

const W = 760;
const H = 320;
const M = { top: 18, right: 20, bottom: 30, left: 52 };
const innerW = W - M.left - M.right;
const innerH = H - M.top - M.bottom;
const dim = ref(0);

const history = computed(() => props.sample.target_history || []);
const future = computed(() => props.sample.target_future || []);
const histLen = computed(() => history.value.length);
const futureLen = computed(() => future.value.length);
const totalLen = computed(() => histLen.value + futureLen.value);
const dimCount = computed(() => Math.max(1, history.value[0]?.length || future.value[0]?.length || 1));
const valid = computed(() => totalLen.value > 0 && seriesList.value.some((series) => series.points.length));

interface Point { i: number; v: number }
interface Series { key: string; label: string; color: string; width: number; points: Point[] }

function at(rows: number[][], index: number): number | undefined {
  const value = rows[index]?.[dim.value];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

const seriesList = computed<Series[]>(() => {
  const historyPoints: Point[] = [];
  for (let i = 0; i < history.value.length; i += 1) {
    const value = at(history.value, i);
    if (value !== undefined) historyPoints.push({ i, v: value });
  }

  const futurePoints: Point[] = [];
  const connector = historyPoints.at(-1);
  if (connector) futurePoints.push(connector);
  for (let i = 0; i < future.value.length; i += 1) {
    const value = at(future.value, i);
    if (value !== undefined) futurePoints.push({ i: histLen.value + i, v: value });
  }

  return [
    { key: 'history', label: t('sampleWindow.history'), color: 'var(--chart-actual)', width: 2.2, points: historyPoints },
    { key: 'future', label: t('sampleWindow.future'), color: 'var(--chart-truth)', width: 2.4, points: futurePoints },
  ].filter((series) => series.points.length);
});

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

const ariaLabel = computed(() => t('sampleWindow.chartAria', {
  history: t(histLen.value === 1 ? 'sampleWindow.historyStepsOne' : 'sampleWindow.historyStepsOther', { count: histLen.value }),
  future: t(futureLen.value === 1 ? 'sampleWindow.futureStepsOne' : 'sampleWindow.futureStepsOther', { count: futureLen.value }),
}));
</script>
