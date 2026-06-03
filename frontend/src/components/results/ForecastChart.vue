<template>
  <div class="chart-card">
    <div class="chart-toolbar">
      <strong>{{ t('results.forecastTitle') }}</strong>
      <span class="spacer" />
      <label v-if="dimCount > 1" class="field" style="flex-direction:row;align-items:center;gap:8px">
        <span class="faint" style="font-size:0.82rem">{{ t('results.dimension') }}</span>
        <select v-model.number="dim" :aria-label="t('results.targetDimension')" style="min-height:32px;width:auto">
          <option v-for="d in dimCount" :key="d" :value="d - 1">{{ targetDimLabel(d - 1) }}</option>
        </select>
      </label>
    </div>

    <div v-if="!valid" class="state-block">
      <span class="state-icon"><Icon name="lineChart" :size="24" /></span>
      <p class="state-desc">{{ t('results.noNumericSeries') }}</p>
    </div>

    <div v-else class="chart-svg-wrap" style="position:relative">
      <svg
        ref="svgEl"
        class="chart-svg"
        :viewBox="`0 0 ${W} ${H}`"
        preserveAspectRatio="none"
        role="img"
        :aria-label="ariaLabel"
        @mousemove="onMove"
        @mouseleave="hoverIndex = null"
      >
        <!-- horizontal gridlines + y ticks -->
        <g class="grid">
          <line v-for="t in yTicks" :key="`g${t.v}`" :x1="M.left" :y1="t.y" :x2="W - M.right" :y2="t.y" />
        </g>
        <g class="axis tick">
          <text v-for="t in yTicks" :key="`yt${t.v}`" :x="M.left - 8" :y="t.y + 4" text-anchor="end">{{ t.label }}</text>
        </g>

        <!-- forecast region shading + boundary -->
        <rect class="band" :x="boundaryX" :y="M.top" :width="(W - M.right) - boundaryX" :height="innerH" />
        <line class="chart-hover-x" :x1="boundaryX" :y1="M.top" :x2="boundaryX" :y2="M.top + innerH" />
        <text class="tick" :x="boundaryX + 6" :y="M.top + 14" fill="var(--text-faint)">{{ t('results.forecastRegion') }}</text>

        <!-- x ticks -->
        <g class="axis tick">
          <text v-for="t in xTicks" :key="`xt${t.i}`" :x="t.x" :y="H - M.bottom + 18" text-anchor="middle">{{ t.label }}</text>
        </g>

        <!-- series -->
        <polyline
          v-for="s in visibleSeries"
          :key="s.key"
          :points="pathFor(s)"
          fill="none"
          :stroke="s.color"
          :stroke-width="s.width || 2"
          :stroke-dasharray="s.dashed ? '6 5' : undefined"
          stroke-linejoin="round"
          stroke-linecap="round"
        />

        <!-- hover guide + markers -->
        <template v-if="hoverIndex !== null">
          <line class="chart-hover-x" :x1="xScale(hoverIndex)" :y1="M.top" :x2="xScale(hoverIndex)" :y2="M.top + innerH" />
          <circle
            v-for="pt in hoverPoints"
            :key="pt.key"
            :cx="xScale(hoverIndex)"
            :cy="yScale(pt.v)"
            r="3.5"
            :fill="pt.color"
            stroke="var(--surface)"
            stroke-width="1.5"
          />
        </template>
      </svg>

      <div
        v-if="hoverIndex !== null && hoverPoints.length"
        class="chart-tooltip"
        :style="{ left: tooltipLeft, top: tooltipTop }"
      >
        <div class="tt-head">{{ hoverHeading }}</div>
        <div v-for="pt in hoverPoints" :key="pt.key" class="tt-row">
          <span class="tt-label"><span class="tt-swatch" :style="{ background: pt.color }" />{{ pt.label }}</span>
          <span class="tt-val">{{ format(pt.v) }}</span>
        </div>
      </div>
    </div>

    <div v-if="valid" class="legend" role="group" :aria-label="t('results.seriesLegend')">
      <button
        v-for="s in series"
        :key="s.key"
        type="button"
        class="legend-chip"
        :class="{ 'is-off': hidden.has(s.key) }"
        :aria-pressed="!hidden.has(s.key)"
        @click="toggle(s.key)"
      >
        <span class="swatch" :class="{ dashed: s.dashed }" :style="{ background: s.dashed ? 'transparent' : s.color, color: s.color }" />
        {{ s.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import type { SampleForecastDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';

const props = defineProps<{ sample: SampleForecastDTO }>();
const { t } = useI18n();
const { formatNumber } = useFormat();

// internal coordinate space (CSS scales it responsively)
const W = 760;
const H = 320;
const M = { top: 18, right: 20, bottom: 30, left: 52 };
const innerW = W - M.left - M.right;
const innerH = H - M.top - M.bottom;

const dim = ref(0);
const svgEl = ref<SVGSVGElement | null>(null);
const hoverIndex = ref<number | null>(null);
const hidden = reactive(new Set<string>());

const history = computed(() => props.sample.target_history || []);
const future = computed(() => props.sample.target_future || []);
const histLen = computed(() => history.value.length);
const dimCount = computed(() => Math.max(1, history.value[0]?.length || future.value[0]?.length || 1));
const targetNames = computed(() => props.sample.target_column_names || []);

const successfulModels = computed(() =>
  props.sample.models.filter((m) => m.status === 'succeeded' && Array.isArray(m.forecast) && m.forecast!.length)
);

const futureLen = computed(() => {
  const fcLens = successfulModels.value.map((m) => m.forecast!.length);
  return Math.max(future.value.length, ...(fcLens.length ? fcLens : [0]));
});

const totalLen = computed(() => histLen.value + futureLen.value);
const valid = computed(() => totalLen.value > 0 && histLen.value > 0);

interface Series { key: string; label: string; color: string; dashed?: boolean; width?: number; points: Array<{ i: number; v: number }>; }

function at(rows: number[][], idx: number): number | undefined {
  const row = rows[idx];
  if (!row) return undefined;
  const v = row[dim.value];
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function targetDimLabel(index: number): string {
  return targetNames.value[index] || t('results.dim', { index });
}

const series = computed<Series[]>(() => {
  const out: Series[] = [];
  const h = history.value;
  const hl = histLen.value;

  const histPts: Array<{ i: number; v: number }> = [];
  for (let i = 0; i < hl; i += 1) {
    const v = at(h, i);
    if (v !== undefined) histPts.push({ i, v });
  }
  out.push({ key: 'history', label: t('results.history'), color: 'var(--chart-actual)', points: histPts });

  const connector = histPts.length ? histPts[histPts.length - 1] : undefined;

  const truthPts: Array<{ i: number; v: number }> = [];
  if (connector) truthPts.push(connector);
  for (let j = 0; j < future.value.length; j += 1) {
    const v = at(future.value, j);
    if (v !== undefined) truthPts.push({ i: hl + j, v });
  }
  if (truthPts.length > 1) out.push({ key: 'truth', label: t('results.groundTruth'), color: 'var(--chart-truth)', width: 2.4, points: truthPts });

  successfulModels.value.forEach((m, idx) => {
    const pts: Array<{ i: number; v: number }> = [];
    if (connector) pts.push(connector);
    (m.forecast || []).forEach((row, j) => {
      const v = row?.[dim.value];
      if (typeof v === 'number' && Number.isFinite(v)) pts.push({ i: hl + j, v });
    });
    if (pts.length > 1) {
      out.push({ key: m.model_id, label: m.model_name || m.model_id, color: `var(--series-${(idx % 8) + 1})`, dashed: true, points: pts });
    }
  });

  return out;
});

const visibleSeries = computed(() => series.value.filter((s) => !hidden.has(s.key)));

const yDomain = computed(() => {
  const vals: number[] = [];
  const pool = visibleSeries.value.length ? visibleSeries.value : series.value;
  pool.forEach((s) => s.points.forEach((p) => vals.push(p.v)));
  if (!vals.length) return { min: 0, max: 1 };
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
});

function xScale(i: number): number {
  if (totalLen.value <= 1) return M.left + innerW / 2;
  return M.left + (i / (totalLen.value - 1)) * innerW;
}

function yScale(v: number): number {
  const { min, max } = yDomain.value;
  return M.top + innerH - ((v - min) / (max - min)) * innerH;
}

function pathFor(s: Series): string {
  return s.points.map((p) => `${xScale(p.i).toFixed(1)},${yScale(p.v).toFixed(1)}`).join(' ');
}

const boundaryX = computed(() => xScale(Math.max(0, histLen.value - 1)));

const yTicks = computed(() => {
  const { min, max } = yDomain.value;
  const n = 4;
  return Array.from({ length: n + 1 }, (_, k) => {
    const v = min + ((max - min) * k) / n;
    return { v, y: yScale(v), label: formatNumber(v, 2) };
  });
});

const xTicks = computed(() => {
  const last = totalLen.value - 1;
  const idxs = [0, Math.max(0, histLen.value - 1), last].filter((v, i, a) => a.indexOf(v) === i);
  return idxs.map((i) => ({ i, x: xScale(i), label: String(i) }));
});

const hoverPoints = computed(() => {
  if (hoverIndex.value === null) return [];
  return visibleSeries.value
    .map((s) => {
      const p = s.points.find((pt) => pt.i === hoverIndex.value);
      return p ? { key: s.key, label: s.label, color: s.color, v: p.v } : null;
    })
    .filter((x): x is { key: string; label: string; color: string; v: number } => x !== null);
});

const tooltipLeft = computed(() => `${(xScale(hoverIndex.value ?? 0) / W) * 100}%`);
const tooltipTop = computed(() => {
  const ys = hoverPoints.value.map((p) => yScale(p.v));
  const y = ys.length ? Math.min(...ys) : M.top;
  return `${(y / H) * 100}%`;
});
const hoverHeading = computed(() => {
  const step = hoverIndex.value ?? 0;
  return t(step >= histLen.value ? 'results.stepHorizon' : 'results.stepContext', { step });
});

const ariaLabel = computed(() => {
  const models = successfulModels.value.length;
  const truth = future.value.length;
  return t('results.forecastAria', {
    history: t(histLen.value === 1 ? 'results.forecastAriaHistoryOne' : 'results.forecastAriaHistoryOther', { count: histLen.value }),
    truth: t(truth === 1 ? 'results.forecastAriaTruthOne' : 'results.forecastAriaTruthOther', { count: truth }),
    models: t(models === 1 ? 'results.forecastAriaModelOne' : 'results.forecastAriaModelOther', { count: models }),
  });
});

function onMove(e: MouseEvent) {
  const el = svgEl.value;
  if (!el || totalLen.value <= 1) return;
  const rect = el.getBoundingClientRect();
  const internalX = ((e.clientX - rect.left) / rect.width) * W;
  const ratio = (internalX - M.left) / innerW;
  const i = Math.round(ratio * (totalLen.value - 1));
  hoverIndex.value = Math.max(0, Math.min(totalLen.value - 1, i));
}

function toggle(key: string) {
  if (hidden.has(key)) hidden.delete(key);
  else hidden.add(key);
}

const format = (v: number) => formatNumber(v, 4);
</script>
