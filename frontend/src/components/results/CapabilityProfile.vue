<template>
  <article v-if="hasProfile" class="card">
    <header class="card-head">
      <h2 class="card-title">
        <Icon name="target" :size="18" style="vertical-align:-3px;margin-right:6px" />{{ t('results.capabilityProfile') }}
      </h2>
      <span class="badge">{{ t('results.lowerIsBetter') }}</span>
    </header>
    <div class="card-body stack">
      <div class="capability-profile-controls">
        <div class="field">
          <label class="label" for="capability-profile-metric">{{ t('results.profileMetric') }}</label>
          <select id="capability-profile-metric" v-model="metricKey">
            <option v-for="metric in metricOptions" :key="metric" :value="metric">{{ metric.toUpperCase() }}</option>
          </select>
        </div>
        <div class="field">
          <label class="label" for="capability-profile-scope">{{ t('results.profileScope') }}</label>
          <select id="capability-profile-scope" v-model="scope">
            <option value="synthetic">{{ t('results.syntheticOnly') }}</option>
            <option value="all">{{ t('results.allTestGroups') }}</option>
          </select>
        </div>
      </div>

      <div class="capability-model-selectors" :aria-label="t('results.profileModels')">
        <label
          v-for="model in modelOptions"
          :key="model.id"
          class="capability-model-chip"
          :class="{ 'is-off': !selectedModelIds.includes(model.id) }"
        >
          <input
            type="checkbox"
            :checked="selectedModelIds.includes(model.id)"
            :aria-label="t('results.selectProfileModel', { model: model.name })"
            @change="onModelToggle(model.id, $event)"
          />
          <span class="swatch" :style="{ background: model.color }" />
          <span>{{ model.name }}</span>
        </label>
      </div>

      <div v-if="showRadar" class="capability-profile-layout">
        <div class="capability-radar-wrap">
          <svg
            class="capability-radar chart-svg"
            viewBox="0 0 640 380"
            role="img"
            :aria-label="t('results.capabilityRadarAria', { count: radarAxes.length, models: radarSeries.length })"
          >
            <g class="radar-grid">
              <polygon
                v-for="ring in radarRings"
                :key="ring"
                :points="ringPoints(ring)"
                fill="none"
              />
              <line
                v-for="(axis, index) in radarAxes"
                :key="axis.id"
                :x1="RADAR.centerX"
                :y1="RADAR.centerY"
                :x2="axisPoint(index, 100).x"
                :y2="axisPoint(index, 100).y"
              />
            </g>
            <g class="radar-labels">
              <text
                v-for="(axis, index) in radarAxes"
                :key="axis.id"
                :x="axisPoint(index, 118).x"
                :y="axisPoint(index, 118).y"
                :text-anchor="labelAnchor(axisPoint(index, 118).x)"
              >
                {{ shortAxisLabel(axis.label) }}
              </text>
            </g>
            <g>
              <polygon
                v-for="series in radarSeries"
                :key="`${series.id}-area`"
                class="radar-area"
                :points="series.points"
                :style="{ stroke: series.color, fill: series.color }"
              />
              <polyline
                v-for="series in radarSeries"
                :key="`${series.id}-line`"
                class="radar-line"
                :points="series.points"
                :style="{ stroke: series.color }"
              />
              <circle
                v-for="point in radarPoints"
                :key="`${point.modelId}-${point.axisId}`"
                r="3"
                :cx="point.x"
                :cy="point.y"
                :style="{ fill: point.color }"
              />
            </g>
          </svg>
        </div>
        <div class="capability-radar-side">
          <h3 class="compact-title">{{ t('results.radarScores') }}</h3>
          <div v-for="series in radarSeries" :key="series.id" class="detail-item">
            <span class="capability-series-name"><span class="swatch-dot" :style="{ background: series.color }" />{{ series.name }}</span>
            <span>{{ t('results.scoreLabel', { score: Math.round(series.averageScore) }) }}</span>
          </div>
        </div>
      </div>

      <p v-else class="empty-state">{{ radarUnavailableText }}</p>

      <div class="table-wrap">
        <table class="data capability-matrix">
          <caption>{{ t('results.testGroupBreakdown') }}</caption>
          <thead>
            <tr>
              <th>{{ t('results.testGroup') }}</th>
              <th>{{ t('results.type') }}</th>
              <th class="num">{{ t('results.samples') }}</th>
              <th v-for="model in selectedModels" :key="model.id" class="num">{{ model.name }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in blockRows" :key="row.block.capability_block_id">
              <td>
                <span style="font-weight:650">{{ row.label }}</span>
                <div class="faint mono" style="font-size:0.74rem">{{ row.block.capability_type }}</div>
              </td>
              <td><span class="badge neutral">{{ blockTypeLabel(row.block) }}</span></td>
              <td class="num">{{ formatInt(Number(row.block.sample_count || 0)) }}</td>
              <td v-for="cell in row.cells" :key="cell.modelId" class="num">
                <div class="capability-metric-cell">
                  <span class="mono">{{ formatMaybe(cell.raw) }}</span>
                  <span v-if="cell.score !== null" class="faint">{{ Math.round(cell.score) }}</span>
                  <span v-if="cell.score !== null" class="capability-score-bar" :aria-label="t('results.relativeScore')">
                    <span :style="{ width: `${cell.score}%` }" />
                  </span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import type { CapabilityBlockReportDTO, CapabilityMetricReportDTO, ReportDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';
import { useModels } from '../../composables/useModels';
import { syntheticCapabilityLabel } from '../../lib/syntheticCapabilities';
import Icon from '../ui/Icon.vue';

type Scope = 'synthetic' | 'all';
type AxisGroup = {
  id: string;
  label: string;
  blocks: CapabilityBlockReportDTO[];
  sampleCount: number;
};
type ModelOption = {
  id: string;
  name: string;
  color: string;
};
type RadarSeries = ModelOption & {
  scores: number[];
  points: string;
  averageScore: number;
};

const props = defineProps<{ report: ReportDTO }>();
const { t } = useI18n();
const { formatNumber, formatInt } = useFormat();
const { modelName } = useModels();
const metricKey = ref('');
const scope = ref<Scope>('synthetic');
const selectedModelIds = ref<string[]>([]);
const SERIES_COLORS = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
  'var(--series-8)',
];
const KNOWN_ORDER = ['mase', 'mse', 'rmse', 'mae', 'smape', 'mape'];
const RADAR = { centerX: 320, centerY: 188, radius: 124 };
const radarRings = [25, 50, 75, 100];

const blocks = computed(() => props.report.capability_blocks || []);
const capabilityMetrics = computed(() => props.report.capability_metrics || []);
const syntheticBlocks = computed(() => blocks.value.filter((block) => block.block_type === 'synthetic'));
const hasProfile = computed(() => blocks.value.length > 0 && capabilityMetrics.value.length > 0 && metricOptions.value.length > 0);

const metricOptions = computed(() => {
  const keys = new Set<string>();
  capabilityMetrics.value.forEach((entry) => {
    Object.entries(entry.metrics || {}).forEach(([key, value]) => {
      const numeric = typeof value === 'number' ? value : Number(value);
      if (Number.isFinite(numeric)) keys.add(key);
    });
  });
  return [...keys].sort(orderMetricKeys);
});

const modelOptions = computed<ModelOption[]>(() => {
  const names = new Map<string, string>();
  props.report.model_metrics.forEach((entry) => {
    const id = String(entry.model_id || '');
    if (!id) return;
    const reportedName = typeof entry.model_name === 'string' ? entry.model_name : '';
    names.set(id, reportedName || modelName(id) || id);
  });
  capabilityMetrics.value.forEach((entry) => {
    const reportedName = typeof entry.model_name === 'string' ? entry.model_name : '';
    if (!names.has(entry.model_id)) names.set(entry.model_id, reportedName || modelName(entry.model_id) || entry.model_id);
  });
  return [...names.entries()].map(([id, name], index) => ({
    id,
    name,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  }));
});

const selectedModels = computed(() => modelOptions.value.filter((model) => selectedModelIds.value.includes(model.id)));

const visibleBlocks = computed(() => {
  if (scope.value === 'synthetic') return syntheticBlocks.value;
  return blocks.value;
});

const syntheticAxisGroups = computed<AxisGroup[]>(() => {
  const grouped = new Map<string, AxisGroup>();
  syntheticBlocks.value.forEach((block) => {
    const id = block.capability_type || block.capability_block_id;
    const current = grouped.get(id);
    if (current) {
      current.blocks.push(block);
      current.sampleCount += Number(block.sample_count || 0);
      return;
    }
    grouped.set(id, {
      id,
      label: capabilityBlockLabel(block),
      blocks: [block],
      sampleCount: Number(block.sample_count || 0),
    });
  });
  return [...grouped.values()];
});

const radarAxes = computed(() => syntheticAxisGroups.value.map((axis) => {
  const values = modelOptions.value.map((model) => weightedMetricForAxis(axis, model.id));
  const valid = values.filter((value): value is number => value !== null);
  const scores = new Map<string, number | null>();
  modelOptions.value.forEach((model, index) => {
    scores.set(model.id, scoreValue(values[index], valid));
  });
  return { ...axis, scores };
}));

const radarSeries = computed<RadarSeries[]>(() => {
  const out: RadarSeries[] = [];
  selectedModels.value.forEach((model) => {
    const scores = radarAxes.value.map((axis) => axis.scores.get(model.id) ?? null);
    if (!scores.every((score): score is number => score !== null)) return;
    const points = scores.map((score, index) => {
      const point = axisPoint(index, score);
      return `${point.x},${point.y}`;
    }).join(' ');
    out.push({
      ...model,
      scores,
      points,
      averageScore: scores.reduce((sum, value) => sum + value, 0) / scores.length,
    });
  });
  return out;
});

const radarPoints = computed(() => radarSeries.value.flatMap((series) =>
  series.scores.map((score, index) => {
    const point = axisPoint(index, score);
    return {
      modelId: series.id,
      axisId: radarAxes.value[index].id,
      x: point.x,
      y: point.y,
      color: series.color,
    };
  })
));

const showRadar = computed(() => radarAxes.value.length >= 3 && radarAxes.value.length <= 8 && radarSeries.value.length > 0);
const radarUnavailableText = computed(() => {
  if (radarAxes.value.length < 3) return t('results.radarUnavailableFew');
  if (radarAxes.value.length > 8) return t('results.radarUnavailableMany');
  return t('results.radarUnavailableMissing');
});

const blockRows = computed(() => visibleBlocks.value.map((block) => {
  const allValues = modelOptions.value.map((model) => rawMetricForBlock(block, model.id));
  const valid = allValues.filter((value): value is number => value !== null);
  return {
    block,
    label: capabilityBlockLabel(block),
    cells: selectedModels.value.map((model) => {
      const raw = rawMetricForBlock(block, model.id);
      return {
        modelId: model.id,
        raw,
        score: scoreValue(raw, valid),
      };
    }),
  };
}));

watch(metricOptions, (options) => {
  if (!options.length) {
    metricKey.value = '';
    return;
  }
  if (!options.includes(metricKey.value)) {
    metricKey.value = options[0];
  }
}, { immediate: true });

watch(modelOptions, (models) => {
  const available = new Set(models.map((model) => model.id));
  const kept = selectedModelIds.value.filter((id) => available.has(id));
  selectedModelIds.value = kept.length ? kept : models.slice(0, 5).map((model) => model.id);
}, { immediate: true });

watch(syntheticBlocks, (items) => {
  if (!items.length) scope.value = 'all';
}, { immediate: true });

function toggleModel(modelId: string, checked: boolean) {
  selectedModelIds.value = checked
    ? [...new Set([...selectedModelIds.value, modelId])]
    : selectedModelIds.value.filter((id) => id !== modelId);
}

function onModelToggle(modelId: string, event: Event) {
  toggleModel(modelId, (event.target as HTMLInputElement).checked);
}

function rawMetricForBlock(block: CapabilityBlockReportDTO, modelId: string): number | null {
  const entries = capabilityMetrics.value.filter((entry) => entry.capability_block_id === block.capability_block_id && entry.model_id === modelId);
  return weightedMean(entries, (entry) => Number(entry.sample_count || block.sample_count || 1));
}

function weightedMetricForAxis(axis: AxisGroup, modelId: string): number | null {
  const values = axis.blocks
    .map((block) => {
      const raw = rawMetricForBlock(block, modelId);
      if (raw === null) return null;
      return { value: raw, weight: Math.max(1, Number(block.sample_count || 1)) };
    })
    .filter((item): item is { value: number; weight: number } => item !== null);
  if (!values.length) return null;
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
  return values.reduce((sum, item) => sum + item.value * item.weight, 0) / totalWeight;
}

function weightedMean(entries: CapabilityMetricReportDTO[], weightOf: (entry: CapabilityMetricReportDTO) => number): number | null {
  let weightedSum = 0;
  let totalWeight = 0;
  for (const entry of entries) {
    const raw = metricValue(entry);
    if (raw === null) continue;
    const weight = Math.max(1, weightOf(entry));
    weightedSum += raw * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : null;
}

function metricValue(entry: CapabilityMetricReportDTO): number | null {
  const value = (entry.metrics || {})[metricKey.value];
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function scoreValue(value: number | null, validValues: number[]): number | null {
  if (value === null || !validValues.length) return null;
  const safeValue = Math.max(0, value);
  if (metricKey.value === 'mase') {
    return clamp(100 / (1 + safeValue), 0, 100);
  }
  const best = Math.min(...validValues.map((item) => Math.max(0, item)));
  if (safeValue === 0) return 100;
  if (best === 0) return 0;
  return clamp((best / safeValue) * 100, 0, 100);
}

function axisPoint(index: number, score: number) {
  const count = Math.max(1, radarAxes.value.length);
  const angle = -Math.PI / 2 + (index * Math.PI * 2) / count;
  const radius = RADAR.radius * (score / 100);
  return {
    x: Number((RADAR.centerX + Math.cos(angle) * radius).toFixed(2)),
    y: Number((RADAR.centerY + Math.sin(angle) * radius).toFixed(2)),
  };
}

function ringPoints(score: number) {
  return radarAxes.value.map((_axis, index) => {
    const point = axisPoint(index, score);
    return `${point.x},${point.y}`;
  }).join(' ');
}

function labelAnchor(x: number) {
  if (x < RADAR.centerX - 12) return 'end';
  if (x > RADAR.centerX + 12) return 'start';
  return 'middle';
}

function shortAxisLabel(label: string) {
  return label.length > 18 ? `${label.slice(0, 17)}...` : label;
}

function blockTypeLabel(block: CapabilityBlockReportDTO) {
  return block.block_type === 'synthetic' ? t('results.syntheticData') : t('results.realData');
}

function capabilityBlockLabel(block: CapabilityBlockReportDTO) {
  if (block.block_type === 'synthetic') {
    return syntheticCapabilityLabel(block.capability_type, block.capability_label || block.name, t);
  }
  return block.name || block.capability_label || block.capability_type || block.capability_block_id;
}

function formatMaybe(value: number | null) {
  return value === null ? t('common.notAvailable') : formatNumber(value);
}

function orderMetricKeys(a: string, b: string) {
  const ia = KNOWN_ORDER.indexOf(a);
  const ib = KNOWN_ORDER.indexOf(b);
  if (ia === -1 && ib === -1) return a.localeCompare(b);
  if (ia === -1) return 1;
  if (ib === -1) return -1;
  return ia - ib;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
</script>
