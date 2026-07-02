<template>
  <article v-if="modelOptions.length >= 2" class="card">
    <header class="card-head">
      <div>
        <h2 class="card-title"><Icon name="barChart" :size="18" style="vertical-align:-3px;margin-right:6px" />{{ t('modelComparison.title') }}</h2>
        <p class="faint" style="margin:4px 0 0;font-size:0.82rem">{{ t('modelComparison.metricHint', { metric: metricKey.toUpperCase() }) }}</p>
      </div>
      <span class="badge">{{ t('results.lowerIsBetter') }}</span>
    </header>
    <div class="card-body stack">
      <div class="toolbar">
        <label class="field">
          <span class="label">{{ t('modelComparison.modelA') }}</span>
          <select v-model="modelAId" :aria-label="t('modelComparison.modelA')">
            <option v-for="model in modelOptions" :key="model.id" :value="model.id" :disabled="model.id === modelBId">{{ model.name }}</option>
          </select>
        </label>
        <label class="field">
          <span class="label">{{ t('modelComparison.modelB') }}</span>
          <select v-model="modelBId" :aria-label="t('modelComparison.modelB')">
            <option v-for="model in modelOptions" :key="model.id" :value="model.id" :disabled="model.id === modelAId">{{ model.name }}</option>
          </select>
        </label>
      </div>

      <p v-if="!rows.length" class="empty-state">{{ t('modelComparison.noRows') }}</p>
      <div v-else class="table-wrap">
        <table class="data">
          <caption>{{ t('modelComparison.caption') }}</caption>
          <thead>
            <tr>
              <th>{{ t('results.testGroup') }}</th>
              <th class="num">{{ selectedModelA?.name || t('modelComparison.modelA') }}</th>
              <th class="num">{{ selectedModelB?.name || t('modelComparison.modelB') }}</th>
              <th>{{ t('modelComparison.winner') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.id">
              <td>
                <span style="font-weight:650">{{ row.label }}</span>
                <div v-if="row.subtitle" class="faint" style="font-size:0.74rem">{{ row.subtitle }}</div>
              </td>
              <td class="num" :class="{ 'is-winner': row.winner === modelAId }">{{ formatMaybe(row.a) }}</td>
              <td class="num" :class="{ 'is-winner': row.winner === modelBId }">{{ formatMaybe(row.b) }}</td>
              <td>
                <span v-if="row.winner" class="badge success">{{ winnerName(row.winner) }}</span>
                <span v-else class="muted">{{ t('common.notAvailable') }}</span>
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
import type { CapabilityBlockReportDTO, CapabilityMetricReportDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';
import { useModels } from '../../composables/useModels';
import { syntheticCapabilityLabel } from '../../lib/syntheticCapabilities';
import Icon from '../ui/Icon.vue';

type ComparisonSource = {
  metric?: string | null;
  model_metrics: Array<Record<string, unknown>>;
  capability_blocks?: CapabilityBlockReportDTO[];
  capability_metrics?: CapabilityMetricReportDTO[];
};
type ModelOption = {
  id: string;
  name: string;
};
type ComparisonRow = {
  id: string;
  label: string;
  subtitle: string;
  a: number | null;
  b: number | null;
  winner: string;
};

const props = defineProps<{ source: ComparisonSource; metric?: string }>();
const { t } = useI18n();
const { modelName } = useModels();
const { formatNumber } = useFormat();
const modelAId = ref('');
const modelBId = ref('');

const metricKey = computed(() => props.metric || props.source.metric || firstMetricKey.value || 'mse');
const modelOptions = computed<ModelOption[]>(() => {
  const names = new Map<string, string>();
  props.source.model_metrics.forEach((entry) => {
    const id = String(entry.model_id || '');
    if (!id) return;
    names.set(id, String(entry.model_name || modelName(id) || id));
  });
  (props.source.capability_metrics || []).forEach((entry) => {
    if (names.has(entry.model_id)) return;
    names.set(entry.model_id, entry.model_name || modelName(entry.model_id) || entry.model_id);
  });
  return [...names.entries()].map(([id, name]) => ({ id, name }));
});
const selectedModelA = computed(() => modelOptions.value.find((model) => model.id === modelAId.value));
const selectedModelB = computed(() => modelOptions.value.find((model) => model.id === modelBId.value));
const firstMetricKey = computed(() => {
  for (const entry of props.source.model_metrics) {
    const metrics = (entry.metrics || {}) as Record<string, unknown>;
    const key = Object.keys(metrics).find((item) => Number.isFinite(typeof metrics[item] === 'number' ? metrics[item] : Number(metrics[item])));
    if (key) return key;
  }
  for (const entry of props.source.capability_metrics || []) {
    const key = Object.keys(entry.metrics || {}).find((item) => Number.isFinite(Number((entry.metrics || {})[item])));
    if (key) return key;
  }
  return '';
});

const rows = computed<ComparisonRow[]>(() => {
  if (!modelAId.value || !modelBId.value || modelAId.value === modelBId.value) return [];
  const blocks = props.source.capability_blocks || [];
  const blockRows = blocks
    .map((block) => {
      const a = metricForBlock(block.capability_block_id, modelAId.value);
      const b = metricForBlock(block.capability_block_id, modelBId.value);
      return makeRow(block.capability_block_id, capabilityBlockLabel(block), block.capability_type || block.block_type || '', a, b);
    })
    .filter((row) => row.a !== null || row.b !== null);
  if (blockRows.length) return blockRows;
  const a = metricForModel(modelAId.value);
  const b = metricForModel(modelBId.value);
  const overall = makeRow('overall', t('modelComparison.overall'), '', a, b);
  return overall.a !== null || overall.b !== null ? [overall] : [];
});

watch(modelOptions, (models) => {
  if (models.length < 2) {
    modelAId.value = '';
    modelBId.value = '';
    return;
  }
  if (!models.some((model) => model.id === modelAId.value)) modelAId.value = models[0].id;
  if (!models.some((model) => model.id === modelBId.value) || modelBId.value === modelAId.value) {
    modelBId.value = models.find((model) => model.id !== modelAId.value)?.id || '';
  }
}, { immediate: true });

watch(modelAId, (value) => {
  if (value && value === modelBId.value) {
    modelBId.value = modelOptions.value.find((model) => model.id !== value)?.id || '';
  }
});

watch(modelBId, (value) => {
  if (value && value === modelAId.value) {
    modelAId.value = modelOptions.value.find((model) => model.id !== value)?.id || '';
  }
});

function metricForBlock(blockId: string, modelId: string) {
  const entries = (props.source.capability_metrics || []).filter((entry) => entry.capability_block_id === blockId && entry.model_id === modelId);
  if (!entries.length) return null;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const entry of entries) {
    const value = metricValue(entry.metrics);
    if (value === null) continue;
    const weight = Math.max(1, Number(entry.sample_count || 1));
    weightedSum += value * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : null;
}

function metricForModel(modelId: string) {
  const entry = props.source.model_metrics.find((item) => String(item.model_id || '') === modelId);
  return entry ? metricValue((entry.metrics || {}) as Record<string, unknown>) : null;
}

function metricValue(metrics: Record<string, unknown>) {
  const value = metrics[metricKey.value];
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function makeRow(id: string, label: string, subtitle: string, a: number | null, b: number | null): ComparisonRow {
  let winner = '';
  if (a !== null && b !== null && a !== b) winner = a < b ? modelAId.value : modelBId.value;
  return { id, label, subtitle, a, b, winner };
}

function winnerName(modelId: string) {
  return modelOptions.value.find((model) => model.id === modelId)?.name || modelId;
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
</script>

<style scoped>
td.is-winner {
  color: var(--success-text);
  font-weight: 750;
}
</style>
