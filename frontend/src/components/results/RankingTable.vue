<template>
  <div class="table-wrap">
    <table class="data">
      <caption>{{ t('results.lowerIsBetter') }}</caption>
      <thead>
        <tr>
          <th style="width:64px">{{ t('results.rank') }}</th>
          <th>{{ t('results.model') }}</th>
          <th class="num" style="width:140px">{{ resolvedMetricLabel }}</th>
          <th style="width:34%">{{ t('results.relative') }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in sorted" :key="row.model_id" :class="{ 'is-winner': row.rank === 1 }">
          <td>
            <span class="rank-badge" :class="medal(row.rank)">{{ row.rank }}</span>
          </td>
          <td>
            <span style="font-weight:600">{{ modelName(row.model_id) }}</span>
            <Icon v-if="row.rank === 1" name="trophy" :size="14" style="margin-left:6px;color:var(--success)" />
            <div v-if="modelName(row.model_id) !== row.model_id" class="faint mono" style="font-size:0.74rem">{{ shortId(row.model_id) }}</div>
          </td>
          <td class="num">{{ formatNumber(row.metric_value) }}</td>
          <td>
            <div class="bar-cell">
              <div class="bar-track">
                <div class="bar-fill" :class="{ 'is-winner': row.rank === 1 }" :style="{ width: barWidth(row.metric_value) + '%' }" />
              </div>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { useModels } from '../../composables/useModels';
import { useFormat } from '../../composables/useFormat';
import { shortId } from '../../lib/format';

const props = defineProps<{
  items: Array<{ model_id: string; rank: number; metric_value: number }>;
  metricLabel?: string;
}>();

const { modelName } = useModels();
const { t } = useI18n();
const { formatNumber } = useFormat();
const resolvedMetricLabel = computed(() => props.metricLabel ?? t('results.metric'));

const sorted = computed(() => [...props.items].sort((a, b) => a.rank - b.rank));
const maxValue = computed(() => Math.max(...props.items.map((i) => Math.abs(i.metric_value)), 1e-9));

function barWidth(v: number): number {
  return Math.max(4, Math.min(100, (Math.abs(v) / maxValue.value) * 100));
}

function medal(rank: number): string {
  return rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
}
</script>
