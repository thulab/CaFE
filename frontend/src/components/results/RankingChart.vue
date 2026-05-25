<template>
  <div class="rank-chart">
    <div v-for="row in sorted" :key="row.model_id" class="rank-row" :class="{ winner: row.rank === 1 }">
      <div class="rank-row-label" :title="modelName(row.model_id)">
        <span class="rank-badge" :class="medal(row.rank)">{{ row.rank }}</span>
        <span class="nowrap" style="overflow:hidden;text-overflow:ellipsis">{{ modelName(row.model_id) }}</span>
      </div>
      <div class="bar-track" style="height:22px">
        <div class="bar-fill" :class="{ 'is-winner': row.rank === 1 }" :style="{ width: barWidth(row.metric_value) + '%' }" />
        <span class="bar-value mono">{{ formatNumber(row.metric_value) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useModels } from '../../composables/useModels';
import { formatNumber } from '../../lib/format';

const props = defineProps<{ items: Array<{ model_id: string; rank: number; metric_value: number }> }>();
const { modelName } = useModels();

const sorted = computed(() => [...props.items].sort((a, b) => a.rank - b.rank));
const maxValue = computed(() => Math.max(...props.items.map((i) => Math.abs(i.metric_value)), 1e-9));

function barWidth(v: number): number {
  return Math.max(4, Math.min(100, (Math.abs(v) / maxValue.value) * 100));
}

function medal(rank: number): string {
  return rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
}
</script>

<style scoped>
.rank-chart { display: grid; gap: 10px; }
.rank-row {
  display: grid;
  grid-template-columns: minmax(120px, 200px) minmax(0, 1fr);
  gap: 14px;
  align-items: center;
}
.rank-row-label {
  display: flex;
  align-items: center;
  gap: 9px;
  font-weight: 600;
  font-size: 0.9rem;
  min-width: 0;
}
.bar-track { position: relative; background: var(--surface-3); border-radius: var(--radius-sm); overflow: hidden; }
.bar-fill { transition: width var(--dur) var(--ease); }
.bar-value {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text);
}
</style>
