<template>
  <article class="card" style="padding:18px;display:flex;flex-direction:column;gap:12px">
    <header style="display:flex;flex-direction:column;gap:4px">
      <h3 style="margin:0;font-size:1.05rem;font-weight:700">{{ item.track_name }}</h3>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
        <span class="badge neutral">{{ item.track_type }} · {{ item.primary_metric_id.toUpperCase() }}</span>
        <span class="faint" style="font-size:0.78rem">updated {{ timeAgo(item.updated_at) }}</span>
      </div>
    </header>

    <div v-if="item.top.length === 0" class="faint" style="padding:14px 0;text-align:center;font-size:0.88rem">
      No ranked results yet
    </div>
    <ul v-else style="list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px">
      <li v-for="row in sorted" :key="row.model_id" style="display:grid;grid-template-columns:28px minmax(0,1fr) auto;gap:10px;align-items:center">
        <span class="rank-badge" :class="medal(row.rank)">{{ row.rank }}</span>
        <span style="min-width:0;display:flex;flex-direction:column;gap:2px">
          <span style="font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ row.model_name ?? modelName(row.model_id) }}</span>
          <span class="bar-cell">
            <span class="bar-track">
              <span class="bar-fill" :class="{ 'is-winner': row.rank === 1 }" :style="{ width: barWidth(row.metric_value) + '%' }" />
            </span>
          </span>
        </span>
        <span class="mono" style="font-size:0.86rem">{{ formatNumber(row.metric_value) }}</span>
      </li>
    </ul>

    <div class="faint" style="font-size:0.78rem">
      {{ item.model_count }} model{{ item.model_count === 1 ? '' : 's' }} · {{ item.run_count }} run{{ item.run_count === 1 ? '' : 's' }}
    </div>

    <a class="btn secondary sm" :href="`#/tracks/${item.track_id}/ranking`" style="align-self:flex-start">
      View full board <Icon name="arrowRight" :size="14" />
    </a>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../ui/Icon.vue';
import { useModels } from '../../composables/useModels';
import { formatNumber, timeAgo } from '../../lib/format';
import type { LeaderboardItem } from '../../api/results';

const props = defineProps<{ item: LeaderboardItem }>();

const { modelName } = useModels();

const sorted = computed(() => [...props.item.top].sort((a, b) => a.rank - b.rank));
const maxValue = computed(() => Math.max(...props.item.top.map((t) => Math.abs(t.metric_value)), 1e-9));

function barWidth(v: number): number {
  return Math.max(4, Math.min(100, (Math.abs(v) / maxValue.value) * 100));
}

function medal(rank: number): string {
  return rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
}
</script>
