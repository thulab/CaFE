<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Shard detail</h1>
        <p class="page-subtitle">Review the materialized evaluation shard, split parameters, and sample index.</p>
      </div>
      <a class="button-link secondary" :href="shard?.dataset_manifest_id ? `#/datasets/${shard.dataset_manifest_id}` : '#/'">Open dataset</a>
    </header>

    <p v-if="!shard" class="status-line">Loading shard...</p>
    <section v-else class="report-sections">
      <div class="metric-strip">
        <span class="stat-pill">{{ shard.status }}</span>
        <span class="stat-pill">{{ shard.sample_count }} samples</span>
        <span class="stat-pill">{{ shard.row_count }} rows</span>
      </div>

      <article class="page-card">
        <h2 class="section-title">Shard configuration</h2>
        <dl class="detail-grid">
          <div class="detail-item">
            <dt>Shard ID</dt>
            <dd class="mono-text">{{ shard.shard_id }}</dd>
          </div>
          <div class="detail-item">
            <dt>Manifest</dt>
            <dd><a class="text-link" :href="`#/datasets/${shard.dataset_manifest_id}`">{{ shard.dataset_manifest_id }}</a></dd>
          </div>
          <div class="detail-item">
            <dt>Load job</dt>
            <dd>
              <a v-if="shard.load_job_id" class="text-link" :href="`#/load-jobs/${shard.load_job_id}`">{{ shard.load_job_id }}</a>
              <span v-else>Not recorded</span>
            </dd>
          </div>
          <div class="detail-item">
            <dt>Split</dt>
            <dd>{{ shard.context_length }} context / {{ shard.horizon }} horizon / {{ shard.stride }} stride</dd>
          </div>
          <div class="detail-item">
            <dt>Targets</dt>
            <dd>{{ shard.target_columns.join(', ') }}</dd>
          </div>
          <div class="detail-item wide">
            <dt>Source URI</dt>
            <dd class="mono-text">{{ shard.source_uri }}</dd>
          </div>
        </dl>
      </article>

      <article class="page-card">
        <h2 class="section-title">Shard samples</h2>
        <p v-if="!samples" class="status-line">Loading sample index...</p>
        <div v-else class="table-scroll">
          <table class="data-table">
            <caption>{{ samples.total ?? samples.items.length }} indexed samples</caption>
            <thead>
              <tr>
                <th>Sample</th>
                <th>Index</th>
                <th>Context</th>
                <th>Horizon</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="sample in samples.items" :key="sample.sample_id">
                <td class="mono-text">{{ sample.sample_id }}</td>
                <td>{{ sample.sample_index ?? '-' }}</td>
                <td>{{ rangeText(sample.context_start, sample.context_end) }}</td>
                <td>{{ rangeText(sample.horizon_start, sample.horizon_end) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getShard, getShardSamples } from '../api/datasets';
import type { ShardDTO, ShardSamplesDTO } from '../api/types';

const props = defineProps<{ shardId: string }>();
const shard = ref<ShardDTO | null>(null);
const samples = ref<ShardSamplesDTO | null>(null);

onMounted(async () => {
  const [loadedShard, loadedSamples] = await Promise.all([
    getShard(props.shardId),
    getShardSamples(props.shardId)
  ]);
  shard.value = loadedShard;
  samples.value = loadedSamples;
});

function rangeText(start?: number, end?: number) {
  if (start === undefined || end === undefined) return '-';
  return `${start} to ${end}`;
}
</script>
