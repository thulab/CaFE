<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Shard detail</h1>
        <p class="page-sub">The materialized evaluation shard, split parameters, and sample index.</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="shard?.dataset_manifest_id ? `#/datasets/${shard.dataset_manifest_id}` : '#/datasets'">
          <Icon name="database" :size="15" /> Open dataset
        </a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="shard" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">Status</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="shard.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">Samples</span><span class="stat-value">{{ formatInt(shard.sample_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">Rows</span><span class="stat-value">{{ formatInt(shard.row_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">Window</span><span class="stat-value" style="font-size:1.1rem">{{ shard.context_length }}→{{ shard.horizon }}</span><span class="stat-foot">context → horizon</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Shard configuration</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>Shard ID</dt><dd class="mono">{{ shard.shard_id }}</dd></div>
              <div class="detail-item"><dt>Manifest</dt><dd><a class="text-link" :href="`#/datasets/${shard.dataset_manifest_id}`">{{ shortId(shard.dataset_manifest_id) }}</a></dd></div>
              <div class="detail-item"><dt>Load job</dt><dd><a v-if="shard.load_job_id" class="text-link" :href="`#/load-jobs/${shard.load_job_id}`">{{ shortId(shard.load_job_id) }}</a><span v-else class="faint">Not recorded</span></dd></div>
              <div class="detail-item"><dt>Split</dt><dd>{{ shard.context_length }} context · {{ shard.horizon }} horizon · {{ shard.stride }} stride</dd></div>
              <div class="detail-item"><dt>Targets</dt><dd>{{ shard.target_columns.join(', ') }}</dd></div>
              <div class="detail-item wide"><dt>Source URI</dt><dd class="mono">{{ shard.source_uri }}</dd></div>
            </dl>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Shard samples</h2><span v-if="samples" class="badge">{{ samples.total ?? samples.items.length }}</span></header>
          <div class="card-body">
            <div v-if="samples" class="table-wrap">
              <table class="data">
                <thead><tr><th>Sample</th><th class="num">Index</th><th>Context</th><th>Horizon</th></tr></thead>
                <tbody>
                  <tr v-for="s in samples.items" :key="s.sample_id">
                    <td class="mono">{{ s.sample_id }}</td>
                    <td class="num">{{ s.sample_index ?? '—' }}</td>
                    <td>{{ range(s.context_start, s.context_end) }}</td>
                    <td>{{ range(s.horizon_start, s.horizon_end) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p v-else class="empty-state">No sample index loaded.</p>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { getShard, getShardSamples } from '../api/datasets';
import type { ShardDTO, ShardSamplesDTO } from '../api/types';
import { formatInt, shortId } from '../lib/format';

const props = defineProps<{ shardId: string }>();
const shard = ref<ShardDTO | null>(null);
const samples = ref<ShardSamplesDTO | null>(null);
const loading = ref(true);
const error = ref<string | null>(null);

onMounted(run);

async function run() {
  loading.value = true;
  error.value = null;
  try {
    const [s, sx] = await Promise.all([getShard(props.shardId), getShardSamples(props.shardId)]);
    shard.value = s;
    samples.value = sx;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load shard';
  } finally {
    loading.value = false;
  }
}

function range(start?: number, end?: number) {
  if (start === undefined || end === undefined) return '—';
  return `${start} → ${end}`;
}
</script>
