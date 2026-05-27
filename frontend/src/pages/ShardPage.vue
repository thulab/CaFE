<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('shard.eyebrow') }}</p>
        <h1>{{ t('shard.title') }}</h1>
        <p class="page-sub">{{ t('shard.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="shard?.dataset_manifest_id ? `#/datasets/${shard.dataset_manifest_id}` : '#/datasets'">
          <Icon name="database" :size="15" /> {{ t('shard.openDataset') }}
        </a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="shard" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">{{ t('shard.status') }}</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="shard.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.samples') }}</span><span class="stat-value">{{ formatInt(shard.sample_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.rows') }}</span><span class="stat-value">{{ formatInt(shard.row_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.window') }}</span><span class="stat-value" style="font-size:1.1rem">{{ shard.context_length }}→{{ shard.horizon }}</span><span class="stat-foot">{{ t('shard.windowFoot') }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('shard.configuration') }}</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>{{ t('shard.shardId') }}</dt><dd class="mono">{{ shard.shard_id }}</dd></div>
              <div class="detail-item"><dt>{{ t('shard.manifest') }}</dt><dd><a class="text-link" :href="`#/datasets/${shard.dataset_manifest_id}`">{{ shortId(shard.dataset_manifest_id) }}</a></dd></div>
              <div class="detail-item"><dt>{{ t('shard.loadJob') }}</dt><dd><a v-if="shard.load_job_id" class="text-link" :href="`#/load-jobs/${shard.load_job_id}`">{{ shortId(shard.load_job_id) }}</a><span v-else class="faint">{{ t('shard.notRecorded') }}</span></dd></div>
              <div class="detail-item"><dt>{{ t('shard.split') }}</dt><dd>{{ t('shard.splitSummary', { context: shard.context_length, horizon: shard.horizon, stride: shard.stride }) }}</dd></div>
              <div class="detail-item"><dt>{{ t('shard.targets') }}</dt><dd>{{ shard.target_columns.join(', ') }}</dd></div>
              <div class="detail-item wide"><dt>{{ t('shard.sourceUri') }}</dt><dd class="mono">{{ shard.source_uri }}</dd></div>
            </dl>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('shard.shardSamples') }}</h2><span v-if="samples" class="badge">{{ formatInt(samples.total ?? samples.items.length) }}</span></header>
          <div class="card-body">
            <div v-if="samples" class="table-wrap">
              <table class="data">
                <thead><tr><th>{{ t('shard.sample') }}</th><th class="num">{{ t('shard.index') }}</th><th>{{ t('shard.context') }}</th><th>{{ t('shard.horizon') }}</th></tr></thead>
                <tbody>
                  <tr v-for="s in samples.items" :key="s.sample_id">
                    <td class="mono">{{ s.sample_id }}</td>
                    <td class="num">{{ s.sample_index ?? t('common.notAvailable') }}</td>
                    <td>{{ range(s.context_start, s.context_end) }}</td>
                    <td>{{ range(s.horizon_start, s.horizon_end) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p v-else class="empty-state">{{ t('shard.noSampleIndex') }}</p>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { getShard, getShardSamples } from '../api/datasets';
import type { ShardDTO, ShardSamplesDTO } from '../api/types';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';

const props = defineProps<{ shardId: string }>();
const shard = ref<ShardDTO | null>(null);
const samples = ref<ShardSamplesDTO | null>(null);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { t } = useI18n();
const { formatInt } = useFormat();

onMounted(run);

async function run() {
  loading.value = true;
  clearError();
  try {
    const [s, sx] = await Promise.all([getShard(props.shardId), getShardSamples(props.shardId)]);
    shard.value = s;
    samples.value = sx;
  } catch (e) {
    setError(e, 'shard.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

function range(start?: number, end?: number) {
  if (start === undefined || end === undefined) return t('common.notAvailable');
  return `${start} → ${end}`;
}
</script>
