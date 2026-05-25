<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Dataset load job</h1>
        <p class="page-sub">CSV validation, split settings, and the shard produced by a dataset load.</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="job?.output_shard_id ? `#/shards/${job.output_shard_id}` : '#/datasets'">
          <Icon name="layers" :size="15" /> Open shard
        </a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="job" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">Status</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="job.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">Seed</span><span class="stat-value">{{ job.seed ?? 0 }}</span></div>
          <div v-if="job.current_step" class="stat-tile"><span class="stat-label">Current step</span><span class="stat-value" style="font-size:1.1rem">{{ humanize(job.current_step) }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Load configuration</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>Load job ID</dt><dd class="mono">{{ job.load_job_id }}</dd></div>
              <div class="detail-item"><dt>Manifest</dt><dd><a class="text-link" :href="`#/datasets/${job.dataset_manifest_id}`">{{ shortId(job.dataset_manifest_id) }}</a></dd></div>
              <div class="detail-item"><dt>Output shard</dt><dd><a v-if="job.output_shard_id" class="text-link" :href="`#/shards/${job.output_shard_id}`">{{ shortId(job.output_shard_id) }}</a><span v-else class="faint">Pending</span></dd></div>
              <div class="detail-item"><dt>Started</dt><dd>{{ formatDateTime(job.started_at) }}</dd></div>
              <div class="detail-item"><dt>Finished</dt><dd>{{ formatDateTime(job.finished_at) }}</dd></div>
              <div class="detail-item wide"><dt>Split config</dt><dd><pre class="mono" style="margin:0;white-space:pre-wrap">{{ pretty(job.split_config) }}</pre></dd></div>
              <div v-if="job.error_message" class="detail-item wide"><dt>Error</dt><dd class="alert" style="margin:0"><Icon class="alert-ico" name="alert" :size="16" /><span>{{ job.error_code }} · {{ job.error_message }}</span></dd></div>
            </dl>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { getLoadJob } from '../api/datasets';
import type { DatasetLoadJobDetailDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { formatDateTime, humanize, shortId } from '../lib/format';

const props = defineProps<{ loadJobId: string }>();
const { data: job, loading, error, run } = useAsyncData<DatasetLoadJobDetailDTO>(() => getLoadJob(props.loadJobId));

onMounted(run);

function pretty(value: Record<string, unknown>) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
</script>
