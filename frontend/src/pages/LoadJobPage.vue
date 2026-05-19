<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Dataset load job</h1>
        <p class="page-subtitle">Inspect CSV validation, split settings, and the shard produced by a dataset load.</p>
      </div>
      <a class="button-link secondary" :href="job?.output_shard_id ? `#/shards/${job.output_shard_id}` : '#/'">Open shard</a>
    </header>

    <p v-if="!job" class="status-line">Loading dataset load job...</p>
    <section v-else class="report-sections">
      <div class="metric-strip">
        <span class="stat-pill">{{ job.status }}</span>
        <span class="stat-pill">Seed {{ job.seed ?? 0 }}</span>
        <span v-if="job.current_step" class="stat-pill">{{ job.current_step }}</span>
      </div>

      <article class="page-card">
        <h2 class="section-title">Load configuration</h2>
        <dl class="detail-grid">
          <div class="detail-item">
            <dt>Load job ID</dt>
            <dd class="mono-text">{{ job.load_job_id }}</dd>
          </div>
          <div class="detail-item">
            <dt>Manifest</dt>
            <dd><a class="text-link" :href="`#/datasets/${job.dataset_manifest_id}`">{{ job.dataset_manifest_id }}</a></dd>
          </div>
          <div class="detail-item">
            <dt>Output shard</dt>
            <dd>
              <a v-if="job.output_shard_id" class="text-link" :href="`#/shards/${job.output_shard_id}`">{{ job.output_shard_id }}</a>
              <span v-else>Pending</span>
            </dd>
          </div>
          <div class="detail-item wide">
            <dt>Split config</dt>
            <dd class="mono-text">{{ formatJson(job.split_config) }}</dd>
          </div>
          <div v-if="job.error_message" class="detail-item wide">
            <dt>Error</dt>
            <dd>{{ job.error_code }} {{ job.error_message }}</dd>
          </div>
        </dl>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getLoadJob } from '../api/datasets';
import type { DatasetLoadJobDetailDTO } from '../api/types';

const props = defineProps<{ loadJobId: string }>();
const job = ref<DatasetLoadJobDetailDTO | null>(null);

onMounted(async () => {
  job.value = await getLoadJob(props.loadJobId);
});

function formatJson(value: Record<string, unknown>) {
  return JSON.stringify(value);
}
</script>
