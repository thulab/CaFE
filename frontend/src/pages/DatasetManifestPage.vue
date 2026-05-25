<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Dataset manifest</h1>
        <p class="page-subtitle">Review the uploaded data source, selected time column, target columns, and load readiness.</p>
      </div>
      <a class="button-link secondary" href="#/">Create workflow</a>
    </header>

    <p v-if="!manifest" class="status-line">Loading dataset manifest...</p>
    <section v-else class="report-sections">
      <div class="metric-strip">
        <span class="stat-pill">{{ manifest.status }}</span>
        <span class="stat-pill">{{ manifest.file_format }}</span>
        <span class="stat-pill">{{ manifest.value_columns.length }} value columns</span>
      </div>

      <article class="page-card">
        <h2 class="section-title">{{ manifest.name }}</h2>
        <dl class="detail-grid">
          <div class="detail-item">
            <dt>Manifest ID</dt>
            <dd class="mono-text">{{ manifest.dataset_manifest_id }}</dd>
          </div>
          <div class="detail-item">
            <dt>Domain</dt>
            <dd>{{ manifest.domain }}</dd>
          </div>
          <div class="detail-item wide">
            <dt>Source URI</dt>
            <dd class="mono-text">{{ manifest.source_uri }}</dd>
          </div>
          <div class="detail-item">
            <dt>Time column</dt>
            <dd>{{ manifest.time_column }}</dd>
          </div>
          <div class="detail-item">
            <dt>Value columns</dt>
            <dd>{{ manifest.value_columns.join(', ') }}</dd>
          </div>
          <div class="detail-item">
            <dt>Frequency</dt>
            <dd>{{ manifest.frequency || 'Not inferred yet' }}</dd>
          </div>
          <div class="detail-item">
            <dt>Timezone</dt>
            <dd>{{ manifest.timezone || 'Not set' }}</dd>
          </div>
        </dl>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getDatasetManifest } from '../api/datasets';
import type { DatasetManifestDTO } from '../api/types';

const props = defineProps<{ datasetManifestId: string }>();
const manifest = ref<DatasetManifestDTO | null>(null);

onMounted(async () => {
  manifest.value = await getDatasetManifest(props.datasetManifestId);
});
</script>
