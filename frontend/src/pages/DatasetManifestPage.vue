<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Dataset</p>
        <h1>Dataset manifest</h1>
        <p class="page-sub">The uploaded source, selected time and target columns, and load readiness.</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" href="#/datasets"><Icon name="list" :size="15" /> All datasets</a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="manifest" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">Status</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="manifest.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">Format</span><span class="stat-value" style="font-size:1.2rem">{{ manifest.file_format.toUpperCase() }}</span></div>
          <div class="stat-tile"><span class="stat-label">Value columns</span><span class="stat-value">{{ manifest.value_columns.length }}</span></div>
          <div class="stat-tile"><span class="stat-label">Domain</span><span class="stat-value" style="font-size:1.2rem">{{ manifest.domain }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ manifest.name }}</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>Manifest ID</dt><dd class="mono">{{ manifest.dataset_manifest_id }}</dd></div>
              <div class="detail-item"><dt>Time column</dt><dd>{{ manifest.time_column }}</dd></div>
              <div class="detail-item"><dt>Value columns</dt><dd>{{ manifest.value_columns.join(', ') }}</dd></div>
              <div class="detail-item"><dt>Frequency</dt><dd>{{ manifest.frequency || '—' }}</dd></div>
              <div class="detail-item"><dt>Timezone</dt><dd>{{ manifest.timezone || '—' }}</dd></div>
              <div class="detail-item"><dt>Created</dt><dd>{{ formatDateTime(manifest.created_at) }}</dd></div>
              <div class="detail-item wide"><dt>Source URI</dt><dd class="mono">{{ manifest.source_uri }}</dd></div>
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
import { getDatasetManifest } from '../api/datasets';
import type { DatasetManifestDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { formatDateTime } from '../lib/format';

const props = defineProps<{ datasetManifestId: string }>();
const { data: manifest, loading, error, run } = useAsyncData<DatasetManifestDTO>(() => getDatasetManifest(props.datasetManifestId));

onMounted(run);
</script>
