<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('datasetManifest.eyebrow') }}</p>
        <h1>{{ t('datasetManifest.title') }}</h1>
        <p class="page-sub">{{ t('datasetManifest.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <ResumeWizardButton resource-type="dataset_manifest" :resource-id="datasetManifestId" />
        <a class="btn secondary sm" href="#/datasets"><Icon name="list" :size="15" /> {{ t('datasetManifest.allDatasets') }}</a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="manifest" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">{{ t('datasetManifest.status') }}</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="manifest.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('datasetManifest.format') }}</span><span class="stat-value" style="font-size:1.2rem">{{ manifest.file_format.toUpperCase() }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('datasetManifest.valueColumns') }}</span><span class="stat-value">{{ formatInt(manifest.value_columns.length) }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('datasetManifest.domain') }}</span><span class="stat-value" style="font-size:1.2rem">{{ manifest.domain }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ manifest.name }}</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>{{ t('datasetManifest.manifestId') }}</dt><dd class="mono">{{ manifest.dataset_manifest_id }}</dd></div>
              <div class="detail-item"><dt>{{ t('datasetManifest.timeColumn') }}</dt><dd>{{ manifest.time_column }}</dd></div>
              <div class="detail-item"><dt>{{ t('datasetManifest.valueColumns') }}</dt><dd>{{ manifest.value_columns.join(', ') }}</dd></div>
              <div class="detail-item"><dt>{{ t('datasetManifest.frequency') }}</dt><dd>{{ manifest.frequency || t('common.notAvailable') }}</dd></div>
              <div class="detail-item"><dt>{{ t('datasetManifest.timezone') }}</dt><dd>{{ manifest.timezone || t('common.notAvailable') }}</dd></div>
              <div class="detail-item"><dt>{{ t('datasetManifest.created') }}</dt><dd>{{ formatDateTime(manifest.created_at) }}</dd></div>
              <div class="detail-item wide"><dt>{{ t('datasetManifest.sourceUri') }}</dt><dd class="mono">{{ manifest.source_uri }}</dd></div>
            </dl>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { getDatasetManifest } from '../api/datasets';
import type { DatasetManifestDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { useFormat } from '../composables/useFormat';

const props = defineProps<{ datasetManifestId: string }>();
const { data: manifest, loading, error, run } = useAsyncData<DatasetManifestDTO>(() => getDatasetManifest(props.datasetManifestId));
const { t } = useI18n();
const { formatDateTime, formatInt } = useFormat();

onMounted(run);
</script>
