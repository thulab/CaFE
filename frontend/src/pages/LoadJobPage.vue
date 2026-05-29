<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('loadJob.eyebrow') }}</p>
        <h1>{{ t('loadJob.title') }}</h1>
        <p class="page-sub">{{ t('loadJob.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <ResumeWizardButton resource-type="load_job" :resource-id="loadJobId" />
        <a class="btn secondary sm" :href="job?.output_shard_id ? `#/shards/${job.output_shard_id}` : '#/datasets'">
          <Icon name="layers" :size="15" /> {{ t('loadJob.openShard') }}
        </a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="job" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">{{ t('loadJob.status') }}</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="job.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('loadJob.seed') }}</span><span class="stat-value">{{ formatInt(job.seed ?? 0) }}</span></div>
          <div v-if="job.current_step" class="stat-tile"><span class="stat-label">{{ t('loadJob.currentStep') }}</span><span class="stat-value" style="font-size:1.1rem">{{ currentStepLabel(job.current_step) }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('loadJob.loadConfiguration') }}</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>{{ t('loadJob.loadJobId') }}</dt><dd class="mono">{{ job.load_job_id }}</dd></div>
              <div class="detail-item"><dt>{{ t('loadJob.manifest') }}</dt><dd><a class="text-link" :href="`#/datasets/${job.dataset_manifest_id}`">{{ shortId(job.dataset_manifest_id) }}</a></dd></div>
              <div class="detail-item"><dt>{{ t('loadJob.outputShard') }}</dt><dd><a v-if="job.output_shard_id" class="text-link" :href="`#/shards/${job.output_shard_id}`">{{ shortId(job.output_shard_id) }}</a><span v-else class="faint">{{ t('loadJob.pending') }}</span></dd></div>
              <div class="detail-item"><dt>{{ t('loadJob.started') }}</dt><dd>{{ formatDateTime(job.started_at) }}</dd></div>
              <div class="detail-item"><dt>{{ t('loadJob.finished') }}</dt><dd>{{ formatDateTime(job.finished_at) }}</dd></div>
              <div class="detail-item wide"><dt>{{ t('loadJob.splitConfig') }}</dt><dd><pre class="mono" style="margin:0;white-space:pre-wrap">{{ pretty(job.split_config) }}</pre></dd></div>
              <div v-if="job.error_message" class="detail-item wide"><dt>{{ t('loadJob.error') }}</dt><dd class="alert" style="margin:0"><Icon class="alert-ico" name="alert" :size="16" /><span>{{ job.error_code }} · {{ job.error_message }}</span></dd></div>
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
import { getLoadJob } from '../api/datasets';
import type { DatasetLoadJobDetailDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';

const props = defineProps<{ loadJobId: string }>();
const { data: job, loading, error, run } = useAsyncData<DatasetLoadJobDetailDTO>(() => getLoadJob(props.loadJobId));
const { t, te } = useI18n();
const { formatDateTime, formatInt } = useFormat();

onMounted(run);

function pretty(value: Record<string, unknown>) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function currentStepLabel(step: string) {
  const key = `loadJob.steps.${step}`;
  return te(key) ? t(key) : step;
}
</script>
