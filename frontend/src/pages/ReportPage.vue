<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('report.eyebrow') }}</p>
        <h1>{{ t('report.title') }}</h1>
        <p class="page-sub">{{ t('report.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <ResumeWizardButton resource-type="report" :resource-id="reportId" />
        <span class="badge primary mono">{{ shortId(reportId) }}</span>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <ReportSummary v-if="report" :report="report" />
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import ReportSummary from '../components/results/ReportSummary.vue';
import { getReport } from '../api/results';
import type { ReportDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ reportId: string }>();
const { data: report, loading, error, run } = useAsyncData<ReportDTO>(() => getReport(props.reportId));
const { t } = useI18n();

onMounted(run);
</script>
