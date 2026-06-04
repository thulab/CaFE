<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('report.eyebrow') }}</p>
        <h1>{{ t('report.title') }}</h1>
        <p class="page-sub">{{ t('report.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a v-if="report?.track_id" class="btn secondary sm" :href="`#/tracks/${report.track_id}`">
          <Icon name="chevronLeft" :size="15" /> {{ t('report.backToTrack') }}
        </a>
        <ResumeWizardButton resource-type="report" :resource-id="reportId" />
        <span class="badge primary mono">{{ shortId(reportId) }}</span>
      </div>
    </header>

    <StateBlock :loading="loading && !report" :error="error" @retry="run">
      <ReportSummary v-if="report" :report="report" @sample-page-change="loadSamplePage" />
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import ReportSummary from '../components/results/ReportSummary.vue';
import { getReport } from '../api/results';
import type { ReportDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ reportId: string }>();
const SAMPLE_LINK_PAGE_SIZE = 10;
const sampleLinkOffset = ref(0);
const { data: report, loading, error, run } = useAsyncData<ReportDTO>(() =>
  getReport(props.reportId, { sampleLinkLimit: SAMPLE_LINK_PAGE_SIZE, sampleLinkOffset: sampleLinkOffset.value })
);
const { t } = useI18n();

function loadSamplePage(page: number) {
  sampleLinkOffset.value = Math.max(0, page - 1) * SAMPLE_LINK_PAGE_SIZE;
  void run();
}

onMounted(run);
</script>
