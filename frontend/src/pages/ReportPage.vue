<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Report</p>
        <h1>Benchmark report</h1>
        <p class="page-subtitle">Review model metrics, task outcomes, and sample-level forecast links.</p>
      </div>
    </header>
    <p v-if="!report" class="status-line">Loading report...</p>
    <ReportSummary v-if="report" :report="report" />
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getReport } from '../api/results';
import type { ReportDTO } from '../api/types';
import ReportSummary from '../components/results/ReportSummary.vue';

const props = defineProps<{ reportId: string }>();
const report = ref<ReportDTO | null>(null);

onMounted(async () => {
  report.value = await getReport(props.reportId);
});
</script>
