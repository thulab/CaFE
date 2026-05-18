<template>
  <main>
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
