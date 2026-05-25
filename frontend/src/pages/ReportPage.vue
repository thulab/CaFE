<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Report</p>
        <h1>Benchmark report</h1>
        <p class="page-sub">Model metrics, task outcomes, and per-sample forecast links for this run.</p>
      </div>
      <div v-if="report" class="head-actions">
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
import StateBlock from '../components/ui/StateBlock.vue';
import ReportSummary from '../components/results/ReportSummary.vue';
import { getReport } from '../api/results';
import type { ReportDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ reportId: string }>();
const { data: report, loading, error, run } = useAsyncData<ReportDTO>(() => getReport(props.reportId));

onMounted(run);
</script>
