<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Sample forecast</p>
        <h1>Forecast comparison</h1>
        <p class="page-subtitle">Inspect target history, ground truth, model forecasts, and per-sample metrics.</p>
      </div>
    </header>

    <p v-if="!sample" class="status-line">Loading sample forecast...</p>
    <section v-if="sample" class="report-sections">
      <div class="page-card">
        <ForecastChart :sample="sample" />
      </div>
      <div class="page-card">
        <SampleMetricTable :models="sample.models" />
      </div>
      <p v-for="model in failedModels" :key="model.model_id" class="alert">
        {{ model.model_name || model.model_id }}: {{ model.error_message || model.status }}
      </p>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { getSampleForecast } from '../api/results';
import type { SampleForecastDTO } from '../api/types';
import ForecastChart from '../components/results/ForecastChart.vue';
import SampleMetricTable from '../components/results/SampleMetricTable.vue';

const props = defineProps<{ sampleId: string; runId: string }>();
const sample = ref<SampleForecastDTO | null>(null);
const failedModels = computed(() => sample.value?.models.filter((model) => model.status !== 'succeeded') || []);

onMounted(async () => {
  sample.value = await getSampleForecast(props.sampleId, props.runId);
});
</script>
