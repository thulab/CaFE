<template>
  <main v-if="sample">
    <ForecastChart :sample="sample" />
    <SampleMetricTable :models="sample.models" />
    <p v-for="model in failedModels" :key="model.model_id">Error: {{ model.error_message || model.status }}</p>
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
