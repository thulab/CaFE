<template>
  <figure class="chart-frame">
    <figcaption class="chart-title">History</figcaption>
    <svg class="forecast-svg" width="640" height="220" viewBox="0 0 320 120" role="img" aria-label="Forecast chart">
      <polyline :points="historyPoints" fill="none" stroke="#2563eb" stroke-width="2" />
      <polyline :points="truthPoints" fill="none" stroke="#111827" stroke-width="2" />
      <polyline
        v-for="model in successfulModels"
        :key="model.model_id"
        :points="forecastPoints(model.forecast || [])"
        fill="none"
        stroke="#dc2626"
        stroke-width="2"
      />
    </svg>
    <div class="chart-legend" aria-label="Chart legend">
      <span class="legend-item"><span class="legend-swatch history"></span>Past context</span>
      <span class="legend-item"><span class="legend-swatch truth"></span>Observed future</span>
      <span class="legend-item"><span class="legend-swatch forecast"></span>Model forecast</span>
    </div>
    <figcaption class="field-help">Ground truth</figcaption>
  </figure>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { SampleForecastDTO } from '../../api/types';

const props = defineProps<{ sample: SampleForecastDTO }>();
const successfulModels = computed(() => props.sample.models.filter((model) => model.status === 'succeeded' && model.forecast));
const historyPoints = computed(() => points(props.sample.target_history, 0));
const truthPoints = computed(() => points(props.sample.target_future, props.sample.target_history.length));

function forecastPoints(values: number[][]) {
  return points(values, props.sample.target_history.length);
}

function points(values: number[][], offset: number) {
  return values.map((row, index) => `${(offset + index) * 24 + 8},${110 - row[0] * 10}`).join(' ');
}
</script>
