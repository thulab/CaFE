<template>
  <div class="table-scroll">
    <table class="data-table">
      <caption>Sample metrics</caption>
      <thead><tr><th>Model</th><th>Status</th><th>MSE</th><th>MAE</th></tr></thead>
      <tbody>
        <tr v-for="model in models" :key="model.model_id">
          <td>{{ model.model_name || model.model_id }}</td>
          <td>{{ model.status }}</td>
          <td>{{ model.metrics?.mse }}</td>
          <td>{{ model.metrics?.mae }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import type { SampleForecastDTO } from '../../api/types';

defineProps<{ models: SampleForecastDTO['models'] }>();
</script>
