<template>
  <div class="table-scroll">
    <table class="data-table">
      <caption>Lower is better</caption>
      <thead><tr><th>Rank</th><th>Model</th><th>Metric</th></tr></thead>
      <tbody>
        <tr v-for="row in items" :key="row.model_id">
          <td>{{ row.rank }}</td>
          <td>{{ row.model_id }}</td>
          <td>{{ row.metric_value }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
defineProps<{ items: Array<{ model_id: string; rank: number; metric_value: number }> }>();
</script>
