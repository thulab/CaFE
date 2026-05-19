<template>
  <section class="report-sections">
    <div class="page-card">
      <div class="table-scroll">
        <table class="data-table">
          <caption>Model metrics</caption>
          <thead><tr><th>Model</th><th>Metrics</th></tr></thead>
          <tbody>
            <tr v-for="model in report.model_metrics" :key="String(model.model_id)">
              <td>{{ model.model_id }}</td>
              <td><code>{{ JSON.stringify(model.metrics) }}</code></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="page-card">
      <h2 class="progress-title">Task outcomes</h2>
      <ul class="field-stack">
        <li v-for="task in report.task_summaries" :key="String(task.task_id)">
          <span class="status-badge">{{ task.task_id }} {{ task.status }}</span>
          <span v-if="task.error_message" class="alert">{{ task.error_message }}</span>
        </li>
      </ul>
    </div>

    <div class="page-card">
      <h2 class="progress-title">Sample forecasts</h2>
      <nav class="link-list" aria-label="Sample forecast links">
        <a v-for="link in report.sample_forecast_links" :key="link.sample_id" class="text-link" :href="`#/samples/${link.sample_id}?run_id=${link.run_id}`">{{ link.sample_id }}</a>
      </nav>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { ReportDTO } from '../../api/types';

defineProps<{ report: ReportDTO }>();
</script>
