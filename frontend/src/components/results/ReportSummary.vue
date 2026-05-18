<template>
  <section>
    <table>
      <tbody>
        <tr v-for="model in report.model_metrics" :key="String(model.model_id)">
          <td>{{ model.model_id }}</td>
          <td>{{ JSON.stringify(model.metrics) }}</td>
        </tr>
      </tbody>
    </table>
    <ul>
      <li v-for="task in report.task_summaries" :key="String(task.task_id)">
        <span>{{ task.task_id }} {{ task.status }}</span>
        <span v-if="task.error_message">{{ task.error_message }}</span>
      </li>
    </ul>
    <nav>
      <a v-for="link in report.sample_forecast_links" :key="link.sample_id" :href="`#/samples/${link.sample_id}?run_id=${link.run_id}`">{{ link.sample_id }}</a>
    </nav>
  </section>
</template>

<script setup lang="ts">
import type { ReportDTO } from '../../api/types';

defineProps<{ report: ReportDTO }>();
</script>
