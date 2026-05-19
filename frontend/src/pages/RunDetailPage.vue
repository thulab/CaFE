<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Evaluation</p>
        <h1>Benchmarking run</h1>
        <p class="page-subtitle">Track model execution progress, unit and task status, recent events, and generated report links.</p>
      </div>
      <div class="header-actions">
        <a v-if="progress?.report_id" class="button-link" :href="`#/reports/${progress.report_id}`">Open report</a>
      </div>
    </header>

    <p v-if="!progress" class="status-line">Loading run progress...</p>
    <section v-else class="report-sections">
      <div class="metric-strip">
        <span class="stat-pill">{{ progress.status }}</span>
        <span class="stat-pill">{{ progress.progress.completed_models ?? 0 }}/{{ progress.progress.total_models ?? 0 }} models</span>
        <span class="stat-pill">{{ progress.progress.completed_tasks ?? 0 }}/{{ progress.progress.total_tasks ?? 0 }} tasks</span>
        <span class="stat-pill">{{ progress.progress.total_samples ?? 0 }} samples</span>
      </div>

      <article class="page-card">
        <h2 class="section-title">Units</h2>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Tasks</th>
                <th>Unit ID</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="unit in progress.units" :key="String(unit.unit_id)">
                <td>{{ unit.model_name || unit.model_id }}</td>
                <td>{{ unit.status }}</td>
                <td>{{ unit.completed_task_count ?? 0 }} / {{ unit.task_count ?? 0 }}</td>
                <td class="mono-text">{{ unit.unit_id }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="page-card">
        <h2 class="section-title">Tasks</h2>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr>
                <th>Capability</th>
                <th>Status</th>
                <th>Samples</th>
                <th>Task ID</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="task in progress.tasks" :key="String(task.task_id)">
                <td>{{ task.capability_block_name || task.capability_block_id }}</td>
                <td>{{ task.status }}</td>
                <td>{{ task.completed_sample_count ?? 0 }} / {{ task.sample_count ?? 0 }}</td>
                <td class="mono-text">{{ task.task_id }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="page-card">
        <h2 class="section-title">Recent events</h2>
        <ul class="event-list">
          <li v-for="event in progress.recent_events" :key="`${event.created_at}-${event.message}`">
            <span class="status-badge">{{ event.level || 'info' }}</span>
            <span>{{ event.message || event.event_type }}</span>
            <time>{{ event.created_at }}</time>
          </li>
        </ul>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getRunProgress } from '../api/runs';
import type { RunProgressDTO } from '../api/types';

const props = defineProps<{ runId: string }>();
const progress = ref<RunProgressDTO | null>(null);

onMounted(async () => {
  progress.value = await getRunProgress(props.runId);
});
</script>
