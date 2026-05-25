<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Evaluation</p>
        <h1>Benchmarking run</h1>
        <p class="page-sub">Live execution progress, unit and task status, recent events, and report links.</p>
      </div>
      <div class="head-actions">
        <StatusBadge v-if="progress" :status="progress.status" big />
        <button v-if="isRunning" class="btn danger sm" type="button" @click="onCancel"><Icon name="ban" :size="15" /> Cancel</button>
        <a v-if="progress?.report_id" class="btn" :href="`#/reports/${progress.report_id}`"><Icon name="barChart" :size="16" /> Open report</a>
      </div>
    </header>

    <StateBlock :loading="loading && !progress" :error="error" @retry="load">
      <section v-if="progress" class="stack">
        <div class="grid-auto">
          <div class="stat-tile">
            <span class="stat-label">Models</span>
            <span class="stat-value">{{ p.completed_models ?? 0 }}<span class="faint" style="font-size:1rem"> / {{ p.total_models ?? 0 }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_models, p.total_models) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">Tasks</span>
            <span class="stat-value">{{ p.completed_tasks ?? 0 }}<span class="faint" style="font-size:1rem"> / {{ p.total_tasks ?? 0 }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_tasks, p.total_tasks) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">Samples</span>
            <span class="stat-value">{{ p.completed_samples ?? 0 }}<span class="faint" style="font-size:1rem"> / {{ p.total_samples ?? 0 }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_samples, p.total_samples) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">Failed samples</span>
            <span class="stat-value" :style="(p.failed_samples ?? 0) > 0 ? 'color:var(--danger-text)' : ''">{{ p.failed_samples ?? 0 }}</span>
            <span class="stat-foot">across all models</span>
          </div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Units</h2><span class="badge">{{ progress.units.length }}</span></header>
          <div class="card-body">
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>Model</th><th>Status</th><th class="num">Tasks</th><th>Unit</th></tr></thead>
                <tbody>
                  <tr v-for="unit in progress.units" :key="String(unit.unit_id)">
                    <td style="font-weight:600">{{ unit.model_name || unit.model_id }}</td>
                    <td><StatusBadge :status="String(unit.status)" /></td>
                    <td class="num">{{ unit.completed_task_count ?? 0 }} / {{ unit.task_count ?? 0 }}</td>
                    <td class="mono faint">{{ shortId(String(unit.unit_id)) }}</td>
                  </tr>
                  <tr v-if="!progress.units.length"><td colspan="4" class="empty-state">No units yet.</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Tasks</h2><span class="badge">{{ progress.tasks.length }}</span></header>
          <div class="card-body">
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>Capability</th><th>Status</th><th class="num">Samples</th><th>Task</th></tr></thead>
                <tbody>
                  <tr v-for="task in progress.tasks" :key="String(task.task_id)">
                    <td>{{ task.capability_block_name || task.capability_block_id }}</td>
                    <td><StatusBadge :status="String(task.status)" /></td>
                    <td class="num">{{ task.completed_sample_count ?? 0 }} / {{ task.sample_count ?? 0 }}</td>
                    <td class="mono faint">{{ shortId(String(task.task_id)) }}</td>
                  </tr>
                  <tr v-if="!progress.tasks.length"><td colspan="4" class="empty-state">No tasks yet.</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">Recent events</h2></header>
          <div class="card-body">
            <ul v-if="progress.recent_events.length" class="timeline">
              <li v-for="(event, i) in progress.recent_events" :key="`${event.created_at}-${i}`">
                <span class="tl-dot" :class="event.level || 'info'" />
                <span class="tl-msg">{{ event.message || event.event_type }}</span>
                <time :datetime="event.created_at" :title="formatDateTime(event.created_at)">{{ timeAgo(event.created_at) }}</time>
              </li>
            </ul>
            <p v-else class="empty-state">No events recorded.</p>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { cancelRun, getRunProgress } from '../api/runs';
import type { RunProgressDTO } from '../api/types';
import { formatDateTime, percent, shortId, timeAgo } from '../lib/format';

const props = defineProps<{ runId: string }>();
const TERMINAL = ['succeeded', 'partial_succeeded', 'failed', 'cancelled'];

const progress = ref<RunProgressDTO | null>(null);
const loading = ref(true);
const error = ref<string | null>(null);
let timer: ReturnType<typeof setInterval> | undefined;

const p = computed(() => progress.value?.progress ?? {});
const isRunning = computed(() => Boolean(progress.value) && !TERMINAL.includes(progress.value!.status));

onMounted(load);
onBeforeUnmount(stopPolling);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    progress.value = await getRunProgress(props.runId);
    if (isRunning.value && !timer) timer = setInterval(load, 4000);
    if (!isRunning.value) stopPolling();
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load run progress';
    stopPolling();
  } finally {
    loading.value = false;
  }
}

async function onCancel() {
  try {
    await cancelRun(props.runId);
    await load();
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to cancel run';
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}

const pct = (a: unknown, b: unknown) => percent(Number(a ?? 0), Number(b ?? 0));
</script>
