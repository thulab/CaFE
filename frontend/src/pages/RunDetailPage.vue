<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('runs.detail.eyebrow') }}</p>
        <h1>{{ t('runs.detail.title') }}</h1>
        <p class="page-sub">{{ t('runs.detail.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <StatusBadge v-if="progress" :status="progress.status" big />
        <span v-if="progress?.archived_at" class="badge warning">{{ t('lifecycle.archived') }}</span>
        <ResumeWizardButton resource-type="run" :resource-id="runId" />
        <button v-if="canCancel" class="btn danger sm" type="button" @click="onCancel"><Icon name="ban" :size="15" /> {{ t('common.cancel') }}</button>
        <button v-else-if="isCancelling" class="btn sm" type="button" disabled><Icon name="ban" :size="15" /> {{ t('runs.detail.cancelling') }}</button>
        <a v-if="progress?.report_id" class="btn" :href="`#/reports/${progress.report_id}`"><Icon name="barChart" :size="16" /> {{ t('runs.detail.openReport') }}</a>
        <button v-if="progress && !progress.archived_at && !isPolling" class="btn secondary sm" type="button" @click="openLifecycle('archive')">{{ t('lifecycle.archive') }}</button>
        <button v-if="progress?.archived_at" class="btn secondary sm" type="button" @click="openLifecycle('restore')">{{ t('lifecycle.restore') }}</button>
        <button v-if="progress && !isPolling" class="btn danger sm" type="button" @click="openLifecycle('purge')">{{ t('lifecycle.permanentDelete') }}</button>
      </div>
    </header>

    <StateBlock :loading="loading && !progress" :error="error" @retry="load">
      <section v-if="progress" class="stack">
        <div class="grid-auto">
          <div class="stat-tile">
            <span class="stat-label">{{ t('runs.detail.models') }}</span>
            <span class="stat-value">{{ formatInt(p.completed_models ?? 0) }}<span class="faint" style="font-size:1rem"> / {{ formatInt(p.total_models ?? 0) }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_models, p.total_models) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">{{ t('runs.detail.tasks') }}</span>
            <span class="stat-value">{{ formatInt(p.completed_tasks ?? 0) }}<span class="faint" style="font-size:1rem"> / {{ formatInt(p.total_tasks ?? 0) }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_tasks, p.total_tasks) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">{{ t('runs.detail.samples') }}</span>
            <span class="stat-value">{{ formatInt(p.completed_samples ?? 0) }}<span class="faint" style="font-size:1rem"> / {{ formatInt(p.total_samples ?? 0) }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.completed_samples, p.total_samples) + '%' }" /></div>
          </div>
          <div class="stat-tile">
            <span class="stat-label">{{ t('runs.detail.failedSamples') }}</span>
            <span class="stat-value" :style="(p.failed_samples ?? 0) > 0 ? 'color:var(--danger-text)' : ''">{{ formatInt(p.failed_samples ?? 0) }}</span>
            <span class="stat-foot">{{ t('runs.detail.acrossAllModels') }}</span>
          </div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('runs.detail.units') }}</h2><span class="badge">{{ formatInt(progress.units.length) }}</span></header>
          <div class="card-body">
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>{{ t('runs.detail.model') }}</th><th>{{ t('runs.detail.status') }}</th><th class="num">{{ t('runs.detail.tasks') }}</th><th>{{ t('runs.detail.unit') }}</th></tr></thead>
                <tbody>
                  <tr v-for="unit in progress.units" :key="String(unit.unit_id)">
                    <td style="font-weight:600">{{ unit.model_name || unit.model_id }}</td>
                    <td><StatusBadge :status="String(unit.status)" /></td>
                    <td class="num">{{ formatInt(unit.completed_task_count ?? 0) }} / {{ formatInt(unit.task_count ?? 0) }}</td>
                    <td class="mono faint">{{ shortId(String(unit.unit_id)) }}</td>
                  </tr>
                  <tr v-if="!progress.units.length"><td colspan="4" class="empty-state">{{ t('runs.detail.noUnits') }}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('runs.detail.tasks') }}</h2><span class="badge">{{ formatInt(progress.tasks.length) }}</span></header>
          <div class="card-body">
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>{{ t('runs.detail.capability') }}</th><th>{{ t('runs.detail.status') }}</th><th class="num">{{ t('runs.detail.samples') }}</th><th>{{ t('runs.detail.task') }}</th></tr></thead>
                <tbody>
                  <tr v-for="task in progress.tasks" :key="String(task.task_id)">
                    <td>{{ task.capability_block_name || task.capability_block_id }}</td>
                    <td><StatusBadge :status="String(task.status)" /></td>
                    <td class="num">{{ formatInt(task.completed_sample_count ?? 0) }} / {{ formatInt(task.sample_count ?? 0) }}</td>
                    <td class="mono faint">{{ shortId(String(task.task_id)) }}</td>
                  </tr>
                  <tr v-if="!progress.tasks.length"><td colspan="4" class="empty-state">{{ t('runs.detail.noTasks') }}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('runs.detail.recentEvents') }}</h2></header>
          <div class="card-body">
            <ul v-if="progress.recent_events.length" class="timeline">
              <li v-for="(event, i) in progress.recent_events" :key="`${event.created_at}-${i}`">
                <span class="tl-dot" :class="event.level || 'info'" />
                <span class="tl-msg">{{ event.message || event.event_type }}</span>
                <time :datetime="event.created_at" :title="formatDateTime(event.created_at)">{{ timeAgo(event.created_at) }}</time>
              </li>
            </ul>
            <p v-else class="empty-state">{{ t('runs.detail.noEvents') }}</p>
          </div>
        </article>
      </section>
    </StateBlock>
    <ResourceActionDialog
      :open="dialog.open"
      resource-type="benchmarking_run"
      :resource-id="runId"
      :action="dialog.action"
      @close="dialog.open = false"
      @done="load"
    />
  </main>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import { ApiError } from '../api/client';
import { cancelRun, getRunProgress } from '../api/runs';
import type { RunProgressDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { percent, shortId } from '../lib/format';
import { useI18n } from 'vue-i18n';

const props = defineProps<{ runId: string }>();
const TERMINAL = ['succeeded', 'partial_succeeded', 'failed', 'cancelled'];

const progress = ref<RunProgressDTO | null>(null);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
let timer: ReturnType<typeof setInterval> | undefined;
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const dialog = reactive<{ open: boolean; action: LifecycleAction }>({
  open: false,
  action: 'archive',
});

const p = computed(() => progress.value?.progress ?? {});
// 仍需轮询：非终态都要继续刷（含 cancel_requested → cancelled 的过渡）。
const isPolling = computed(() => Boolean(progress.value) && !TERMINAL.includes(progress.value!.status));
// 可发起取消：仅 queued / running；cancel_requested 已经请求过了，不再可点。
const canCancel = computed(() => progress.value?.status === 'queued' || progress.value?.status === 'running');
const isCancelling = computed(() => progress.value?.status === 'cancel_requested');

onMounted(load);
onBeforeUnmount(stopPolling);

async function load() {
  loading.value = true;
  clearError();
  try {
    progress.value = await getRunProgress(props.runId);
    if (isPolling.value && !timer) timer = setInterval(load, 4000);
    if (!isPolling.value) stopPolling();
  } catch (e) {
    setError(e, 'errors.failedToLoadRunProgress');
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
    // 后端 409：run 已是终态——按钮显示是因为本地状态过期，刷一下就同步了。
    if (e instanceof ApiError && e.status === 409) {
      await load();
      return;
    }
    setError(e, 'errors.failedToCancelRun');
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}

function openLifecycle(action: LifecycleAction) {
  dialog.action = action;
  dialog.open = true;
}

const pct = (a: unknown, b: unknown) => percent(Number(a ?? 0), Number(b ?? 0));
</script>
