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
            <span class="stat-value">{{ formatInt(p.processed_samples ?? p.completed_samples ?? 0) }}<span class="faint" style="font-size:1rem"> / {{ formatInt(p.total_samples ?? 0) }}</span></span>
            <div class="progress" style="margin-top:8px"><span :style="{ width: pct(p.processed_samples ?? p.completed_samples, p.total_samples) + '%' }" /></div>
          </div>
          <button class="stat-tile stat-action" type="button" :disabled="(p.failed_samples ?? 0) <= 0" @click="openFailedSamples">
            <span class="stat-label">{{ t('runs.detail.failedSamples') }}</span>
            <span class="stat-value" :style="(p.failed_samples ?? 0) > 0 ? 'color:var(--danger-text)' : ''">{{ formatInt(p.failed_samples ?? 0) }}</span>
            <span class="stat-foot">{{ (p.failed_samples ?? 0) > 0 ? t('runs.detail.viewFailureReasons') : t('runs.detail.acrossAllModels') }}</span>
          </button>
        </div>

        <article v-if="failedSamplesOpen" class="card">
          <header class="card-head">
            <h2 class="card-title">{{ t('runs.detail.failedSampleReasons') }}</h2>
            <div class="head-actions">
              <span class="badge danger">{{ formatInt(failedSamplesTotal) }}</span>
              <button class="btn ghost sm" type="button" :disabled="failedSamplesLoading" @click="refreshFailedSamplePanel">
                <Icon name="refresh" :size="14" /> {{ t('common.retry') }}
              </button>
              <button v-if="canRerunFailures" class="btn secondary sm" type="button" :disabled="rerunBusy" @click="onRerunFailedSamples">
                <span v-if="rerunBusy" class="spinner" />
                <Icon v-else name="refresh" :size="14" /> {{ t('runs.detail.rerunFailedSamples') }}
              </button>
            </div>
          </header>
          <div class="card-body">
            <p v-if="rerunMessage" class="note-success"><Icon name="checkCircle" :size="16" />{{ rerunMessage }}</p>
            <section v-if="activeRerun" class="rerun-progress-panel">
              <div class="rerun-progress-head">
                <div>
                  <p class="panel-kicker">{{ t('runs.detail.rerunningFailedSamples') }}</p>
                  <p class="status-line">{{ t('runs.detail.rerunSampleProgress') }} {{ formatInt(activeRerun.processed_samples) }} / {{ formatInt(activeRerun.total_samples) }}</p>
                </div>
                <StatusBadge :status="activeRerun.activity_status || activeRerun.status" />
              </div>
              <div class="progress striped"><span :style="{ width: pct(activeRerun.processed_samples, activeRerun.total_samples) + '%' }" /></div>
              <div class="mini-stat-row">
                <span>{{ t('runs.detail.rerunSucceeded') }} <strong>{{ formatInt(activeRerun.succeeded_samples) }}</strong></span>
                <span>{{ t('runs.detail.rerunStillFailed') }} <strong>{{ formatInt(activeRerun.failed_samples) }}</strong></span>
                <span>{{ t('runs.detail.rerunPending') }} <strong>{{ formatInt(rerunPending) }}</strong></span>
              </div>
            </section>
            <StateBlock
              :loading="failedSamplesLoading && failedSummary.length === 0"
              :error="failedSamplesError || ''"
              :empty="!failedSamplesLoading && !failedSamplesError && failedSummary.length === 0"
              empty-icon="checkCircle"
              :empty-title="t('runs.detail.noFailedSamples')"
              :empty-desc="t('runs.detail.noFailedSamplesDesc')"
              @retry="refreshFailedSamplePanel"
            >
              <div class="stack compact">
                <section>
                  <h3 class="section-label">{{ t('runs.detail.failureReasonSummary') }}</h3>
                  <div class="table-wrap">
                    <table class="data table-fixed">
                      <thead>
                        <tr>
                          <th class="col-16">{{ t('runs.detail.errorCode') }}</th>
                          <th class="col-32">{{ t('runs.detail.errorReason') }}</th>
                          <th class="num col-10">{{ t('runs.detail.failedSamples') }}</th>
                          <th class="num col-10">{{ t('runs.detail.models') }}</th>
                          <th class="num col-12">{{ t('runs.detail.capability') }}</th>
                          <th class="col-20">{{ t('common.actions') }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="group in failedSummary" :key="failureGroupKey(group)">
                          <td><span class="cell-wrap mono">{{ group.error_code || t('common.notAvailable') }}</span></td>
                          <td><span class="cell-wrap" :title="group.error_message || ''">{{ group.error_message || t('common.notAvailable') }}</span></td>
                          <td class="num">{{ formatInt(group.count) }}</td>
                          <td class="num">{{ formatInt(group.model_count) }}</td>
                          <td class="num">{{ formatInt(group.capability_count) }}</td>
                          <td>
                            <button class="btn secondary sm" type="button" @click="selectFailureGroup(group)">
                              <Icon name="list" :size="14" /> {{ t('runs.detail.viewSamples') }}
                            </button>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                <section v-if="selectedFailure">
                  <div class="sample-pager">
                    <h3 class="section-label">{{ t('runs.detail.failedSampleDetails') }}</h3>
                    <div class="head-actions">
                      <button class="btn secondary sm" type="button" :disabled="failedSamplesOffset <= 0 || failedSamplesLoading" @click="changeFailedSamplePage(-1)">{{ t('results.previousPage') }}</button>
                      <span class="status-line">{{ t('results.pageStatus', { page: failedSamplePage, pages: failedSamplePageCount }) }}</span>
                      <button class="btn secondary sm" type="button" :disabled="failedSamplePage >= failedSamplePageCount || failedSamplesLoading" @click="changeFailedSamplePage(1)">{{ t('results.nextPage') }}</button>
                      <button class="btn ghost sm" type="button" @click="collapseFailedSampleDetails">
                        <Icon name="x" :size="14" /> {{ t('runs.detail.collapseSamples') }}
                      </button>
                    </div>
                  </div>
                  <div class="table-wrap">
                    <table class="data table-fixed">
                      <thead>
                        <tr>
                          <th class="col-14">{{ t('runs.detail.sample') }}</th>
                          <th class="col-20">{{ t('runs.detail.model') }}</th>
                          <th class="col-20">{{ t('runs.detail.capability') }}</th>
                          <th class="col-14">{{ t('runs.detail.errorCode') }}</th>
                          <th class="col-24">{{ t('runs.detail.errorReason') }}</th>
                          <th class="col-8">{{ t('common.actions') }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="item in failedSamples" :key="`${item.forecast_artifact_id}-${item.sample_id}`">
                          <td>
                            <span class="cell-wrap" style="font-weight:600">{{ failedSampleLabel(item) }}</span>
                            <div class="faint mono cell-id">{{ shortId(item.sample_id) }}</div>
                          </td>
                          <td><span class="cell-wrap" :title="item.model_name || item.model_id">{{ item.model_name || item.model_id }}</span></td>
                          <td><span class="cell-wrap" :title="item.capability_block_name || item.capability_block_id || ''">{{ item.capability_block_name || item.capability_block_id || t('common.notAvailable') }}</span></td>
                          <td><span class="cell-wrap mono">{{ item.error_code || t('common.notAvailable') }}</span></td>
                          <td><span class="cell-wrap" :title="item.error_message || ''">{{ item.error_message || t('common.notAvailable') }}</span></td>
                          <td>
                            <a class="btn secondary sm" :href="sampleHref(item)">
                              <Icon name="lineChart" :size="14" /> {{ t('common.open') }}
                            </a>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            </StateBlock>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('runs.detail.units') }}</h2><span class="badge">{{ formatInt(progress.units.length) }}</span></header>
          <div class="card-body">
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>{{ t('runs.detail.model') }}</th><th>{{ t('runs.detail.status') }}</th><th class="num">{{ t('runs.detail.tasks') }}</th><th>{{ t('runs.detail.unit') }}</th></tr></thead>
                <tbody>
                  <tr v-for="unit in progress.units" :key="String(unit.unit_id)">
                    <td><span class="cell-wrap" style="font-weight:600" :title="unit.model_name || unit.model_id">{{ unit.model_name || unit.model_id }}</span></td>
                    <td><StatusBadge :status="String(unit.activity_status || unit.status)" /></td>
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
                    <td><span class="cell-wrap" :title="task.capability_block_name || task.capability_block_id">{{ task.capability_block_name || task.capability_block_id }}</span></td>
                    <td><StatusBadge :status="String(task.status)" /></td>
                    <td class="num">{{ formatInt(task.processed_sample_count ?? task.completed_sample_count ?? 0) }} / {{ formatInt(task.sample_count ?? 0) }}</td>
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
import { cancelRun, getActiveFailedSampleRerun, getFailedSampleRerun, getFailedSamples, getRunProgress, rerunFailedSamples } from '../api/runs';
import type { FailedSampleDTO, FailedSampleSummaryDTO, RerunFailedSamplesDTO, RunProgressDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { percent, shortId } from '../lib/format';
import { has } from '../stores/auth';
import { useI18n } from 'vue-i18n';

const props = defineProps<{ runId: string }>();
const TERMINAL = ['succeeded', 'partial_succeeded', 'failed', 'cancelled'];
const ACTIVE_RERUN = ['queued', 'running'];
const FAILED_SAMPLE_PAGE_SIZE = 50;

const progress = ref<RunProgressDTO | null>(null);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: failedSamplesError, clear: clearFailedSamplesError, setError: setFailedSamplesError } = useDisplayMessage();
let timer: ReturnType<typeof setInterval> | undefined;
let rerunTimer: ReturnType<typeof setInterval> | undefined;
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const failedSamplesOpen = ref(false);
const failedSamplesLoading = ref(false);
const failedSamples = ref<FailedSampleDTO[]>([]);
const failedSummary = ref<FailedSampleSummaryDTO[]>([]);
const failedSamplesTotal = ref(0);
const failedDetailsTotal = ref(0);
const failedSamplesOffset = ref(0);
const selectedFailure = ref<FailedSampleSummaryDTO | null>(null);
const activeRerun = ref<RerunFailedSamplesDTO | null>(null);
const rerunBusy = ref(false);
const rerunMessage = ref('');
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
const isRerunActive = computed(() => Boolean(activeRerun.value) && ACTIVE_RERUN.includes(activeRerun.value!.status));
const canRerunFailures = computed(() => has('run.execute') && failedSamplesTotal.value > 0 && !isPolling.value && !isRerunActive.value);
const rerunPending = computed(() => Math.max(0, Number(activeRerun.value?.total_samples ?? 0) - Number(activeRerun.value?.processed_samples ?? 0)));
const failedSamplePage = computed(() => Math.floor(failedSamplesOffset.value / FAILED_SAMPLE_PAGE_SIZE) + 1);
const failedSamplePageCount = computed(() => Math.max(1, Math.ceil(failedDetailsTotal.value / FAILED_SAMPLE_PAGE_SIZE)));

onMounted(load);
onBeforeUnmount(() => {
  stopPolling();
  stopRerunPolling();
});

async function load() {
  loading.value = true;
  clearError();
  try {
    progress.value = await getRunProgress(props.runId);
    await loadActiveRerun();
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

async function openFailedSamples() {
  if ((p.value.failed_samples ?? 0) <= 0) return;
  failedSamplesOpen.value = true;
  await refreshFailedSamplePanel();
}

async function refreshFailedSamplePanel() {
  failedSamplesLoading.value = true;
  clearFailedSamplesError();
  rerunMessage.value = '';
  try {
    const summary = await getFailedSamples(props.runId, { limit: 0 });
    failedSummary.value = summary.summary;
    failedSamplesTotal.value = summary.total;
    if (selectedFailure.value) {
      await loadFailedSampleDetails();
    } else {
      failedSamples.value = [];
      failedDetailsTotal.value = 0;
    }
  } catch (e) {
    setFailedSamplesError(e, 'runs.detail.errors.failedToLoadFailedSamples');
  } finally {
    failedSamplesLoading.value = false;
  }
}

async function onRerunFailedSamples() {
  rerunBusy.value = true;
  clearFailedSamplesError();
  rerunMessage.value = '';
  try {
    activeRerun.value = await rerunFailedSamples(props.runId);
    failedSamplesOpen.value = true;
    await refreshFailedSamplePanel();
    startRerunPolling();
  } catch (e) {
    setFailedSamplesError(e, 'runs.detail.errors.failedToRerunFailedSamples');
  } finally {
    rerunBusy.value = false;
  }
}

function stopPolling() {
  if (timer) clearInterval(timer);
  timer = undefined;
}

function startRerunPolling() {
  if (!isRerunActive.value) {
    stopRerunPolling();
    return;
  }
  if (!rerunTimer) rerunTimer = setInterval(refreshActiveRerun, 2000);
}

function stopRerunPolling() {
  if (rerunTimer) clearInterval(rerunTimer);
  rerunTimer = undefined;
}

async function loadActiveRerun() {
  try {
    activeRerun.value = (await getActiveFailedSampleRerun(props.runId)).active;
    if (activeRerun.value) {
      failedSamplesOpen.value = true;
      await refreshFailedSamplePanel();
      startRerunPolling();
    } else {
      stopRerunPolling();
    }
  } catch (_error) {
    activeRerun.value = null;
    stopRerunPolling();
  }
}

async function refreshActiveRerun() {
  if (!activeRerun.value) return;
  try {
    const job = await getFailedSampleRerun(props.runId, activeRerun.value.rerun_job_id);
    activeRerun.value = job;
    if (!isRerunActive.value) {
      stopRerunPolling();
      await load();
      await refreshFailedSamplePanel();
      rerunMessage.value = t('runs.detail.rerunComplete', { count: formatInt(job.processed_samples), remaining: formatInt(job.failed_samples) });
    }
  } catch (e) {
    setFailedSamplesError(e, 'runs.detail.errors.failedToLoadFailedSamples');
    stopRerunPolling();
  }
}

function openLifecycle(action: LifecycleAction) {
  dialog.action = action;
  dialog.open = true;
}

const pct = (a: unknown, b: unknown) => percent(Number(a ?? 0), Number(b ?? 0));

function failedSampleLabel(item: FailedSampleDTO) {
  if (typeof item.sample_index === 'number') return t('results.sampleWindowLabel', { index: item.sample_index + 1 });
  return t('results.sampleFallbackLabel', { id: shortId(item.sample_id) });
}

function failureGroupKey(group: FailedSampleSummaryDTO) {
  return `${group.error_code || ''}:${group.error_message || ''}`;
}

async function selectFailureGroup(group: FailedSampleSummaryDTO) {
  selectedFailure.value = group;
  failedSamplesOffset.value = 0;
  await loadFailedSampleDetails();
}

async function changeFailedSamplePage(delta: number) {
  failedSamplesOffset.value = Math.max(0, failedSamplesOffset.value + delta * FAILED_SAMPLE_PAGE_SIZE);
  await loadFailedSampleDetails();
}

function collapseFailedSampleDetails() {
  selectedFailure.value = null;
  failedSamples.value = [];
  failedDetailsTotal.value = 0;
  failedSamplesOffset.value = 0;
}

async function loadFailedSampleDetails() {
  if (!selectedFailure.value) return;
  const page = await getFailedSamples(props.runId, {
    limit: FAILED_SAMPLE_PAGE_SIZE,
    offset: failedSamplesOffset.value,
    error_code: selectedFailure.value.error_code,
    error_message: selectedFailure.value.error_message,
  });
  failedSamples.value = page.items;
  failedDetailsTotal.value = page.total;
}

function sampleHref(item: FailedSampleDTO) {
  return `#/samples/${encodeURIComponent(item.sample_id)}?run_id=${encodeURIComponent(props.runId)}${progress.value?.report_id ? `&report_id=${encodeURIComponent(progress.value.report_id)}` : ''}`;
}
</script>
