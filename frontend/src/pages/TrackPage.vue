<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('track.eyebrow') }}</p>
        <h1>{{ track?.name || t('track.title') }}</h1>
        <p class="page-sub">{{ t('track.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <span v-if="track?.archived_at" class="badge warning">{{ t('lifecycle.archived') }}</span>
        <a class="btn secondary sm" :href="`#/tracks/${trackId}/ranking`"><Icon name="trophy" :size="15" /> {{ t('track.standaloneRanking') }}</a>
        <a v-if="track && !track.archived_at" class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> {{ t('track.newEvaluation') }}</a>
        <button v-if="track && !track.archived_at" class="btn secondary sm" type="button" @click="openLifecycle('archive')">{{ t('lifecycle.archive') }}</button>
        <button v-if="track?.archived_at" class="btn secondary sm" type="button" @click="openLifecycle('restore')">{{ t('lifecycle.restore') }}</button>
        <button v-if="track" class="btn danger sm" type="button" @click="openLifecycle('purge')">{{ t('lifecycle.permanentDelete') }}</button>
      </div>
    </header>

    <section class="stack">
      <article class="card">
        <header class="card-head"><h2 class="card-title">{{ t('track.metadata') }}</h2></header>
        <div class="card-body">
          <dl class="detail-grid">
            <div class="detail-item"><dt>{{ t('track.trackId') }}</dt><dd class="mono">{{ trackId }}</dd></div>
            <div v-if="track?.archived_at" class="detail-item"><dt>{{ t('lifecycle.state') }}</dt><dd><span class="badge warning">{{ t('lifecycle.archived') }}</span></dd></div>
            <div class="detail-item"><dt>{{ t('track.rankingRoute') }}</dt><dd><a class="text-link" :href="`#/tracks/${trackId}/ranking`">{{ t('track.openStandaloneRanking') }}</a></dd></div>
          </dl>
        </div>
      </article>

      <article v-if="track && !track.archived_at" class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('runPanel.startFromTrack') }}</h2>
        </header>
        <div class="card-body">
          <TrackRunPanel :track-id="trackId" @run-created="loadRuns" />
        </div>
      </article>
      <p v-else class="alert" role="note"><Icon class="alert-ico" name="info" :size="16" />{{ t('track.archivedNoRuns') }}</p>

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('track.runs') }}</h2>
          <span class="badge">{{ formatInt(runs.length) }}</span>
        </header>
        <div class="card-body">
          <StateBlock
            :loading="runsLoading"
            :error="runsError"
            :empty="!runsLoading && !runsError && runs.length === 0"
            empty-icon="activity"
            :empty-title="t('track.noRuns')"
            :empty-desc="t('track.noRunsDesc')"
            @retry="loadRuns"
          >
            <div class="table-wrap">
              <table class="data">
                <thead>
                  <tr>
                    <th>{{ t('runs.run') }}</th>
                    <th>{{ t('runs.lastStatus') }}</th>
                    <th>{{ t('runs.detail.models') }}</th>
                    <th>{{ t('runs.detail.tasks') }}</th>
                    <th>{{ t('runs.detail.samples') }}</th>
                    <th>{{ t('runs.created') }}</th>
                    <th>{{ t('common.actions') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="run in runs" :key="run.benchmarking_run_id">
                    <td>
                      <a class="text-link" :href="`#/runs/${run.benchmarking_run_id}`">{{ runTitle(run.model_count) }}</a>
                      <div class="faint mono" style="font-size:0.74rem">{{ shortId(run.benchmarking_run_id) }}</div>
                    </td>
                    <td><StatusBadge :status="run.status" /></td>
                    <td class="muted">{{ formatInt(run.model_count) }}</td>
                    <td class="muted">{{ formatInt(run.task_count) }}</td>
                    <td class="muted">{{ formatInt(run.sample_count) }}</td>
                    <td class="muted nowrap" :title="run.created_at ? formatDateTime(run.created_at) : ''">{{ run.created_at ? timeAgo(run.created_at) : t('common.notAvailable') }}</td>
                    <td>
                      <div class="pill-row">
                        <a class="btn secondary sm" :href="`#/runs/${run.benchmarking_run_id}`"><Icon name="external" :size="14" /> {{ t('runs.openRun') }}</a>
                        <a v-if="run.report_id" class="btn secondary sm" :href="`#/reports/${run.report_id}`"><Icon name="barChart" :size="14" /> {{ t('runs.detail.openReport') }}</a>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </StateBlock>
        </div>
      </article>

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('track.ranking') }}</h2>
        </header>
        <div class="card-body">
          <div class="toolbar">
            <div class="field">
              <label class="label">{{ t('ranking.metric') }}</label>
              <select v-model="metric" :aria-label="t('ranking.metric')" @change="load">
                <option value="mase">MASE</option>
                <option value="mse">MSE</option>
                <option value="mae">MAE</option>
              </select>
            </div>
            <div class="field">
              <label class="label">{{ t('ranking.policy') }}</label>
              <select v-model="policy" :aria-label="t('ranking.policy')" @change="load">
                <option value="latest_valid_result">latest_valid_result</option>
                <option value="best_result">best_result</option>
              </select>
            </div>
          </div>

          <StateBlock
            :loading="loading"
            :error="error"
            :empty="!loading && !error && items.length === 0"
            :empty-title="t('ranking.noRanking')"
            :empty-desc="t('ranking.noRankingDesc')"
            @retry="load"
          >
            <div class="stack">
              <RankingChart :items="items" />
              <hr class="divider" />
              <RankingTable :items="items" :metric-label="metric.toUpperCase()" />
            </div>
          </StateBlock>
        </div>
      </article>
    </section>
    <ResourceActionDialog
      :open="dialog.open"
      resource-type="track"
      :resource-id="trackId"
      :action="dialog.action"
      @close="dialog.open = false"
      @done="afterLifecycleDone"
    />
  </main>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import RankingTable from '../components/results/RankingTable.vue';
import RankingChart from '../components/results/RankingChart.vue';
import TrackRunPanel from '../components/tracks/TrackRunPanel.vue';
import { getRanking } from '../api/results';
import { listRuns } from '../api/runs';
import { getTrack } from '../api/tracks';
import type { BenchmarkingRunSummaryDTO, TrackDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';

const props = defineProps<{ trackId: string }>();
const metric = ref('mase');
const policy = ref('latest_valid_result');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const loading = ref(true);
const track = ref<TrackDTO | null>(null);
const runs = ref<BenchmarkingRunSummaryDTO[]>([]);
const runsLoading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: runsError, clear: clearRunsError, setError: setRunsError } = useDisplayMessage();
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const dialog = reactive<{ open: boolean; action: LifecycleAction }>({
  open: false,
  action: 'archive',
});

onMounted(() => {
  void loadTrack();
  void load();
  void loadRuns();
});

async function loadTrack() {
  try {
    track.value = await getTrack(props.trackId);
  } catch {
    track.value = null;
  }
}

async function load() {
  loading.value = true;
  clearError();
  try {
    items.value = (await getRanking(props.trackId, metric.value, policy.value)).items;
  } catch (e) {
    setError(e, 'ranking.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

async function loadRuns() {
  runsLoading.value = true;
  clearRunsError();
  try {
    runs.value = (await listRuns({ limit: 200, trackId: props.trackId })).items ?? [];
  } catch (e) {
    setRunsError(e, 'errors.failedToLoadRuns');
  } finally {
    runsLoading.value = false;
  }
}

function runTitle(modelCount: number) {
  const key = modelCount === 1 ? 'artifacts.runTitleOne' : 'artifacts.runTitleOther';
  return t(key, { count: formatInt(modelCount) });
}

function openLifecycle(action: LifecycleAction) {
  dialog.action = action;
  dialog.open = true;
}

async function afterLifecycleDone() {
  await loadTrack();
  await loadRuns();
}
</script>
