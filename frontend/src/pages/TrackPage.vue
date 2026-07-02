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
        <ResumeWizardButton resource-type="track" :resource-id="trackId" />
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

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('track.testCaseSets') }}</h2>
          <span class="badge">{{ formatInt(trackShards.length || track?.shard_count || 0) }}</span>
        </header>
        <div class="card-body">
          <StateBlock
            :loading="trackShardsLoading"
            :error="trackShardsError"
            :empty="!trackShardsLoading && !trackShardsError && trackShards.length === 0"
            empty-icon="layers"
            :empty-title="t('track.noTestCaseSets')"
            :empty-desc="t('track.noTestCaseSetsDesc')"
            @retry="loadTrackShards"
          >
            <div class="table-wrap">
              <table class="data table-fixed track-shards-table">
                <thead>
                  <tr>
                    <th :class="hasTrackCovariates ? 'col-24' : 'col-28'">{{ t('track.testCaseSet') }}</th>
                    <th :class="hasTrackCovariates ? 'col-22' : 'col-26'">{{ t('track.dataset') }}</th>
                    <th :class="hasTrackCovariates ? 'col-12' : 'col-14'">{{ t('track.window') }}</th>
                    <th class="col-8 num">{{ t('track.samples') }}</th>
                    <th :class="hasTrackCovariates ? 'col-12' : 'col-14'">{{ t('track.targets') }}</th>
                    <th v-if="hasTrackCovariates" class="col-12">{{ t('track.covariates') }}</th>
                    <th class="col-10">{{ t('track.status') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="shard in trackShards" :key="shard.shard_id">
                    <td>
                      <div class="cell-stack">
                        <a class="text-link cell-title" :href="`#/shards/${shard.shard_id}`" :title="shardTitle(shard)">{{ shardTitle(shard) }}</a>
                        <div class="faint mono cell-id">{{ shortId(shard.shard_id) }}</div>
                      </div>
                    </td>
                    <td class="muted"><span class="cell-subtitle" :title="datasetLabel(shard)">{{ datasetLabel(shard) }}</span></td>
                    <td class="muted"><span class="cell-wrap" :title="windowLabel(shard)">{{ windowLabel(shard) }}</span></td>
                    <td class="muted num">{{ formatInt(shard.sample_count ?? 0) }}</td>
                    <td class="muted"><span class="cell-wrap" :title="targetLabel(shard)">{{ targetLabel(shard) }}</span></td>
                    <td v-if="hasTrackCovariates" class="muted"><span class="cell-wrap" :title="covariateLabel(shard)">{{ covariateLabel(shard) }}</span></td>
                    <td><StatusBadge :status="shard.status" /></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </StateBlock>
        </div>
      </article>

      <article v-if="track && !track.archived_at" class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('runPanel.startFromTrack') }}</h2>
          <button class="btn secondary sm" type="button" :aria-expanded="runsExpanded" @click="runsExpanded = !runsExpanded">
            <Icon name="list" :size="14" /> {{ runsExpanded ? t('track.hideRunHistory') : t('track.showRunHistory') }}
          </button>
        </header>
        <div class="card-body">
          <TrackRunPanel :track-id="trackId" :model-statuses="trackResults?.model_statuses || []" @run-created="afterRunCreated" />
        </div>
      </article>
      <p v-else class="alert" role="note"><Icon class="alert-ico" name="info" :size="16" />{{ t('track.archivedNoRuns') }}</p>

      <article v-if="runsExpanded" class="card">
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
                    <td><StatusBadge :status="run.activity_status || run.status" /></td>
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
          </div>

          <StateBlock
            :loading="loading"
            :error="error"
            :empty="!loading && !error && items.length === 0"
            :empty-title="t('ranking.noRanking')"
            :empty-desc="t('ranking.noRankingDesc')"
            @retry="load"
          >
            <RankingTable :items="items" :metric-label="metric.toUpperCase()" />
          </StateBlock>
        </div>
      </article>

      <CapabilityProfile v-if="trackResults" :report="trackResults" />

      <SampleForecastLinksCard
        v-if="trackResults"
        :source="trackResults"
        :sample-capability-block-id="sampleLinkCapabilityBlockId"
        :sample-model-id="sampleLinkModelId"
        :sample-sort="sampleLinkSort"
        @sample-page-change="loadSamplePage"
        @sample-query-change="updateSampleQuery"
      />
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
import { computed, onMounted, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import RankingTable from '../components/results/RankingTable.vue';
import CapabilityProfile from '../components/results/CapabilityProfile.vue';
import SampleForecastLinksCard from '../components/results/SampleForecastLinksCard.vue';
import TrackRunPanel from '../components/tracks/TrackRunPanel.vue';
import { getShard } from '../api/datasets';
import { getRanking, getTrackResults } from '../api/results';
import { listRuns } from '../api/runs';
import { getTrack } from '../api/tracks';
import type { BenchmarkingRunSummaryDTO, SampleForecastSort, ShardDTO, TrackDTO, TrackResultsDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';
import { syntheticCapabilityLabel } from '../lib/syntheticCapabilities';

const props = defineProps<{ trackId: string }>();
const SAMPLE_LINK_PAGE_SIZE = 10;
const metric = ref('mase');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const loading = ref(true);
const track = ref<TrackDTO | null>(null);
const trackShards = ref<ShardDTO[]>([]);
const trackShardsLoading = ref(true);
const runs = ref<BenchmarkingRunSummaryDTO[]>([]);
const runsLoading = ref(true);
const runsExpanded = ref(false);
const trackResults = ref<TrackResultsDTO | null>(null);
const sampleLinkOffset = ref(0);
const sampleLinkCapabilityBlockId = ref('');
const sampleLinkModelId = ref('');
const sampleLinkSort = ref<SampleForecastSort>('sample_index');
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: trackShardsError, clear: clearTrackShardsError, setError: setTrackShardsError } = useDisplayMessage();
const { text: runsError, clear: clearRunsError, setError: setRunsError } = useDisplayMessage();
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const dialog = reactive<{ open: boolean; action: LifecycleAction }>({
  open: false,
  action: 'archive',
});
const hasTrackCovariates = computed(() => trackShards.value.some((shard) => Boolean(shard.covariate_columns?.length)));

onMounted(() => {
  void loadTrack();
  void loadRuns();
});

async function loadTrack() {
  try {
    track.value = await getTrack(props.trackId);
    if (track.value.primary_metric_id) metric.value = track.value.primary_metric_id;
    void loadTrackShards();
    void load();
    void loadTrackResults();
  } catch {
    track.value = null;
    trackShards.value = [];
    trackResults.value = null;
    trackShardsLoading.value = false;
    void load();
  }
}

async function loadTrackShards() {
  const shardIds = track.value?.shard_ids ?? [];
  trackShards.value = [];
  clearTrackShardsError();
  if (shardIds.length === 0) {
    trackShardsLoading.value = false;
    return;
  }
  trackShardsLoading.value = true;
  try {
    trackShards.value = await Promise.all(shardIds.map((shardId) => getShard(shardId)));
  } catch (e) {
    setTrackShardsError(e, 'track.errors.failedToLoadTestCaseSets');
  } finally {
    trackShardsLoading.value = false;
  }
}

async function load() {
  loading.value = true;
  clearError();
  try {
    items.value = (await getRanking(props.trackId, metric.value, 'latest_valid_result')).items ?? [];
  } catch (e) {
    setError(e, 'ranking.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

async function loadTrackResults() {
  try {
    const result = await getTrackResults(props.trackId, {
      sampleLinkLimit: SAMPLE_LINK_PAGE_SIZE,
      sampleLinkOffset: sampleLinkOffset.value,
      sampleLinkCapabilityBlockId: sampleLinkCapabilityBlockId.value,
      sampleLinkModelId: sampleLinkModelId.value,
      sampleLinkSort: sampleLinkSort.value
    });
    trackResults.value = isTrackResults(result) ? result : null;
  } catch {
    trackResults.value = null;
  }
}

async function afterRunCreated() {
  await Promise.all([loadRuns(), loadTrackResults(), load()]);
}

function loadSamplePage(page: number) {
  sampleLinkOffset.value = Math.max(0, page - 1) * SAMPLE_LINK_PAGE_SIZE;
  void loadTrackResults();
}

function updateSampleQuery(query: { capabilityBlockId: string; modelId: string; sort: SampleForecastSort }) {
  sampleLinkCapabilityBlockId.value = query.capabilityBlockId;
  sampleLinkModelId.value = query.modelId;
  sampleLinkSort.value = query.sort;
  sampleLinkOffset.value = 0;
  void loadTrackResults();
}

function isTrackResults(value: TrackResultsDTO): value is TrackResultsDTO {
  return Boolean(value)
    && Array.isArray(value.model_statuses)
    && Array.isArray(value.model_metrics)
    && Array.isArray(value.sample_forecast_links);
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

function shardTitle(shard: ShardDTO) {
  if (shard.name) return shard.name;
  return t('artifacts.shardTitle', { target: shard.target_columns?.[0] ?? t('artifacts.unknownTarget') });
}

function windowLabel(shard: ShardDTO) {
  return t('track.windowValue', { context: shard.context_length, horizon: shard.horizon, stride: shard.stride });
}

function targetLabel(shard: ShardDTO) {
  return shard.target_columns.join(', ') || t('common.notAvailable');
}

function covariateLabel(shard: ShardDTO) {
  return shard.covariate_columns?.length ? shard.covariate_columns.join(', ') : t('common.notAvailable');
}

function datasetLabel(shard: ShardDTO) {
  if (shard.shard_type === 'synthetic') {
    const config = shard.generation_config || {};
    const capabilityId = typeof config.capability_id === 'string' ? config.capability_id : shard.capability_type;
    const fallback = typeof config.capability_label === 'string' ? config.capability_label : shard.capability_type;
    const capability = syntheticCapabilityLabel(capabilityId, fallback, t);
    return capability ? t('track.syntheticCapability', { capability }) : t('track.syntheticData');
  }
  return shard.dataset_name || shard.source_uri || t('common.notAvailable');
}
</script>
