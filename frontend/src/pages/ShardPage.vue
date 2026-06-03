<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('shard.eyebrow') }}</p>
        <h1>{{ shard?.name || t('shard.title') }}</h1>
        <p class="page-sub">{{ t('shard.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <ResumeWizardButton resource-type="shard" :resource-id="shardId" />
        <a class="btn secondary sm" :href="shard?.dataset_manifest_id ? `#/datasets/${shard.dataset_manifest_id}` : '#/datasets'">
          <Icon name="database" :size="15" /> {{ t('shard.openDataset') }}
        </a>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="shard" class="stack">
        <div class="grid-auto">
          <div class="stat-tile"><span class="stat-label">{{ t('shard.status') }}</span><span class="stat-value" style="font-size:1.1rem"><StatusBadge :status="shard.status" big /></span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.samples') }}</span><span class="stat-value">{{ formatInt(shard.sample_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.rows') }}</span><span class="stat-value">{{ formatInt(shard.row_count) }}</span></div>
          <div class="stat-tile"><span class="stat-label">{{ t('shard.window') }}</span><span class="stat-value" style="font-size:1.1rem">{{ shard.context_length }}→{{ shard.horizon }}</span><span class="stat-foot">{{ t('shard.windowFoot') }}</span></div>
        </div>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('shard.configuration') }}</h2></header>
          <div class="card-body">
            <dl class="detail-grid">
              <div class="detail-item"><dt>{{ t('shard.shardId') }}</dt><dd class="mono">{{ shard.shard_id }}</dd></div>
              <div v-if="shard.name" class="detail-item"><dt>{{ t('shard.name') }}</dt><dd>{{ shard.name }}</dd></div>
              <div class="detail-item"><dt>{{ t('shard.manifest') }}</dt><dd><a class="text-link" :href="`#/datasets/${shard.dataset_manifest_id}`">{{ shortId(shard.dataset_manifest_id) }}</a></dd></div>
              <div class="detail-item"><dt>{{ t('shard.loadJob') }}</dt><dd><a v-if="shard.load_job_id" class="text-link" :href="`#/load-jobs/${shard.load_job_id}`">{{ shortId(shard.load_job_id) }}</a><span v-else class="faint">{{ t('shard.notRecorded') }}</span></dd></div>
              <div class="detail-item"><dt>{{ t('shard.split') }}</dt><dd>{{ t('shard.splitSummary', { context: shard.context_length, horizon: shard.horizon, stride: shard.stride }) }}</dd></div>
              <div class="detail-item"><dt>{{ t('shard.targets') }}</dt><dd>{{ columnLabel(shard.target_columns) }}</dd></div>
              <div class="detail-item"><dt>{{ t('shard.covariates') }}</dt><dd>{{ columnLabel(shard.covariate_columns) }}</dd></div>
              <div class="detail-item wide"><dt>{{ t('shard.sourceUri') }}</dt><dd class="mono">{{ shard.source_uri }}</dd></div>
            </dl>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('shard.shardSamples') }}</h2><span v-if="samples" class="badge">{{ formatInt(samples.total ?? samples.items.length) }}</span></header>
          <div class="card-body">
            <p v-if="sampleError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ sampleError }}</p>
            <p v-else-if="sampleLoading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('shard.loadingSamples') }}</p>
            <div v-else-if="samples" class="stack">
              <div class="table-wrap">
                <table class="data">
                  <caption>{{ samplePageRange }}</caption>
                  <thead><tr><th>{{ t('shard.sample') }}</th><th class="num">{{ t('shard.index') }}</th><th>{{ t('shard.context') }}</th><th>{{ t('shard.horizon') }}</th><th>{{ t('common.actions') }}</th></tr></thead>
                  <tbody>
                    <tr v-for="s in samples.items" :key="s.sample_id">
                      <td>
                        <span style="font-weight:650">{{ sampleLabel(s) }}</span>
                        <div class="faint mono" style="font-size:0.74rem">{{ shortId(s.sample_id) }}</div>
                      </td>
                      <td class="num">{{ s.sample_index ?? t('common.notAvailable') }}</td>
                      <td>{{ range(s.context_start, s.context_end) }}</td>
                      <td>{{ range(s.horizon_start, s.horizon_end) }}</td>
                      <td>
                        <a class="btn secondary sm" :href="sampleHref(s)">
                          <Icon name="lineChart" :size="14" /> {{ t('shard.openCurve') }}
                        </a>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div class="sample-pager" :aria-label="t('shard.samplePagination')">
                <button class="btn ghost sm" type="button" :disabled="sampleOffset <= 0 || sampleLoading" @click="changeSamplePage(previousSampleOffset)">
                  <Icon name="chevronLeft" :size="14" /> {{ t('common.previousPage') }}
                </button>
                <span class="status-line">{{ t('results.pageStatus', { page: samplePage, pages: samplePageCount }) }}</span>
                <button class="btn ghost sm" type="button" :disabled="sampleOffset + SAMPLE_PAGE_SIZE >= sampleTotal || sampleLoading" @click="changeSamplePage(nextSampleOffset)">
                  {{ t('common.nextPage') }} <Icon name="chevronRight" :size="14" />
                </button>
              </div>
            </div>
            <p v-else class="empty-state">{{ t('shard.noSampleIndex') }}</p>
          </div>
        </article>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { getShard, getShardSamples } from '../api/datasets';
import type { SampleIndexDTO, ShardDTO, ShardSamplesDTO } from '../api/types';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';

const props = defineProps<{ shardId: string }>();
const SAMPLE_PAGE_SIZE = 10;
const shard = ref<ShardDTO | null>(null);
const samples = ref<ShardSamplesDTO | null>(null);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: sampleError, clear: clearSampleError, setError: setSampleError } = useDisplayMessage();
const sampleLoading = ref(false);
const sampleOffset = ref(0);
const { t } = useI18n();
const { formatInt } = useFormat();

onMounted(run);

const sampleTotal = computed(() => samples.value?.total ?? samples.value?.items.length ?? 0);
const samplePage = computed(() => Math.floor(sampleOffset.value / SAMPLE_PAGE_SIZE) + 1);
const samplePageCount = computed(() => Math.max(1, Math.ceil(sampleTotal.value / SAMPLE_PAGE_SIZE)));
const previousSampleOffset = computed(() => Math.max(0, sampleOffset.value - SAMPLE_PAGE_SIZE));
const nextSampleOffset = computed(() => sampleOffset.value + SAMPLE_PAGE_SIZE);
const samplePageRange = computed(() => {
  if (!sampleTotal.value) return '';
  const start = sampleOffset.value + 1;
  const end = Math.min(sampleOffset.value + (samples.value?.items.length ?? 0), sampleTotal.value);
  return t('shard.samplePageRange', { start, end, total: sampleTotal.value });
});

async function run() {
  loading.value = true;
  clearError();
  try {
    shard.value = await getShard(props.shardId);
    sampleOffset.value = 0;
    await loadSamples();
  } catch (e) {
    setError(e, 'shard.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

async function loadSamples() {
  sampleLoading.value = true;
  clearSampleError();
  try {
    samples.value = await getShardSamples(props.shardId, { limit: SAMPLE_PAGE_SIZE, offset: sampleOffset.value });
  } catch (e) {
    setSampleError(e, 'shard.errors.failedToLoadSamples');
  } finally {
    sampleLoading.value = false;
  }
}

async function changeSamplePage(offset: number) {
  sampleOffset.value = offset;
  await loadSamples();
}

function range(start?: number, end?: number) {
  if (start === undefined || end === undefined) return t('common.notAvailable');
  return `${start} → ${end}`;
}

function sampleLabel(sample: SampleIndexDTO) {
  if (typeof sample.sample_index === 'number') return t('results.sampleWindowLabel', { index: sample.sample_index + 1 });
  return t('results.sampleFallbackLabel', { id: shortId(sample.sample_id) });
}

function sampleHref(sample: SampleIndexDTO) {
  return `#/shards/${props.shardId}/samples/${encodeURIComponent(sample.sample_id)}`;
}

function columnLabel(columns?: string[]) {
  return columns?.length ? columns.join(', ') : t('common.notAvailable');
}
</script>
