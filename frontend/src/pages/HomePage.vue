<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">TSBenchmark</p>
        <h1>{{ t('home.title') }}</h1>
        <p class="page-sub">
          {{ t('home.subtitle') }}
        </p>
      </div>
      <div class="head-actions">
        <a class="btn lg accent" href="#/new"><Icon name="sparkles" :size="18" /> {{ t('home.startNewEvaluation') }}</a>
      </div>
    </header>

    <section class="stack">
      <div class="grid-auto">
        <a class="stat-tile card interactive" href="#/datasets" style="text-decoration:none">
          <span class="stat-label"><Icon name="database" :size="14" style="vertical-align:-2px" /> {{ t('nav.datasets') }}</span>
          <span class="stat-value">{{ counts.datasets + counts.shards }}</span>
          <span class="stat-foot">{{ t('home.manifestsAndShards') }}</span>
        </a>
        <a class="stat-tile card interactive" href="#/runs" style="text-decoration:none">
          <span class="stat-label"><Icon name="activity" :size="14" style="vertical-align:-2px" /> {{ t('nav.runs') }}</span>
          <span class="stat-value">{{ counts.runs }}</span>
          <span class="stat-foot">{{ t('home.benchmarkingExecutions') }}</span>
        </a>
        <a class="stat-tile card interactive" href="#/tracks" style="text-decoration:none">
          <span class="stat-label"><Icon name="target" :size="14" style="vertical-align:-2px" /> {{ t('home.tracks') }}</span>
          <span class="stat-value">{{ counts.tracks }}</span>
          <span class="stat-foot">{{ t('home.benchmarkTargets') }}</span>
        </a>
        <div class="stat-tile">
          <span class="stat-label"><Icon name="barChart" :size="14" style="vertical-align:-2px" /> {{ t('home.reports') }}</span>
          <span class="stat-value">{{ counts.reports }}</span>
          <span class="stat-foot">{{ t('home.publishedResults') }}</span>
        </div>
      </div>

      <div class="grid-2">
        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('home.recentActivity') }}</h2></header>
          <div class="card-body">
            <StateBlock
              :loading="activityLoading"
              :error="activityError || ''"
              :empty="!activityLoading && !activityError && recents.length === 0"
              empty-icon="inbox"
              :empty-title="t('home.noActivity')"
              :empty-desc="t('home.noActivityDesc')"
              @retry="loadActivity"
            >
              <template #empty-action>
                <a class="btn sm" href="#/new"><Icon name="plus" :size="15" /> {{ t('nav.newEvaluation') }}</a>
              </template>
              <ul class="artifact-list">
                <li v-for="item in recents.slice(0, 8)" :key="`${item.kind}-${item.id}`">
                  <a :href="item.href">
                    <Icon class="art-ico" :name="kindIcon(item.kind)" :size="16" />
                    <span style="min-width:0">
                      <span class="nowrap" style="display:block;overflow:hidden;text-overflow:ellipsis;font-weight:600">{{ activityTitle(item) }}</span>
                      <span class="faint" style="font-size:0.76rem">{{ t(`home.kind.${item.kind}`) }} · {{ timeAgo(item.createdAt) }}</span>
                    </span>
                    <Icon name="arrowRight" :size="14" style="margin-left:auto;opacity:.5" />
                  </a>
                </li>
              </ul>
            </StateBlock>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">{{ t('home.howItWorks') }}</h2></header>
          <div class="card-body">
            <ol class="timeline" style="counter-reset:step">
              <li v-for="(s, i) in steps" :key="i">
                <span class="rank-badge" style="margin-top:2px">{{ i + 1 }}</span>
                <span class="tl-msg"><strong>{{ s.t }}</strong><br /><span class="faint" style="font-size:0.84rem">{{ s.d }}</span></span>
              </li>
            </ol>
            <a class="btn secondary" href="#/new" style="margin-top:14px"><Icon name="arrowRight" :size="16" /> {{ t('home.beginGuidedRun') }}</a>
          </div>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import { listDatasetManifests, listShards } from '../api/datasets';
import { listReports } from '../api/results';
import { listRuns } from '../api/runs';
import { useResourceCounts } from '../composables/useResourceCounts';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { useI18n } from 'vue-i18n';

type Kind = 'dataset' | 'shard' | 'run' | 'report';
interface ActivityItem {
  kind: Kind;
  id: string;
  title?: string;
  target?: string;
  count?: number;
  href: string;
  createdAt: string;
}

const { state: countsState, refresh: refreshCounts } = useResourceCounts();
const counts = computed(() => countsState.counts);
const { t } = useI18n();
const { formatInt, timeAgo } = useFormat();

const recents = ref<ActivityItem[]>([]);
const activityLoading = ref(true);
const { text: activityError, clear: clearActivityError, setError: setActivityError } = useDisplayMessage();

const steps = computed(() => [
  { t: t('home.steps.upload.title'), d: t('home.steps.upload.desc') },
  { t: t('home.steps.shard.title'), d: t('home.steps.shard.desc') },
  { t: t('home.steps.run.title'), d: t('home.steps.run.desc') },
  { t: t('home.steps.review.title'), d: t('home.steps.review.desc') }
]);

const ICONS: Record<Kind, string> = {
  dataset: 'database', shard: 'layers', run: 'activity', report: 'barChart'
};
const kindIcon = (k: Kind) => ICONS[k] || 'file';

function activityTitle(item: ActivityItem): string {
  if (item.kind === 'dataset') return item.title ?? '';
  if (item.kind === 'shard') return t('artifacts.shardTitle', { target: item.target ?? t('artifacts.unknownTarget') });
  if (item.kind === 'run') {
    const count = item.count ?? 0;
    return t(count === 1 ? 'artifacts.runTitleOne' : 'artifacts.runTitleOther', { count: formatInt(count) });
  }
  return t('artifacts.report');
}

async function loadActivity() {
  activityLoading.value = true;
  clearActivityError();
  try {
    // 各类只拉 top 8 就够混排出 8 条最新；4 路并行避免顺序等待。
    const [d, s, r, p] = await Promise.all([
      listDatasetManifests({ limit: 8 }),
      listShards({ limit: 8 }),
      listRuns({ limit: 8 }),
      listReports({ limit: 8 })
    ]);
    const merged: ActivityItem[] = [];
    for (const m of d.items) {
      merged.push({ kind: 'dataset', id: m.dataset_manifest_id, title: m.name, href: `#/datasets/${m.dataset_manifest_id}`, createdAt: m.created_at ?? '' });
    }
    for (const sh of s.items) {
      merged.push({ kind: 'shard', id: sh.shard_id, target: sh.target_columns?.[0], href: `#/shards/${sh.shard_id}`, createdAt: sh.created_at ?? '' });
    }
    for (const run of r.items) {
      merged.push({ kind: 'run', id: run.benchmarking_run_id, count: run.model_count || run.model_ids?.length || 0, href: `#/runs/${run.benchmarking_run_id}`, createdAt: run.created_at ?? '' });
    }
    for (const rep of p.items) {
      merged.push({ kind: 'report', id: rep.report_id, href: `#/reports/${rep.report_id}`, createdAt: rep.created_at ?? '' });
    }
    merged.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    recents.value = merged;
  } catch (e) {
    setActivityError(e, 'errors.failedToLoadActivity');
    recents.value = [];
  } finally {
    activityLoading.value = false;
  }
}

onMounted(() => {
  void refreshCounts();
  void loadActivity();
});
</script>
