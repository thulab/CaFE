<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">TSBenchmark</p>
        <h1>Workbench overview</h1>
        <p class="page-sub">
          Benchmark time-series forecasting models end to end — upload data, materialize evaluation shards,
          run model adapters, and review ranked, per-sample results.
        </p>
      </div>
      <div class="head-actions">
        <a class="btn lg accent" href="#/new"><Icon name="sparkles" :size="18" /> Start new evaluation</a>
      </div>
    </header>

    <section class="stack">
      <div class="grid-auto">
        <a class="stat-tile card interactive" href="#/datasets" style="text-decoration:none">
          <span class="stat-label"><Icon name="database" :size="14" style="vertical-align:-2px" /> Datasets</span>
          <span class="stat-value">{{ counts.datasets }}</span>
          <span class="stat-foot">manifests &amp; shards created here</span>
        </a>
        <a class="stat-tile card interactive" href="#/runs" style="text-decoration:none">
          <span class="stat-label"><Icon name="activity" :size="14" style="vertical-align:-2px" /> Runs</span>
          <span class="stat-value">{{ counts.runs }}</span>
          <span class="stat-foot">benchmarking executions</span>
        </a>
        <div class="stat-tile">
          <span class="stat-label"><Icon name="target" :size="14" style="vertical-align:-2px" /> Tracks</span>
          <span class="stat-value">{{ counts.tracks }}</span>
          <span class="stat-foot">benchmark targets</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label"><Icon name="barChart" :size="14" style="vertical-align:-2px" /> Reports</span>
          <span class="stat-value">{{ counts.reports }}</span>
          <span class="stat-foot">published results</span>
        </div>
      </div>

      <div class="grid-2">
        <article class="card">
          <header class="card-head">
            <h2 class="card-title">Recent activity</h2>
            <button v-if="recents.length" class="btn ghost sm" type="button" @click="onClear">Clear</button>
          </header>
          <div class="card-body">
            <StateBlock
              :empty="recents.length === 0"
              empty-icon="inbox"
              empty-title="No activity yet"
              empty-desc="Artifacts you create appear here for quick access."
            >
              <template #empty-action>
                <a class="btn sm" href="#/new"><Icon name="plus" :size="15" /> New evaluation</a>
              </template>
              <ul class="artifact-list">
                <li v-for="item in recents.slice(0, 8)" :key="`${item.kind}-${item.id}`">
                  <a :href="item.href">
                    <Icon class="art-ico" :name="kindIcon(item.kind)" :size="16" />
                    <span style="min-width:0">
                      <span class="nowrap" style="display:block;overflow:hidden;text-overflow:ellipsis;font-weight:600">{{ item.title }}</span>
                      <span class="faint" style="font-size:0.76rem">{{ humanize(item.kind) }} · {{ timeAgo(item.createdAt) }}</span>
                    </span>
                    <Icon name="arrowRight" :size="14" style="margin-left:auto;opacity:.5" />
                  </a>
                </li>
              </ul>
            </StateBlock>
          </div>
        </article>

        <article class="card">
          <header class="card-head"><h2 class="card-title">How it works</h2></header>
          <div class="card-body">
            <ol class="timeline" style="counter-reset:step">
              <li v-for="(s, i) in steps" :key="i">
                <span class="rank-badge" style="margin-top:2px">{{ i + 1 }}</span>
                <span class="tl-msg"><strong>{{ s.t }}</strong><br /><span class="faint" style="font-size:0.84rem">{{ s.d }}</span></span>
              </li>
            </ol>
            <a class="btn secondary" href="#/new" style="margin-top:14px"><Icon name="arrowRight" :size="16" /> Begin a guided run</a>
          </div>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import { clearRecents, listRecents, type RecentKind } from '../stores/recents';
import { humanize, timeAgo } from '../lib/format';

const recents = computed(() => listRecents());
const counts = computed(() => ({
  datasets: listRecents(['dataset', 'shard']).length,
  runs: listRecents('run').length,
  tracks: listRecents('track').length,
  reports: listRecents('report').length
}));

const steps = [
  { t: 'Upload & configure', d: 'Pick a CSV, choose target columns, and set the context/horizon split.' },
  { t: 'Materialize a shard', d: 'Generate a deterministic evaluation set of forecast samples.' },
  { t: 'Run model adapters', d: 'Bind the shard to a track and execute the selected models.' },
  { t: 'Review results', d: 'Compare ranked metrics and inspect per-sample forecasts.' }
];

const ICONS: Record<RecentKind, string> = {
  dataset: 'database', shard: 'layers', track: 'target', run: 'activity', report: 'barChart'
};
const kindIcon = (k: RecentKind) => ICONS[k] || 'file';

function onClear() {
  clearRecents();
}
</script>
