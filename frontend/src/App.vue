<template>
  <div class="app-frame">
    <header class="app-topbar">
      <a class="brand-link" href="#/">TSBenchmark</a>
      <nav class="top-nav" aria-label="Primary">
        <a class="top-nav-link" href="#/">Create workflow</a>
        <a v-if="wizardState.trackId" class="top-nav-link" :href="`#/tracks/${wizardState.trackId}`">Current track</a>
        <a v-if="wizardState.runId" class="top-nav-link" :href="`#/runs/${wizardState.runId}`">Current run</a>
        <a v-if="wizardState.reportId" class="top-nav-link" :href="`#/reports/${wizardState.reportId}`">Current report</a>
      </nav>
    </header>
    <component :is="currentRoute.component" v-bind="currentRoute.props" />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, type Component } from 'vue';
import DatasetManifestPage from './pages/DatasetManifestPage.vue';
import EvaluationWizardPage from './pages/EvaluationWizardPage.vue';
import LoadJobPage from './pages/LoadJobPage.vue';
import RankingPage from './pages/RankingPage.vue';
import ReportPage from './pages/ReportPage.vue';
import RunDetailPage from './pages/RunDetailPage.vue';
import SampleForecastPage from './pages/SampleForecastPage.vue';
import ShardPage from './pages/ShardPage.vue';
import TrackPage from './pages/TrackPage.vue';
import { wizardState } from './stores/wizard';

type RouteView = {
  component: Component;
  props: Record<string, unknown>;
};

const routeHash = ref(currentHash());

onMounted(() => window.addEventListener('hashchange', syncRoute));
onBeforeUnmount(() => window.removeEventListener('hashchange', syncRoute));

const currentRoute = computed<RouteView>(() => {
  const [path, query = ''] = routeHash.value.split('?');
  const parts = path.split('/').filter(Boolean);
  const params = new URLSearchParams(query);

  if (parts[0] === 'datasets' && parts[1]) {
    return { component: DatasetManifestPage, props: { datasetManifestId: parts[1] } };
  }
  if (parts[0] === 'load-jobs' && parts[1]) {
    return { component: LoadJobPage, props: { loadJobId: parts[1] } };
  }
  if (parts[0] === 'shards' && parts[1]) {
    return { component: ShardPage, props: { shardId: parts[1] } };
  }
  if (parts[0] === 'tracks' && parts[1] && parts[2] === 'ranking') {
    return { component: RankingPage, props: { trackId: parts[1] } };
  }
  if (parts[0] === 'tracks' && parts[1]) {
    return { component: TrackPage, props: { trackId: parts[1] } };
  }
  if (parts[0] === 'rankings' && parts[1]) {
    return { component: RankingPage, props: { trackId: parts[1] } };
  }
  if (parts[0] === 'runs' && parts[1]) {
    return { component: RunDetailPage, props: { runId: parts[1] } };
  }
  if (parts[0] === 'reports' && parts[1]) {
    return { component: ReportPage, props: { reportId: parts[1] } };
  }
  if (parts[0] === 'samples' && parts[1]) {
    return { component: SampleForecastPage, props: { sampleId: parts[1], runId: params.get('run_id') || '' } };
  }
  return { component: EvaluationWizardPage, props: {} };
});

function syncRoute() {
  routeHash.value = currentHash();
}

function currentHash() {
  return window.location.hash.replace(/^#/, '') || '/';
}
</script>
