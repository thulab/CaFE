<template>
  <div class="app-shell" :class="{ 'nav-open': navOpen }">
    <aside class="app-sidebar" aria-label="Primary">
      <a class="brand" href="#/" @click="closeNav">
        <span class="brand-mark"><Icon name="gauge" :size="20" /></span>
        <span>
          <span class="brand-name">TSBenchmark</span>
          <span class="brand-sub">Workbench</span>
        </span>
      </a>

      <p class="nav-group-label">Workspace</p>
      <nav aria-label="Sections">
        <a
          v-for="item in navItems"
          :key="item.key"
          class="nav-link"
          :class="{ 'is-active': item.key === route.navKey }"
          :href="item.href"
          :aria-current="item.key === route.navKey ? 'page' : undefined"
          @click="closeNav"
        >
          <Icon class="nav-ico" :name="item.icon" :size="18" />
          <span>{{ item.label }}</span>
          <span v-if="item.count" class="badge neutral sm" style="margin-left:auto">{{ item.count }}</span>
        </a>
      </nav>

      <span class="nav-spacer" />
      <div class="sidebar-foot">
        MVP · local workspace
      </div>
    </aside>

    <div v-if="navOpen" class="nav-scrim" @click="closeNav" />

    <div class="app-main">
      <header class="app-topbar">
        <button class="icon-btn menu-toggle" type="button" aria-label="Toggle navigation" @click="navOpen = !navOpen">
          <Icon name="menu" :size="18" />
        </button>

        <nav class="breadcrumb" aria-label="Breadcrumb">
          <template v-for="(crumb, i) in route.crumbs" :key="i">
            <a v-if="crumb.href" :href="crumb.href">{{ crumb.label }}</a>
            <span v-else class="current" aria-current="page">{{ crumb.label }}</span>
            <span v-if="i < route.crumbs.length - 1" class="sep"><Icon name="chevronRight" :size="14" /></span>
          </template>
        </nav>

        <div class="topbar-actions">
          <a class="btn accent sm" href="#/new">
            <Icon name="plus" :size="16" /> New evaluation
          </a>
          <button
            class="icon-btn"
            type="button"
            :aria-label="`Theme: ${pref}. Click to change.`"
            :title="`Theme: ${pref}`"
            @click="cycleTheme"
          >
            <Icon :name="themeIcon" :size="18" />
          </button>
        </div>
      </header>

      <component :is="route.component" v-bind="route.props" :key="routeHash" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, type Component } from 'vue';
import Icon from './components/ui/Icon.vue';
import HomePage from './pages/HomePage.vue';
import DatasetsPage from './pages/DatasetsPage.vue';
import RunsPage from './pages/RunsPage.vue';
import DatasetManifestPage from './pages/DatasetManifestPage.vue';
import EvaluationWizardPage from './pages/EvaluationWizardPage.vue';
import LoadJobPage from './pages/LoadJobPage.vue';
import RankingPage from './pages/RankingPage.vue';
import ReportPage from './pages/ReportPage.vue';
import RunDetailPage from './pages/RunDetailPage.vue';
import SampleForecastPage from './pages/SampleForecastPage.vue';
import ShardPage from './pages/ShardPage.vue';
import TrackPage from './pages/TrackPage.vue';
import { useTheme } from './composables/useTheme';
import { countRecents } from './stores/recents';
import { shortId } from './lib/format';

const { pref, resolvedTheme, cycleTheme } = useTheme();

const routeHash = ref(currentHash());
const navOpen = ref(false);

onMounted(() => window.addEventListener('hashchange', syncRoute));
onBeforeUnmount(() => window.removeEventListener('hashchange', syncRoute));
watch(routeHash, () => { navOpen.value = false; });

const themeIcon = computed(() =>
  pref.value === 'system' ? 'monitor' : resolvedTheme.value === 'dark' ? 'moon' : 'sun'
);

const navItems = computed(() => [
  { key: 'home', label: 'Overview', icon: 'dashboard', href: '#/', count: 0 },
  { key: 'new', label: 'New evaluation', icon: 'sparkles', href: '#/new', count: 0 },
  { key: 'datasets', label: 'Datasets', icon: 'database', href: '#/datasets', count: countRecents(['dataset', 'shard']) },
  { key: 'runs', label: 'Runs', icon: 'activity', href: '#/runs', count: countRecents('run') }
]);

interface RouteView {
  component: Component;
  props: Record<string, unknown>;
  navKey: string;
  crumbs: Array<{ label: string; href?: string }>;
}

const HOME_CRUMB = { label: 'Overview', href: '#/' };

const route = computed<RouteView>(() => {
  const [path, query = ''] = routeHash.value.split('?');
  const parts = path.split('/').filter(Boolean);
  const params = new URLSearchParams(query);
  const id = parts[1] || '';

  if (parts[0] === 'new') {
    return { component: EvaluationWizardPage, props: {}, navKey: 'new', crumbs: [HOME_CRUMB, { label: 'New evaluation' }] };
  }
  if (parts[0] === 'datasets' && id) {
    return {
      component: DatasetManifestPage, props: { datasetManifestId: id }, navKey: 'datasets',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'datasets') {
    return { component: DatasetsPage, props: {}, navKey: 'datasets', crumbs: [HOME_CRUMB, { label: 'Datasets' }] };
  }
  if (parts[0] === 'load-jobs' && id) {
    return {
      component: LoadJobPage, props: { loadJobId: id }, navKey: 'datasets',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: 'Load job' }]
    };
  }
  if (parts[0] === 'shards' && id) {
    return {
      component: ShardPage, props: { shardId: id }, navKey: 'datasets',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: 'Shard' }]
    };
  }
  if (parts[0] === 'tracks' && id && parts[2] === 'ranking') {
    return {
      component: RankingPage, props: { trackId: id }, navKey: 'runs',
      crumbs: [HOME_CRUMB, { label: 'Track', href: `#/tracks/${id}` }, { label: 'Ranking' }]
    };
  }
  if (parts[0] === 'tracks' && id) {
    return { component: TrackPage, props: { trackId: id }, navKey: 'runs', crumbs: [HOME_CRUMB, { label: 'Track' }] };
  }
  if (parts[0] === 'rankings' && id) {
    return { component: RankingPage, props: { trackId: id }, navKey: 'runs', crumbs: [HOME_CRUMB, { label: 'Ranking' }] };
  }
  if (parts[0] === 'runs' && id) {
    return {
      component: RunDetailPage, props: { runId: id }, navKey: 'runs',
      crumbs: [HOME_CRUMB, { label: 'Runs', href: '#/runs' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'runs') {
    return { component: RunsPage, props: {}, navKey: 'runs', crumbs: [HOME_CRUMB, { label: 'Runs' }] };
  }
  if (parts[0] === 'reports' && id) {
    return {
      component: ReportPage, props: { reportId: id }, navKey: 'runs',
      crumbs: [HOME_CRUMB, { label: 'Runs', href: '#/runs' }, { label: 'Report' }]
    };
  }
  if (parts[0] === 'samples' && id) {
    return {
      component: SampleForecastPage, props: { sampleId: id, runId: params.get('run_id') || '' }, navKey: 'runs',
      crumbs: [HOME_CRUMB, { label: 'Sample forecast' }]
    };
  }
  return { component: HomePage, props: {}, navKey: 'home', crumbs: [{ label: 'Overview' }] };
});

function syncRoute() {
  routeHash.value = currentHash();
}

function closeNav() {
  navOpen.value = false;
}

function currentHash() {
  return window.location.hash.replace(/^#/, '') || '/';
}
</script>
