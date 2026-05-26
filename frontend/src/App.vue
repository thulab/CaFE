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

      <template v-if="user">
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

        <template v-if="adminItems.length">
          <p class="nav-group-label" style="margin-top:18px">Administration</p>
          <nav aria-label="Administration">
            <a
              v-for="item in adminItems"
              :key="item.key"
              class="nav-link"
              :class="{ 'is-active': item.key === route.navKey }"
              :href="item.href"
              :aria-current="item.key === route.navKey ? 'page' : undefined"
              @click="closeNav"
            >
              <Icon class="nav-ico" :name="item.icon" :size="18" />
              <span>{{ item.label }}</span>
            </a>
          </nav>
        </template>
      </template>

      <template v-else>
        <p class="nav-group-label">Public</p>
        <nav aria-label="Public">
          <a class="nav-link" :class="{ 'is-active': route.navKey === 'leaderboards' }" href="#/leaderboards" @click="closeNav">
            <Icon class="nav-ico" name="trophy" :size="18" />
            <span>Leaderboards</span>
          </a>
        </nav>
      </template>

      <span class="nav-spacer" />
      <div class="sidebar-foot">
        <template v-if="user">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px">
            <Icon name="user" :size="14" />
            <span style="font-weight:600">{{ user.username }}</span>
          </div>
          <div class="faint" style="font-size:0.74rem; margin-bottom:8px">
            <span v-for="(role, i) in user.roles" :key="role" class="badge sm" :class="role === 'admin' ? 'accent' : 'neutral'" style="margin-right:4px">{{ role }}</span>
            <span v-if="!user.roles.length">no role</span>
          </div>
          <button class="btn ghost sm" type="button" style="width:100%; justify-content:center" @click="onLogout">
            <Icon name="logOut" :size="14" /> Sign out
          </button>
        </template>
        <template v-else>
          <a class="btn accent sm" href="#/login" style="width:100%; justify-content:center" @click="closeNav">
            <Icon name="logIn" :size="14" /> Sign in
          </a>
        </template>
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
          <a
            v-if="user && hasRunExecute"
            class="btn accent sm"
            href="#/new"
          >
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
import LoginPage from './pages/LoginPage.vue';
import ForbiddenPage from './pages/ForbiddenPage.vue';
import LeaderboardsPage from './pages/LeaderboardsPage.vue';
import UsersPage from './pages/admin/UsersPage.vue';
import RolesPage from './pages/admin/RolesPage.vue';
import ProfilePage from './pages/admin/ProfilePage.vue';
import { useTheme } from './composables/useTheme';
import { countRecents } from './stores/recents';
import { shortId } from './lib/format';
import { authState, has, logout } from './stores/auth';
import { evaluateGuard, type Tier } from './composables/useAuthGuard';

const { pref, resolvedTheme, cycleTheme } = useTheme();

const routeHash = ref(currentHash());
const navOpen = ref(false);

onMounted(() => window.addEventListener('hashchange', syncRoute));
onBeforeUnmount(() => window.removeEventListener('hashchange', syncRoute));
watch(routeHash, () => { navOpen.value = false; });

const themeIcon = computed(() =>
  pref.value === 'system' ? 'monitor' : resolvedTheme.value === 'dark' ? 'moon' : 'sun'
);

const user = computed(() => authState.user);
const hasRunExecute = computed(() => has('run.execute'));
const isAdmin = computed(() => !!user.value && (user.value.is_superuser || user.value.roles.includes('admin')));

const navItems = computed(() => [
  { key: 'home', label: 'Overview', icon: 'dashboard', href: '#/', count: 0 },
  { key: 'new', label: 'New evaluation', icon: 'sparkles', href: '#/new', count: 0 },
  { key: 'datasets', label: 'Datasets', icon: 'database', href: '#/datasets', count: countRecents(['dataset', 'shard']) },
  { key: 'runs', label: 'Runs', icon: 'activity', href: '#/runs', count: countRecents('run') },
  { key: 'leaderboards', label: 'Leaderboards', icon: 'trophy', href: '#/leaderboards', count: 0 }
]);

const adminItems = computed(() => {
  if (!user.value) return [];
  const items: { key: string; label: string; icon: string; href: string }[] = [];
  if (has('user.manage')) items.push({ key: 'admin-users', label: 'Users', icon: 'users', href: '#/admin/users' });
  if (has('role.manage')) items.push({ key: 'admin-roles', label: 'Roles', icon: 'shield', href: '#/admin/roles' });
  items.push({ key: 'profile', label: 'My profile', icon: 'user', href: '#/profile' });
  return items;
});

interface RouteView {
  component: Component;
  props: Record<string, unknown>;
  navKey: string;
  tier: Tier;
  perm?: string | null;
  crumbs: Array<{ label: string; href?: string }>;
}

const HOME_CRUMB = { label: 'Overview', href: '#/' };

function resolveRoute(): RouteView {
  // Strip query before splitting path.
  const [path, _query = ''] = routeHash.value.split('?');
  const parts = path.split('/').filter(Boolean);
  const id = parts[1] || '';

  // ----- public (Tier 0) -----
  if (parts[0] === 'login') {
    return { component: LoginPage, props: {}, navKey: '', tier: 'public', crumbs: [{ label: 'Sign in' }] };
  }
  if (parts[0] === 'leaderboards') {
    return { component: LeaderboardsPage, props: {}, navKey: 'leaderboards', tier: 'public', crumbs: [{ label: 'Leaderboards' }] };
  }
  if (parts[0] === 'tracks' && id && parts[2] === 'ranking') {
    return {
      component: RankingPage, props: { trackId: id }, navKey: 'leaderboards', tier: 'public',
      crumbs: [{ label: 'Leaderboards', href: '#/leaderboards' }, { label: 'Ranking' }]
    };
  }
  if (parts[0] === 'rankings' && id) {
    return { component: RankingPage, props: { trackId: id }, navKey: 'leaderboards', tier: 'public', crumbs: [{ label: 'Leaderboards', href: '#/leaderboards' }, { label: 'Ranking' }] };
  }
  if (parts[0] === 'forbidden') {
    return { component: ForbiddenPage, props: {}, navKey: '', tier: 'public', crumbs: [{ label: '403' }] };
  }

  // ----- admin (Tier 2) -----
  if (parts[0] === 'admin' && parts[1] === 'users') {
    return {
      component: UsersPage, props: {}, navKey: 'admin-users', tier: 'perm', perm: 'user.manage',
      crumbs: [HOME_CRUMB, { label: 'Administration' }, { label: 'Users' }]
    };
  }
  if (parts[0] === 'admin' && parts[1] === 'roles') {
    return {
      component: RolesPage, props: {}, navKey: 'admin-roles', tier: 'perm', perm: 'role.manage',
      crumbs: [HOME_CRUMB, { label: 'Administration' }, { label: 'Roles' }]
    };
  }
  if (parts[0] === 'profile') {
    return {
      component: ProfilePage, props: {}, navKey: 'profile', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'My profile' }]
    };
  }

  // ----- workbench (Tier 1) -----
  if (parts[0] === 'new') {
    return { component: EvaluationWizardPage, props: {}, navKey: 'new', tier: 'perm', perm: 'run.execute', crumbs: [HOME_CRUMB, { label: 'New evaluation' }] };
  }
  if (parts[0] === 'datasets' && id) {
    return {
      component: DatasetManifestPage, props: { datasetManifestId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'datasets') {
    return { component: DatasetsPage, props: {}, navKey: 'datasets', tier: 'authed', crumbs: [HOME_CRUMB, { label: 'Datasets' }] };
  }
  if (parts[0] === 'load-jobs' && id) {
    return {
      component: LoadJobPage, props: { loadJobId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: 'Load job' }]
    };
  }
  if (parts[0] === 'shards' && id) {
    return {
      component: ShardPage, props: { shardId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Datasets', href: '#/datasets' }, { label: 'Shard' }]
    };
  }
  if (parts[0] === 'tracks' && id) {
    return { component: TrackPage, props: { trackId: id }, navKey: 'runs', tier: 'authed', crumbs: [HOME_CRUMB, { label: 'Track' }] };
  }
  if (parts[0] === 'runs' && id) {
    return {
      component: RunDetailPage, props: { runId: id }, navKey: 'runs', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Runs', href: '#/runs' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'runs') {
    return { component: RunsPage, props: {}, navKey: 'runs', tier: 'authed', crumbs: [HOME_CRUMB, { label: 'Runs' }] };
  }
  if (parts[0] === 'reports' && id) {
    return {
      component: ReportPage, props: { reportId: id }, navKey: 'runs', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Runs', href: '#/runs' }, { label: 'Report' }]
    };
  }
  if (parts[0] === 'samples' && id) {
    const qIdx = routeHash.value.indexOf('?');
    const params = new URLSearchParams(qIdx >= 0 ? routeHash.value.slice(qIdx + 1) : '');
    return {
      component: SampleForecastPage, props: { sampleId: id, runId: params.get('run_id') || '' }, navKey: 'runs', tier: 'authed',
      crumbs: [HOME_CRUMB, { label: 'Sample forecast' }]
    };
  }
  return { component: HomePage, props: {}, navKey: 'home', tier: 'authed', crumbs: [{ label: 'Overview' }] };
}

const route = computed<RouteView>(() => {
  const candidate = resolveRoute();
  const [path] = routeHash.value.split('?');

  // 匿名访问 #/ → 重定向到 #/leaderboards（设计稿 §8 匿名首屏决策）
  if (!authState.user && (path === '/' || path === '')) {
    queueRedirect('/leaderboards');
    return candidate;
  }

  const outcome = evaluateGuard(candidate.tier, candidate.perm ?? null, path);
  if (outcome.redirectTo !== null) {
    queueRedirect(outcome.redirectTo);
  }
  return candidate;
});

function queueRedirect(targetPath: string) {
  // Use microtask so we don't mutate during compute.
  queueMicrotask(() => {
    if (window.location.hash.replace(/^#/, '') !== targetPath) {
      window.location.hash = targetPath;
    }
  });
}

function syncRoute() {
  routeHash.value = currentHash();
}

function closeNav() {
  navOpen.value = false;
}

function currentHash() {
  return window.location.hash.replace(/^#/, '') || '/';
}

async function onLogout() {
  logout();
  window.location.hash = '/leaderboards';
  closeNav();
}
</script>
