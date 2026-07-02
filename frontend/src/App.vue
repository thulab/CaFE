<template>
  <div class="app-shell" :class="{ 'nav-open': navOpen }">
    <aside class="app-sidebar" :aria-label="t('a11y.primaryNavigation')">
      <a class="brand" href="#/" @click="closeNav">
        <span class="brand-mark"><Icon name="gauge" :size="20" /></span>
        <span>
          <span class="brand-name">TSBenchmark</span>
          <span class="brand-sub">{{ t('common.workbench') }}</span>
        </span>
      </a>

      <template v-if="user">
        <p class="nav-group-label">{{ t('nav.workspace') }}</p>
        <nav :aria-label="t('a11y.sectionNavigation')">
          <a
            v-for="item in navItems"
            :key="item.key"
            class="nav-link"
            :class="{ 'is-active': item.key === route.navKey }"
            :href="item.href"
            :aria-current="item.key === route.navKey ? 'page' : undefined"
            @click="onNavItemClick(item.key, $event)"
          >
            <Icon class="nav-ico" :name="item.icon" :size="18" />
            <span>{{ item.label }}</span>
            <span v-if="item.count" class="badge neutral sm" style="margin-left:auto">{{ item.count }}</span>
          </a>
        </nav>

        <template v-if="adminItems.length">
          <p class="nav-group-label" style="margin-top:18px">{{ t('nav.administration') }}</p>
          <nav :aria-label="t('nav.administration')">
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
        <p class="nav-group-label">{{ t('nav.public') }}</p>
        <nav :aria-label="t('nav.public')">
          <a class="nav-link" :class="{ 'is-active': route.navKey === 'leaderboards' }" href="#/leaderboards" @click="closeNav">
            <Icon class="nav-ico" name="trophy" :size="18" />
            <span>{{ t('nav.leaderboards') }}</span>
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
            <span v-if="!user.roles.length">{{ t('auth.noRole') }}</span>
          </div>
          <button class="btn ghost sm" type="button" style="width:100%; justify-content:center" @click="onLogout">
            <Icon name="logOut" :size="14" /> {{ t('auth.signOut') }}
          </button>
        </template>
        <template v-else>
          <a class="btn accent sm" href="#/login" style="width:100%; justify-content:center" @click="closeNav">
            <Icon name="logIn" :size="14" /> {{ t('auth.signIn') }}
          </a>
        </template>
      </div>
    </aside>

    <div v-if="navOpen" class="nav-scrim" @click="closeNav" />

    <div class="app-main">
      <header class="app-topbar">
        <button
          v-if="route.navKey"
          class="icon-btn menu-toggle"
          type="button"
          :aria-label="t('a11y.toggleNavigation')"
          @click="navOpen = !navOpen"
        >
          <Icon name="menu" :size="18" />
        </button>

        <nav class="breadcrumb" :aria-label="t('a11y.breadcrumb')">
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
            @click.prevent="startNewEvaluation"
          >
            <Icon :name="hasActiveDraft ? 'arrowRight' : 'plus'" :size="16" /> {{ topbarEvaluationLabel }}
          </a>
          <div class="locale-switch" role="group" :aria-label="t('a11y.languageSwitcher')">
            <button
              v-for="option in localeOptions"
              :key="option.code"
              type="button"
              :class="{ 'is-active': locale === option.code }"
              :aria-pressed="locale === option.code"
              :title="t(option.labelKey)"
              @click="changeLocale(option.code)"
            >
              {{ option.short }}
            </button>
          </div>
          <button
            class="icon-btn"
            type="button"
            :aria-label="t('common.theme', { theme: t(`common.${pref}`) })"
            :title="t('common.theme', { theme: t(`common.${pref}`) })"
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
import { useI18n } from 'vue-i18n';
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
import SampleWindowPreviewPage from './pages/SampleWindowPreviewPage.vue';
import ShardPage from './pages/ShardPage.vue';
import TrackCreateWizardPage from './pages/TrackCreateWizardPage.vue';
import TrackPage from './pages/TrackPage.vue';
import TracksPage from './pages/TracksPage.vue';
import LoginPage from './pages/LoginPage.vue';
import ForbiddenPage from './pages/ForbiddenPage.vue';
import LeaderboardsPage from './pages/LeaderboardsPage.vue';
import UsersPage from './pages/admin/UsersPage.vue';
import RolesPage from './pages/admin/RolesPage.vue';
import ProfilePage from './pages/admin/ProfilePage.vue';
import { useTheme } from './composables/useTheme';
import { useResourceCounts } from './composables/useResourceCounts';
import { setLocale } from './i18n';
import { LOCALES, type LocaleCode } from './i18n/keys';
import { shortId } from './lib/format';
import { authState, has, logout } from './stores/auth';
import { hasIncompleteWizardDraft, resetWizard } from './stores/wizard';
import { evaluateGuard, type Tier } from './composables/useAuthGuard';

const { pref, resolvedTheme, cycleTheme } = useTheme();
const { t, locale } = useI18n();

const localeOptions: Array<{ code: LocaleCode; labelKey: string; short: string }> = [
  { code: 'en-US', labelKey: 'common.english', short: 'EN' },
  { code: 'zh-CN', labelKey: 'common.chinese', short: '中' },
];

function changeLocale(next: LocaleCode) {
  if ((LOCALES as readonly string[]).includes(next)) setLocale(next);
}

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
const hasActiveDraft = computed(() => hasIncompleteWizardDraft());
const topbarEvaluationLabel = computed(() => hasActiveDraft.value ? t('nav.continueEvaluation') : t('nav.newEvaluation'));
const isAdmin = computed(() => !!user.value && (user.value.is_superuser || user.value.roles.includes('admin')));

const { state: countsState, refresh: refreshCounts } = useResourceCounts();

// 在 /datasets、/runs、/ 之间切换时刷新角标，保证用户回到列表前看到的就是最新数；
// 单独的 New evaluation 提交后会自己 refresh，这里只兜底全局导航场景。
watch(routeHash, (next) => {
  const [path] = next.split('?');
  if (path === '/' || path === '/datasets' || path === '/runs') {
    void refreshCounts();
  }
});

watch(user, (next) => {
  if (next) void refreshCounts();
});

const navItems = computed(() => [
  { key: 'home', label: t('nav.overview'), icon: 'dashboard', href: '#/', count: 0 },
  { key: 'new', label: topbarEvaluationLabel.value, icon: hasActiveDraft.value ? 'arrowRight' : 'sparkles', href: '#/new', count: 0 },
  { key: 'datasets', label: t('nav.datasets'), icon: 'database', href: '#/datasets', count: countsState.counts.datasets + countsState.counts.shards },
  { key: 'tracks', label: t('nav.tracks'), icon: 'target', href: '#/tracks', count: countsState.counts.tracks },
  { key: 'runs', label: t('nav.runs'), icon: 'activity', href: '#/runs', count: countsState.counts.runs },
  { key: 'leaderboards', label: t('nav.leaderboards'), icon: 'trophy', href: '#/leaderboards', count: 0 }
]);

const adminItems = computed(() => {
  if (!user.value) return [];
  const items: { key: string; label: string; icon: string; href: string }[] = [];
  if (has('user.manage')) items.push({ key: 'admin-users', label: t('nav.users'), icon: 'users', href: '#/admin/users' });
  if (has('role.manage')) items.push({ key: 'admin-roles', label: t('nav.roles'), icon: 'shield', href: '#/admin/roles' });
  items.push({ key: 'profile', label: t('nav.profile'), icon: 'user', href: '#/profile' });
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

const HOME_CRUMB = computed(() => ({ label: t('nav.overview'), href: '#/' }));

const trackRankingCrumb = computed(() => `${t('artifacts.track')} · ${t('nav.leaderboards')}`);

function resolveRoute(): RouteView {
  // Strip query before splitting path.
  const [path, _query = ''] = routeHash.value.split('?');
  const parts = path.split('/').filter(Boolean);
  const id = parts[1] || '';

  // ----- public (Tier 0) -----
  if (!authState.user && parts.length === 0) {
    return { component: LeaderboardsPage, props: {}, navKey: 'leaderboards', tier: 'public', crumbs: [{ label: t('nav.leaderboards') }] };
  }
  if (parts[0] === 'login') {
    return { component: LoginPage, props: {}, navKey: '', tier: 'public', crumbs: [{ label: t('auth.signIn') }] };
  }
  if (parts[0] === 'leaderboards') {
    return { component: LeaderboardsPage, props: {}, navKey: 'leaderboards', tier: 'public', crumbs: [{ label: t('nav.leaderboards') }] };
  }
  if (parts[0] === 'tracks' && id && parts[2] === 'ranking') {
    return {
      component: RankingPage, props: { trackId: id }, navKey: 'leaderboards', tier: 'public',
      crumbs: [{ label: t('nav.leaderboards'), href: '#/leaderboards' }, { label: trackRankingCrumb.value }]
    };
  }
  if (parts[0] === 'rankings' && id) {
    return { component: RankingPage, props: { trackId: id }, navKey: 'leaderboards', tier: 'public', crumbs: [{ label: t('nav.leaderboards'), href: '#/leaderboards' }, { label: trackRankingCrumb.value }] };
  }
  if (parts[0] === 'forbidden') {
    return { component: ForbiddenPage, props: {}, navKey: '', tier: 'public', crumbs: [{ label: '403' }] };
  }

  // ----- admin (Tier 2) -----
  if (parts[0] === 'admin' && parts[1] === 'users') {
    return {
      component: UsersPage, props: {}, navKey: 'admin-users', tier: 'perm', perm: 'user.manage',
      crumbs: [HOME_CRUMB.value, { label: t('nav.administration') }, { label: t('nav.users') }]
    };
  }
  if (parts[0] === 'admin' && parts[1] === 'roles') {
    return {
      component: RolesPage, props: {}, navKey: 'admin-roles', tier: 'perm', perm: 'role.manage',
      crumbs: [HOME_CRUMB.value, { label: t('nav.administration') }, { label: t('nav.roles') }]
    };
  }
  if (parts[0] === 'profile') {
    return {
      component: ProfilePage, props: {}, navKey: 'profile', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.profile') }]
    };
  }

  // ----- workbench (Tier 1) -----
  if (parts[0] === 'new') {
    return { component: EvaluationWizardPage, props: {}, navKey: 'new', tier: 'perm', perm: 'run.execute', crumbs: [HOME_CRUMB.value, { label: t('nav.newEvaluation') }] };
  }
  if (parts[0] === 'datasets' && id) {
    return {
      component: DatasetManifestPage, props: { datasetManifestId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.datasets'), href: '#/datasets' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'datasets') {
    return { component: DatasetsPage, props: {}, navKey: 'datasets', tier: 'authed', crumbs: [HOME_CRUMB.value, { label: t('nav.datasets') }] };
  }
  if (parts[0] === 'load-jobs' && id) {
    return {
      component: LoadJobPage, props: { loadJobId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.datasets'), href: '#/datasets' }, { label: t('artifacts.loadJob') }]
    };
  }
  if (parts[0] === 'shards' && id && parts[2] === 'samples' && parts[3]) {
    return {
      component: SampleWindowPreviewPage, props: { shardId: id, sampleId: parts[3] }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.datasets'), href: '#/datasets' }, { label: t('artifacts.shard'), href: `#/shards/${id}` }, { label: shortId(parts[3]) }]
    };
  }
  if (parts[0] === 'shards' && id) {
    return {
      component: ShardPage, props: { shardId: id }, navKey: 'datasets', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.datasets'), href: '#/datasets' }, { label: t('artifacts.shard') }]
    };
  }
  if (parts[0] === 'tracks' && id === 'new') {
    return {
      component: TrackCreateWizardPage, props: {}, navKey: 'tracks', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.tracks'), href: '#/tracks' }, { label: t('trackCreate.title') }]
    };
  }
  if (parts[0] === 'tracks' && id) {
    return {
      component: TrackPage, props: { trackId: id }, navKey: 'tracks', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.tracks'), href: '#/tracks' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'tracks') {
    return { component: TracksPage, props: {}, navKey: 'tracks', tier: 'authed', crumbs: [HOME_CRUMB.value, { label: t('nav.tracks') }] };
  }
  if (parts[0] === 'runs' && id) {
    return {
      component: RunDetailPage, props: { runId: id }, navKey: 'runs', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.runs'), href: '#/runs' }, { label: shortId(id) }]
    };
  }
  if (parts[0] === 'runs') {
    return { component: RunsPage, props: {}, navKey: 'runs', tier: 'authed', crumbs: [HOME_CRUMB.value, { label: t('nav.runs') }] };
  }
  if (parts[0] === 'reports' && id) {
    return {
      component: ReportPage, props: { reportId: id }, navKey: 'runs', tier: 'authed',
      crumbs: [HOME_CRUMB.value, { label: t('nav.runs'), href: '#/runs' }, { label: t('nav.report') }]
    };
  }
  if (parts[0] === 'samples' && id) {
    const qIdx = routeHash.value.indexOf('?');
    const params = new URLSearchParams(qIdx >= 0 ? routeHash.value.slice(qIdx + 1) : '');
    const reportId = params.get('report_id') || '';
    const trackId = params.get('track_id') || '';
    const reportQuery = reportSampleQuery(params);
    const reportHref = reportId ? `#/reports/${reportId}${reportQuery ? `?${reportQuery}` : ''}` : '';
    const sampleNavKey = trackId ? 'tracks' : 'runs';
    const sampleCrumbs = trackId
      ? [HOME_CRUMB.value, { label: t('nav.tracks'), href: '#/tracks' }, { label: shortId(trackId), href: `#/tracks/${trackId}` }, { label: shortId(id) }]
      : [
          HOME_CRUMB.value,
          { label: t('nav.runs'), href: '#/runs' },
          reportId ? { label: t('nav.report'), href: reportHref } : { label: t('home.steps.review.title') },
          { label: shortId(id) },
        ];
    return {
      component: SampleForecastPage,
      props: {
        sampleId: id,
        runId: params.get('run_id') || '',
        trackId,
        reportId,
        sampleLinkOffset: numericQuery(params.get('sample_link_offset')),
        sampleCursorOffset: numericQuery(params.get('sample_cursor_offset')),
        sampleCapabilityBlockId: params.get('sample_link_capability_block_id') || '',
        sampleModelId: params.get('sample_link_model_id') || '',
        sampleMetric: params.get('sample_link_metric') || '',
        sampleSort: parseSampleForecastSort(params.get('sample_link_sort')),
      },
      navKey: sampleNavKey,
      tier: 'authed',
      crumbs: sampleCrumbs
    };
  }
  return { component: HomePage, props: {}, navKey: 'home', tier: 'authed', crumbs: [HOME_CRUMB.value] };
}

const route = computed<RouteView>(() => {
  void locale.value;
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

function numericQuery(value: string | null): number | undefined {
  if (value === null || value === '') return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseSampleForecastSort(value: string | null): 'sample_index' | 'metric_desc' | 'metric_asc' {
  return value === 'metric_desc' || value === 'metric_asc' || value === 'sample_index' ? value : 'sample_index';
}

function reportSampleQuery(params: URLSearchParams): string {
  const out = new URLSearchParams();
  for (const key of ['sample_link_offset', 'sample_link_capability_block_id', 'sample_link_model_id', 'sample_link_sort']) {
    const value = params.get(key);
    if (value) out.set(key, value);
  }
  return out.toString();
}

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

function onNavItemClick(key: string, event: MouseEvent) {
  if (key === 'new') {
    startNewEvaluation(event);
    return;
  }
  closeNav();
}

function startNewEvaluation(event?: Event) {
  event?.preventDefault();
  closeNav();
  if (hasIncompleteWizardDraft()) {
    window.location.hash = '/new';
    routeHash.value = '/new';
    return;
  }
  resetWizard();
  const next = `/new?session=${Date.now()}`;
  window.location.hash = next;
  routeHash.value = next;
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
