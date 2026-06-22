# Frontend Bilingual i18n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add English and Chinese display-language support to the Vue frontend while keeping the backend API contract unchanged.

**Architecture:** The frontend owns all user-visible translations through `vue-i18n`, with `en-US` as the fallback locale and `zh-CN` as the second supported locale. API DTO values, ids, status codes, permission codes, metric keys, and backend `message` values remain language-neutral inputs; display helpers translate them at the edge.

**Tech Stack:** Vue 3.5, Vite 7, TypeScript, `vue-i18n`, Vitest, Vue Testing Library, current self-rolled hash routing.

---

## Source Spec

Implement the design in `docs/superpowers/specs/2026-05-26-frontend-bilingual-i18n-design.md`.

Do not modify backend code. Do not introduce `vue-router`. Do not translate model ids, dataset ids, run ids, permission codes, metric keys, API error codes, CSV filenames, or CSV column names.

---

## File Structure

Create:

- `frontend/src/i18n/keys.ts` - locale constants, locale normalization, URL/localStorage resolution.
- `frontend/src/i18n/locales/en-US.ts` - English message catalog.
- `frontend/src/i18n/locales/zh-CN.ts` - Chinese message catalog with the same key shape as English.
- `frontend/src/i18n/index.ts` - `vue-i18n` instance and runtime locale setter.
- `frontend/src/composables/useFormat.ts` - locale-aware formatting wrapper.
- `frontend/src/lib/errors.ts` - `ApiError` to translated display-message helper.
- `frontend/src/tests/i18n.test.ts` - locale key and locale resolution tests.
- `frontend/src/tests/format.test.ts` - locale-aware formatting tests.
- `frontend/src/tests/errors.test.ts` - translated error display tests.

Modify:

- `frontend/package.json` and `frontend/package-lock.json` - add `vue-i18n`.
- `frontend/src/main.ts` - register i18n with Vue.
- `frontend/src/lib/format.ts` - accept explicit locale parameters and use `Intl.RelativeTimeFormat`.
- `frontend/src/components/ui/StateBlock.vue` - translate default state text.
- `frontend/src/components/ui/StatusBadge.vue` - translate known status labels.
- `frontend/src/styles.css` - add compact language switch styles.
- `frontend/src/App.vue` - translate app shell, navigation, breadcrumbs, theme labels, and language switch.
- `frontend/src/pages/EvaluationWizardPage.vue` and `frontend/src/components/wizard/*.vue` - translate the guided workflow.
- `frontend/src/pages/HomePage.vue`, `frontend/src/pages/DatasetsPage.vue`, `frontend/src/pages/RunsPage.vue`, `frontend/src/pages/RunDetailPage.vue` - translate workspace pages.
- `frontend/src/pages/LeaderboardsPage.vue`, `frontend/src/pages/RankingPage.vue`, `frontend/src/pages/ReportPage.vue`, `frontend/src/pages/TrackPage.vue`, `frontend/src/pages/SampleForecastPage.vue`, `frontend/src/components/results/*.vue` - translate result pages/components.
- `frontend/src/pages/LoginPage.vue`, `frontend/src/pages/ForbiddenPage.vue`, `frontend/src/pages/LoadJobPage.vue`, `frontend/src/pages/DatasetManifestPage.vue`, `frontend/src/pages/ShardPage.vue`, `frontend/src/pages/admin/*.vue` - translate remaining page surfaces.
- Existing `frontend/src/tests/*.test.ts` files - keep default assertions in English, add focused Chinese-switch tests where behavior changes.

---

## Task 1: i18n Foundation

**Files:**

- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `frontend/src/main.ts`
- Create: `frontend/src/i18n/keys.ts`
- Create: `frontend/src/i18n/locales/en-US.ts`
- Create: `frontend/src/i18n/locales/zh-CN.ts`
- Create: `frontend/src/i18n/index.ts`
- Test: `frontend/src/tests/i18n.test.ts`

- [ ] **Step 1: Install `vue-i18n`**

Run:

```bash
cd frontend && npm install vue-i18n
```

Expected:

- `frontend/package.json` includes `vue-i18n` under `dependencies`.
- `frontend/package-lock.json` changes.
- Command exits with status 0.

- [ ] **Step 2: Add locale key helpers**

Create `frontend/src/i18n/keys.ts`:

```ts
export const LOCALES = ['en-US', 'zh-CN'] as const;
export type LocaleCode = typeof LOCALES[number];

export const DEFAULT_LOCALE: LocaleCode = 'en-US';
export const STORAGE_KEY = 'tsbenchmark.locale';

export function isLocaleCode(value: unknown): value is LocaleCode {
  return typeof value === 'string' && (LOCALES as readonly string[]).includes(value);
}

export function normalizeLocale(value: unknown): LocaleCode | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  const normalized = value.trim();
  if (isLocaleCode(normalized)) return normalized;
  const lower = normalized.toLowerCase();
  if (lower === 'zh' || lower.startsWith('zh-')) return 'zh-CN';
  if (lower === 'en' || lower.startsWith('en-')) return 'en-US';
  return null;
}

export function readLocaleFromUrl(search = window.location.search, hash = window.location.hash): LocaleCode | null {
  const direct = new URLSearchParams(search).get('lang');
  const fromDirect = normalizeLocale(direct);
  if (fromDirect) return fromDirect;

  const queryIndex = hash.indexOf('?');
  if (queryIndex < 0) return null;
  const hashQuery = hash.slice(queryIndex + 1);
  return normalizeLocale(new URLSearchParams(hashQuery).get('lang'));
}

export function readStoredLocale(storage: Pick<Storage, 'getItem'> = window.localStorage): LocaleCode | null {
  try {
    return normalizeLocale(storage.getItem(STORAGE_KEY));
  } catch (_error) {
    return null;
  }
}

export function resolveInitialLocale(): LocaleCode {
  return readLocaleFromUrl()
    ?? readStoredLocale()
    ?? normalizeLocale(window.navigator.language)
    ?? DEFAULT_LOCALE;
}
```

- [ ] **Step 3: Add English messages**

Create `frontend/src/i18n/locales/en-US.ts`:

```ts
const enUS = {
  common: {
    appName: 'TSBenchmark',
    loading: 'Loading...',
    retry: 'Try again',
    cancel: 'Cancel',
    open: 'Open',
    reset: 'Reset',
    back: 'Back',
    done: 'Done',
    current: 'Current',
    ready: 'Ready',
    locked: 'Locked',
    inProgress: 'In progress',
    none: 'None',
    notAvailable: '—',
    language: 'Language',
    english: 'English',
    chinese: '中文',
    switchLanguage: 'Switch language',
    theme: 'Theme: {theme}. Click to change.',
    system: 'system',
    dark: 'dark',
    light: 'light',
  },
  nav: {
    workspace: 'Workspace',
    administration: 'Administration',
    public: 'Public',
    overview: 'Overview',
    newEvaluation: 'New evaluation',
    datasets: 'Datasets',
    runs: 'Runs',
    leaderboards: 'Leaderboards',
    users: 'Users',
    roles: 'Roles',
    profile: 'My profile',
    report: 'Report',
  },
  auth: {
    signIn: 'Sign in',
    signOut: 'Sign out',
    noRole: 'no role',
  },
  state: {
    somethingWentWrong: 'Something went wrong',
    nothingHereYet: 'Nothing here yet',
  },
  status: {
    succeeded: 'Succeeded',
    success: 'Success',
    ready: 'Ready',
    complete: 'Complete',
    completed: 'Completed',
    ready_to_load: 'Ready to load',
    partial_succeeded: 'Partially succeeded',
    running: 'Running',
    in_progress: 'In progress',
    processing: 'Processing',
    queued: 'Queued',
    pending: 'Pending',
    idle: 'Idle',
    failed: 'Failed',
    error: 'Error',
    cancelled: 'Cancelled',
    canceled: 'Canceled',
  },
  errors: {
    apiError: 'Request failed',
    auth_required: 'Please sign in to continue.',
    forbidden: 'You do not have permission to perform this action.',
    invalid_json_response: 'API returned an invalid JSON response.',
    failedToLoadModels: 'Failed to load models',
    failedToStartRun: 'Failed to start run',
    failedToReadProgress: 'Failed to read progress',
    failedToCancelRun: 'Failed to cancel run',
    failedToLoadActivity: 'Failed to load activity',
    failedToLoadDatasets: 'Failed to load datasets',
    failedToLoadRuns: 'Failed to load runs',
  },
  artifacts: {
    datasetManifest: 'Dataset manifest',
    loadJob: 'Load job',
    shard: 'Shard',
    track: 'Track',
    run: 'Run',
    report: 'Report',
    createdArtifacts: 'Created artifacts',
    shardTitle: 'Shard · {target}',
    runTitle: 'Run · {count} models',
  },
  wizard: {
    eyebrow: 'Guided workflow',
    title: 'New evaluation',
    subtitle: 'Take a time-series CSV from upload to a published benchmark report — configure the split, materialize an evaluation shard, run model adapters, and review results.',
    progress: 'Progress · {done}/{total}',
    footComplete: 'Step complete — pick the next step on the left.',
    footIncomplete: 'Finish this step to unlock the next one.',
    steps: {
      uploadCsv: {
        title: 'Upload CSV',
        kicker: 'Data source',
        description: 'Pick a local CSV and inspect detected columns before configuring a benchmark.',
      },
      configureSplit: {
        title: 'Configure split',
        kicker: 'Dataset manifest',
        description: 'Choose the time and target columns, then set context, horizon, and stride.',
      },
      confirmShard: {
        title: 'Confirm shard',
        kicker: 'Sample load',
        description: 'Review the materialized shard so the run starts from a known evaluation set.',
      },
      createTrack: {
        title: 'Create track',
        kicker: 'Benchmark target',
        description: 'Bind the shard to a real-dataset track with MASE as the primary metric.',
      },
      runModels: {
        title: 'Run models',
        kicker: 'Execution',
        description: 'Select model adapters and start the run; progress updates until completion.',
      },
      openReport: {
        title: 'Open report',
        kicker: 'Results',
        description: 'Jump into the generated report once the run publishes its identifier.',
      },
    },
    runStep: {
      modelAdapters: 'Model adapters',
      selectAll: 'Select all',
      clearAll: 'Clear all',
      availableModels: 'Available models',
      loadingModels: 'Loading model adapters...',
      loadingSelectedModels: 'Loading selected timer-service models...',
      percentComplete: '{percent}% complete',
      modelsProgress: '{done}/{total} models',
      tasksProgress: '{done}/{total} tasks',
      samplesProgress: '{done}/{total} samples',
      openRun: 'Open run',
      run: 'Run',
      loaded: 'loaded',
      loading: 'loading',
      notLoaded: 'not loaded',
    },
  },
  home: {
    eyebrow: 'TSBenchmark',
    title: 'Workbench overview',
    subtitle: 'Benchmark time-series forecasting models end to end — upload data, materialize evaluation shards, run model adapters, and review ranked, per-sample results.',
    startNewEvaluation: 'Start new evaluation',
    datasets: 'Datasets',
    runs: 'Runs',
    tracks: 'Tracks',
    reports: 'Reports',
    manifestsAndShards: 'manifests & shards',
    benchmarkingExecutions: 'benchmarking executions',
    benchmarkTargets: 'benchmark targets',
    publishedResults: 'published results',
    recentActivity: 'Recent activity',
    noActivity: 'No activity yet',
    noActivityDesc: 'Artifacts you create appear here for quick access.',
    howItWorks: 'How it works',
    beginGuidedRun: 'Begin a guided run',
    kind: {
      dataset: 'Dataset',
      shard: 'Shard',
      run: 'Run',
      report: 'Report',
    },
    steps: {
      upload: { title: 'Upload & configure', desc: 'Pick a CSV, choose target columns, and set the context/horizon split.' },
      shard: { title: 'Materialize a shard', desc: 'Generate a deterministic evaluation set of forecast samples.' },
      run: { title: 'Run model adapters', desc: 'Bind the shard to a track and execute the selected models.' },
      review: { title: 'Review results', desc: 'Compare ranked metrics and inspect per-sample forecasts.' },
    },
  },
  datasets: {
    eyebrow: 'Workspace',
    title: 'Datasets',
    subtitle: 'Dataset manifests and evaluation shards stored in this workspace.',
    noDatasets: 'No datasets yet',
    noDatasetsDesc: 'Upload a CSV in a new evaluation to create your first dataset manifest and shard.',
    uploadCsv: 'Upload a CSV',
    artifact: 'Artifact',
    type: 'Type',
    detail: 'Detail',
    created: 'Created',
    rows: '{count} rows',
  },
  runs: {
    eyebrow: 'Workspace',
    title: 'Runs',
    subtitle: 'Benchmarking runs launched in this workspace. Open one for live progress and results.',
    noRuns: 'No runs yet',
    noRunsDesc: 'Create a track and execute model adapters in a new evaluation to see runs here.',
    startRun: 'Start a run',
    run: 'Run',
    lastStatus: 'Last status',
    created: 'Created',
  },
};

export default enUS;
```

- [ ] **Step 4: Add Chinese messages with the same shape**

Create `frontend/src/i18n/locales/zh-CN.ts`:

```ts
const zhCN = {
  common: {
    appName: 'TSBenchmark',
    loading: '加载中...',
    retry: '重试',
    cancel: '取消',
    open: '打开',
    reset: '重置',
    back: '返回',
    done: '完成',
    current: '当前',
    ready: '就绪',
    locked: '锁定',
    inProgress: '进行中',
    none: '无',
    notAvailable: '—',
    language: '语言',
    english: 'English',
    chinese: '中文',
    switchLanguage: '切换语言',
    theme: '主题：{theme}。点击切换。',
    system: '跟随系统',
    dark: '深色',
    light: '浅色',
  },
  nav: {
    workspace: '工作区',
    administration: '管理',
    public: '公开',
    overview: '概览',
    newEvaluation: '新建评测',
    datasets: '数据集',
    runs: '运行',
    leaderboards: '排行榜',
    users: '用户',
    roles: '角色',
    profile: '我的资料',
    report: '报告',
  },
  auth: {
    signIn: '登录',
    signOut: '退出登录',
    noRole: '无角色',
  },
  state: {
    somethingWentWrong: '出了点问题',
    nothingHereYet: '暂无内容',
  },
  status: {
    succeeded: '成功',
    success: '成功',
    ready: '就绪',
    complete: '完成',
    completed: '已完成',
    ready_to_load: '可加载',
    partial_succeeded: '部分成功',
    running: '运行中',
    in_progress: '进行中',
    processing: '处理中',
    queued: '排队中',
    pending: '待处理',
    idle: '空闲',
    failed: '失败',
    error: '错误',
    cancelled: '已取消',
    canceled: '已取消',
  },
  errors: {
    apiError: '请求失败',
    auth_required: '请先登录。',
    forbidden: '你没有执行此操作的权限。',
    invalid_json_response: 'API 返回了无效的 JSON 响应。',
    failedToLoadModels: '加载模型失败',
    failedToStartRun: '启动运行失败',
    failedToReadProgress: '读取进度失败',
    failedToCancelRun: '取消运行失败',
    failedToLoadActivity: '加载最近活动失败',
    failedToLoadDatasets: '加载数据集失败',
    failedToLoadRuns: '加载运行失败',
  },
  artifacts: {
    datasetManifest: '数据集清单',
    loadJob: '加载任务',
    shard: '分片',
    track: '赛道',
    run: '运行',
    report: '报告',
    createdArtifacts: '已创建产物',
    shardTitle: '分片 · {target}',
    runTitle: '运行 · {count} 个模型',
  },
  wizard: {
    eyebrow: '引导流程',
    title: '新建评测',
    subtitle: '将时间序列 CSV 从上传推进到发布评测报告：配置切分、物化评测分片、运行模型适配器并查看结果。',
    progress: '进度 · {done}/{total}',
    footComplete: '当前步骤已完成，请在左侧选择下一步。',
    footIncomplete: '完成当前步骤后可解锁下一步。',
    steps: {
      uploadCsv: {
        title: '上传 CSV',
        kicker: '数据来源',
        description: '选择本地 CSV，并在配置评测前检查识别出的列。',
      },
      configureSplit: {
        title: '配置切分',
        kicker: '数据集清单',
        description: '选择时间列和目标列，并设置 context、horizon 与 stride。',
      },
      confirmShard: {
        title: '确认分片',
        kicker: '样本加载',
        description: '检查已物化的评测分片，确保运行从确定的评测集开始。',
      },
      createTrack: {
        title: '创建赛道',
        kicker: '评测目标',
        description: '将分片绑定到真实数据赛道，并使用 MASE 作为主指标。',
      },
      runModels: {
        title: '运行模型',
        kicker: '执行',
        description: '选择模型适配器并启动运行，进度会持续更新直到完成。',
      },
      openReport: {
        title: '打开报告',
        kicker: '结果',
        description: '运行发布报告标识后，进入生成的报告查看结果。',
      },
    },
    runStep: {
      modelAdapters: '模型适配器',
      selectAll: '全选',
      clearAll: '清空',
      availableModels: '可用模型',
      loadingModels: '正在加载模型适配器...',
      loadingSelectedModels: '正在加载已选 timer-service 模型...',
      percentComplete: '完成 {percent}%',
      modelsProgress: '{done}/{total} 个模型',
      tasksProgress: '{done}/{total} 个任务',
      samplesProgress: '{done}/{total} 个样本',
      openRun: '打开运行',
      run: '运行',
      loaded: '已加载',
      loading: '加载中',
      notLoaded: '未加载',
    },
  },
  home: {
    eyebrow: 'TSBenchmark',
    title: '工作台概览',
    subtitle: '端到端评测时间序列预测模型：上传数据、物化评测分片、运行模型适配器，并查看排名和逐样本结果。',
    startNewEvaluation: '开始新评测',
    datasets: '数据集',
    runs: '运行',
    tracks: '赛道',
    reports: '报告',
    manifestsAndShards: '清单与分片',
    benchmarkingExecutions: '评测执行',
    benchmarkTargets: '评测目标',
    publishedResults: '已发布结果',
    recentActivity: '最近活动',
    noActivity: '暂无活动',
    noActivityDesc: '你创建的产物会出现在这里，便于快速访问。',
    howItWorks: '工作流程',
    beginGuidedRun: '开始引导运行',
    kind: {
      dataset: '数据集',
      shard: '分片',
      run: '运行',
      report: '报告',
    },
    steps: {
      upload: { title: '上传并配置', desc: '选择 CSV、目标列，并设置 context/horizon 切分。' },
      shard: { title: '物化分片', desc: '生成确定性的预测样本评测集。' },
      run: { title: '运行模型适配器', desc: '将分片绑定到赛道，并执行已选择的模型。' },
      review: { title: '查看结果', desc: '比较排名指标，并检查逐样本预测。' },
    },
  },
  datasets: {
    eyebrow: '工作区',
    title: '数据集',
    subtitle: '当前工作区中保存的数据集清单和评测分片。',
    noDatasets: '暂无数据集',
    noDatasetsDesc: '在新建评测中上传 CSV，即可创建第一个数据集清单和分片。',
    uploadCsv: '上传 CSV',
    artifact: '产物',
    type: '类型',
    detail: '详情',
    created: '创建时间',
    rows: '{count} 行',
  },
  runs: {
    eyebrow: '工作区',
    title: '运行',
    subtitle: '当前工作区中启动过的评测运行。打开某次运行可查看实时进度和结果。',
    noRuns: '暂无运行',
    noRunsDesc: '在新建评测中创建赛道并执行模型适配器后，运行会显示在这里。',
    startRun: '启动运行',
    run: '运行',
    lastStatus: '最新状态',
    created: '创建时间',
  },
};

export default zhCN;
```

- [ ] **Step 5: Add i18n instance and runtime setter**

Create `frontend/src/i18n/index.ts`:

```ts
import { createI18n } from 'vue-i18n';
import enUS from './locales/en-US';
import zhCN from './locales/zh-CN';
import { DEFAULT_LOCALE, STORAGE_KEY, type LocaleCode, normalizeLocale, resolveInitialLocale } from './keys';

export const messages = {
  'en-US': enUS,
  'zh-CN': zhCN,
};

export const i18n = createI18n({
  legacy: false,
  locale: resolveInitialLocale(),
  fallbackLocale: DEFAULT_LOCALE,
  messages,
});

export function setDocumentLocale(locale: LocaleCode) {
  document.documentElement.lang = locale;
}

export function setLocale(locale: LocaleCode) {
  i18n.global.locale.value = locale;
  setDocumentLocale(locale);
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch (_error) {
    // Ignore storage failures; the runtime locale still changes for this session.
  }
}

export function setLocaleFromUnknown(value: unknown): LocaleCode {
  const next = normalizeLocale(value) ?? DEFAULT_LOCALE;
  setLocale(next);
  return next;
}

setDocumentLocale(i18n.global.locale.value as LocaleCode);
```

- [ ] **Step 6: Register i18n in the app bootstrap**

Modify `frontend/src/main.ts` so the final bootstrap block is:

```ts
import { createApp } from 'vue';
import App from './App.vue';
import './styles.css';
import { configureAuth } from './api/client';
import { i18n } from './i18n';
import { bootstrap, getToken, logout } from './stores/auth';

// 让 api/client 通过这两个钩子访问 auth store，避免双向 import 循环。
configureAuth({
  getToken,
  onUnauthorized: () => {
    logout();
    const here = window.location.hash.replace(/^#/, '') || '/';
    // 不要从 /login 自己再跳 /login，否则死循环。
    if (!here.startsWith('/login')) {
      const next = encodeURIComponent('#' + here);
      window.location.hash = `/login?next=${next}`;
    }
  }
});

// 启动期用已有 token 拉一次 /auth/me（若失败会自动 logout），完成后再挂载 App，
// 避免首屏在未知态下错误闪烁登录页。
bootstrap().finally(() => {
  createApp(App).use(i18n).mount('#app');
});
```

- [ ] **Step 7: Add i18n tests**

Create `frontend/src/tests/i18n.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { normalizeLocale, readLocaleFromUrl } from '../i18n/keys';
import enUS from '../i18n/locales/en-US';
import zhCN from '../i18n/locales/zh-CN';

function flattenKeys(value: unknown, prefix = ''): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [prefix];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
    flattenKeys(child, prefix ? `${prefix}.${key}` : key)
  );
}

describe('i18n locale catalogs', () => {
  it('keeps English and Chinese key sets aligned', () => {
    expect(flattenKeys(zhCN).sort()).toEqual(flattenKeys(enUS).sort());
  });

  it('does not ship empty translation values', () => {
    for (const catalog of [enUS, zhCN]) {
      const keys = flattenKeys(catalog);
      for (const key of keys) {
        const value = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)[part], catalog);
        expect(typeof value === 'string' && value.length > 0, key).toBe(true);
      }
    }
  });
});

describe('locale resolution', () => {
  it('normalizes supported browser locale variants', () => {
    expect(normalizeLocale('zh')).toBe('zh-CN');
    expect(normalizeLocale('zh-Hans')).toBe('zh-CN');
    expect(normalizeLocale('en')).toBe('en-US');
    expect(normalizeLocale('fr-FR')).toBeNull();
  });

  it('reads lang from page query and hash query', () => {
    expect(readLocaleFromUrl('?lang=zh-CN', '#/runs')).toBe('zh-CN');
    expect(readLocaleFromUrl('', '#/runs?lang=en-US')).toBe('en-US');
    expect(readLocaleFromUrl('', '#/runs')).toBeNull();
  });
});
```

- [ ] **Step 8: Run focused i18n tests**

Run:

```bash
cd frontend && npm test -- src/tests/i18n.test.ts
```

Expected:

- Vitest exits with status 0.
- All tests in `i18n.test.ts` pass.

- [ ] **Step 9: Commit foundation**

Run:

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/main.ts frontend/src/i18n frontend/src/tests/i18n.test.ts
git commit -m "add frontend i18n foundation"
```

Expected:

- Commit contains only the foundation files listed above.

---

## Task 2: Common Components, Formatting, and Error Display

**Files:**

- Modify: `frontend/src/lib/format.ts`
- Create: `frontend/src/composables/useFormat.ts`
- Create: `frontend/src/lib/errors.ts`
- Modify: `frontend/src/components/ui/StateBlock.vue`
- Modify: `frontend/src/components/ui/StatusBadge.vue`
- Test: `frontend/src/tests/format.test.ts`
- Test: `frontend/src/tests/errors.test.ts`
- Modify or create focused component tests for `StateBlock.vue` and `StatusBadge.vue`

- [ ] **Step 1: Add failing formatting tests**

Create `frontend/src/tests/format.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { formatDateTime, formatInt, formatNumber, timeAgo } from '../lib/format';

describe('locale-aware format helpers', () => {
  it('formats numbers with an explicit locale', () => {
    expect(formatInt(1234567, 'en-US')).toContain('1');
    expect(formatInt(1234567, 'zh-CN')).toContain('1');
    expect(formatNumber(0.123456, 2, 'en-US')).toContain('0');
  });

  it('formats dates with an explicit locale', () => {
    expect(formatDateTime('2026-05-26T12:00:00Z', 'en-US')).toMatch(/2026|May|26/);
    expect(formatDateTime('2026-05-26T12:00:00Z', 'zh-CN')).toMatch(/2026|5|26/);
  });

  it('formats relative time without English-only string assembly', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-26T12:00:10Z'));
    expect(timeAgo('2026-05-26T12:00:00Z', 'en-US')).toContain('second');
    expect(timeAgo('2026-05-26T12:00:00Z', 'zh-CN')).toMatch(/秒|10/);
    vi.useRealTimers();
  });
});
```

- [ ] **Step 2: Run formatting tests and confirm failure**

Run:

```bash
cd frontend && npm test -- src/tests/format.test.ts
```

Expected:

- Fails because `formatInt`, `formatDateTime`, and `timeAgo` do not yet accept locale consistently and `timeAgo` still assembles English strings.

- [ ] **Step 3: Update `format.ts`**

Modify `frontend/src/lib/format.ts` to keep existing function names and add locale parameters:

```ts
// Display formatting helpers shared across views. Pure functions, no deps.

/** Format a metric/number compactly with sensible precision. */
export function formatNumber(value: unknown, digits = 4, locale?: string): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 1e6 || abs < 1e-4) return n.toExponential(2);
  if (Number.isInteger(n)) return n.toLocaleString(locale);
  return n.toLocaleString(locale, { maximumFractionDigits: digits });
}

/** Integer with thousands separators. */
export function formatInt(value: unknown, locale?: string): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString(locale) : String(value);
}

/** Absolute timestamp, locale-aware. */
export function formatDateTime(value?: string | null, locale?: string): string {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(locale, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

/** Human relative time string. */
export function timeAgo(value?: string | null, locale?: string): string {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  const seconds = Math.round((d.getTime() - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  const abs = Math.abs(seconds);
  if (abs < 5) return rtf.format(0, 'second');
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['second', 60],
    ['minute', 60],
    ['hour', 24],
    ['day', 7],
    ['week', 4.345],
    ['month', 12],
    ['year', Infinity],
  ];
  let valueInUnit = seconds;
  for (const [unit, step] of units) {
    if (Math.abs(valueInUnit) < step) {
      return rtf.format(Math.round(valueInUnit), unit);
    }
    valueInUnit /= step;
  }
  return formatDateTime(value, locale);
}

/** Compact short id for display (keeps prefix + tail). */
export function shortId(id?: string | null, head = 8): string {
  if (!id) return '—';
  if (id.length <= head + 4) return id;
  return `${id.slice(0, head)}…${id.slice(-4)}`;
}

/** Percentage 0-100 from a 0..1 or completed/total pair. */
export function percent(done: number, total: number): number {
  if (!total || total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
}

/** Title-case a snake/kebab status string for display fallback. */
export function humanize(value?: string | null): string {
  if (!value) return '';
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
```

- [ ] **Step 4: Add locale-aware format composable**

Create `frontend/src/composables/useFormat.ts`:

```ts
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { formatDateTime, formatInt, formatNumber, timeAgo } from '../lib/format';

export function useFormat() {
  const { locale } = useI18n();
  const currentLocale = computed(() => String(locale.value));

  return {
    locale: currentLocale,
    formatNumber: (value: unknown, digits = 4) => formatNumber(value, digits, currentLocale.value),
    formatInt: (value: unknown) => formatInt(value, currentLocale.value),
    formatDateTime: (value?: string | null) => formatDateTime(value, currentLocale.value),
    timeAgo: (value?: string | null) => timeAgo(value, currentLocale.value),
  };
}
```

- [ ] **Step 5: Add error display helper and tests**

Create `frontend/src/lib/errors.ts`:

```ts
import { ApiError } from '../api/client';

export type TranslateFn = (key: string, params?: Record<string, unknown>) => string;
export type TranslationExistsFn = (key: string) => boolean;

export function displayError(
  error: unknown,
  t: TranslateFn,
  te: TranslationExistsFn,
  fallbackKey = 'errors.apiError'
): string {
  if (error instanceof ApiError) {
    const key = `errors.${error.error_code}`;
    if (te(key)) return t(key);
    return error.message || t(fallbackKey);
  }
  if (error instanceof Error && error.message) return error.message;
  return t(fallbackKey);
}
```

Create `frontend/src/tests/errors.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import enUS from '../i18n/locales/en-US';
import zhCN from '../i18n/locales/zh-CN';
import { displayError } from '../lib/errors';

function lookup(catalog: Record<string, unknown>, key: string): string | undefined {
  const value = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], catalog);
  return typeof value === 'string' ? value : undefined;
}

const t = (catalog: Record<string, unknown>) => (key: string) => lookup(catalog, key) ?? key;
const te = (catalog: Record<string, unknown>) => (key: string) => lookup(catalog, key) !== undefined;

describe('displayError', () => {
  it('maps known ApiError codes to the active locale', () => {
    const error = new ApiError('auth_required', 'login required', {}, 401);
    expect(displayError(error, t(enUS), te(enUS))).toBe('Please sign in to continue.');
    expect(displayError(error, t(zhCN), te(zhCN))).toBe('请先登录。');
  });

  it('falls back to backend message for unknown ApiError codes', () => {
    const error = new ApiError('new_backend_code', 'Backend fallback', {}, 400);
    expect(displayError(error, t(enUS), te(enUS))).toBe('Backend fallback');
  });

  it('uses translated fallback for non-error values', () => {
    expect(displayError(null, t(enUS), te(enUS))).toBe('Request failed');
    expect(displayError(null, t(zhCN), te(zhCN))).toBe('请求失败');
  });
});
```

- [ ] **Step 6: Update `StateBlock.vue`**

Modify `frontend/src/components/ui/StateBlock.vue` so text defaults come from i18n:

```vue
<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from './Icon.vue';

const props = withDefaults(defineProps<{
  loading?: boolean;
  error?: string | null;
  empty?: boolean;
  loadingText?: string;
  errorTitle?: string;
  emptyTitle?: string;
  emptyDesc?: string;
  emptyIcon?: string;
}>(), {
  loading: false,
  error: null,
  empty: false,
  emptyDesc: '',
  emptyIcon: 'inbox',
});

const { t } = useI18n();

const resolvedLoadingText = computed(() => props.loadingText ?? t('common.loading'));
const resolvedErrorTitle = computed(() => props.errorTitle ?? t('state.somethingWentWrong'));
const resolvedEmptyTitle = computed(() => props.emptyTitle ?? t('state.nothingHereYet'));

defineEmits<{ retry: [] }>();
</script>
```

Update its template references:

```vue
<p class="state-desc">{{ resolvedLoadingText }}</p>
<p class="state-title">{{ resolvedErrorTitle }}</p>
<button class="btn secondary sm" type="button" @click="$emit('retry')">
  <Icon name="refresh" :size="15" /> {{ t('common.retry') }}
</button>
<p class="state-title">{{ resolvedEmptyTitle }}</p>
```

- [ ] **Step 7: Update `StatusBadge.vue`**

Modify `frontend/src/components/ui/StatusBadge.vue` script:

```ts
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from './Icon.vue';
import { humanize } from '../../lib/format';

const props = withDefaults(defineProps<{ status?: string | null; big?: boolean; label?: string }>(), {
  big: false,
});

const { t, te } = useI18n();

type Variant = 'success' | 'warning' | 'danger' | 'info' | 'primary' | 'neutral';

const MAP: Record<string, { variant: Variant; icon?: string }> = {
  succeeded: { variant: 'success', icon: 'checkCircle' },
  success: { variant: 'success', icon: 'checkCircle' },
  ready: { variant: 'success', icon: 'checkCircle' },
  complete: { variant: 'success', icon: 'checkCircle' },
  completed: { variant: 'success', icon: 'checkCircle' },
  ready_to_load: { variant: 'success', icon: 'check' },
  partial_succeeded: { variant: 'warning', icon: 'alert' },
  running: { variant: 'primary', icon: 'refresh' },
  in_progress: { variant: 'primary', icon: 'refresh' },
  processing: { variant: 'primary', icon: 'refresh' },
  queued: { variant: 'info', icon: 'clock' },
  pending: { variant: 'neutral', icon: 'clock' },
  idle: { variant: 'neutral', icon: 'clock' },
  failed: { variant: 'danger', icon: 'x' },
  error: { variant: 'danger', icon: 'x' },
  cancelled: { variant: 'danger', icon: 'ban' },
  canceled: { variant: 'danger', icon: 'ban' },
};

const normalized = computed(() => (props.status || '').toLowerCase());
const entry = computed(() => MAP[normalized.value] ?? { variant: 'neutral' as Variant, icon: 'info' });
const variant = computed(() => entry.value.variant);
const icon = computed(() => entry.value.icon);
const label = computed(() => {
  if (props.label) return props.label;
  const key = `status.${normalized.value}`;
  if (normalized.value && te(key)) return t(key);
  return humanize(props.status) || '—';
});
```

- [ ] **Step 8: Run common-layer focused tests**

Run:

```bash
cd frontend && npm test -- src/tests/format.test.ts src/tests/errors.test.ts
```

Expected:

- Vitest exits with status 0.
- Formatting and error tests pass.

- [ ] **Step 9: Run full frontend tests after common-layer changes**

Run:

```bash
cd frontend && npm test
```

Expected:

- Existing tests still pass in default English locale.

- [ ] **Step 10: Commit common layer**

Run:

```bash
git add frontend/src/lib/format.ts frontend/src/composables/useFormat.ts frontend/src/lib/errors.ts frontend/src/components/ui/StateBlock.vue frontend/src/components/ui/StatusBadge.vue frontend/src/tests/format.test.ts frontend/src/tests/errors.test.ts
git commit -m "add translated frontend display helpers"
```

Expected:

- Commit contains common display helpers, component updates, and tests.

---

## Task 3: App Shell and Language Switch

**Files:**

- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/tests/AppRoutes.test.ts`

- [ ] **Step 1: Add compact language-switch styles**

Append near the button styles in `frontend/src/styles.css`:

```css
.locale-switch {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  min-height: 32px;
  padding: 2px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.locale-switch button {
  min-width: 42px;
  min-height: 26px;
  padding: 0 8px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font: inherit;
  font-size: 0.78rem;
  font-weight: 700;
}

.locale-switch button:hover {
  color: var(--text);
  background: var(--surface-hover);
}

.locale-switch button.is-active {
  background: var(--primary-soft);
  color: var(--primary-text);
}
```

- [ ] **Step 2: Import i18n helpers in `App.vue`**

In `frontend/src/App.vue`, add imports:

```ts
import { useI18n } from 'vue-i18n';
import { LOCALES, type LocaleCode } from './i18n/keys';
import { setLocale } from './i18n';
```

Add setup state near the theme setup:

```ts
const { t, locale } = useI18n();

const localeOptions: Array<{ code: LocaleCode; labelKey: string; short: string }> = [
  { code: 'en-US', labelKey: 'common.english', short: 'EN' },
  { code: 'zh-CN', labelKey: 'common.chinese', short: '中' },
];

function changeLocale(next: LocaleCode) {
  if ((LOCALES as readonly string[]).includes(next)) setLocale(next);
}
```

- [ ] **Step 3: Replace app-shell literals with `t()`**

In `App.vue`, replace nav groups and buttons:

```vue
<p class="nav-group-label">{{ t('nav.workspace') }}</p>
<p class="nav-group-label" style="margin-top:18px">{{ t('nav.administration') }}</p>
<p class="nav-group-label">{{ t('nav.public') }}</p>
<span v-if="!user.roles.length">{{ t('auth.noRole') }}</span>
<Icon name="logOut" :size="14" /> {{ t('auth.signOut') }}
<Icon name="logIn" :size="14" /> {{ t('auth.signIn') }}
<Icon name="plus" :size="16" /> {{ t('nav.newEvaluation') }}
```

Replace theme label:

```vue
<button
  class="icon-btn"
  type="button"
  :aria-label="t('common.theme', { theme: t(`common.${pref}`) })"
  :title="t('common.theme', { theme: t(`common.${pref}`) })"
  @click="cycleTheme"
>
  <Icon :name="themeIcon" :size="18" />
</button>
```

Add language switch before the theme button:

```vue
<div class="locale-switch" :aria-label="t('common.switchLanguage')">
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
```

- [ ] **Step 4: Convert route labels to computed translations**

Replace `HOME_CRUMB`, `navItems`, `adminItems`, and route crumb literals with translated values. Use this pattern:

```ts
const HOME_CRUMB = computed(() => ({ label: t('nav.overview'), href: '#/' }));

const navItems = computed(() => [
  { key: 'home', label: t('nav.overview'), icon: 'dashboard', href: '#/', count: 0 },
  { key: 'new', label: t('nav.newEvaluation'), icon: 'sparkles', href: '#/new', count: 0 },
  { key: 'datasets', label: t('nav.datasets'), icon: 'database', href: '#/datasets', count: countsState.counts.datasets + countsState.counts.shards },
  { key: 'runs', label: t('nav.runs'), icon: 'activity', href: '#/runs', count: countsState.counts.runs },
  { key: 'leaderboards', label: t('nav.leaderboards'), icon: 'trophy', href: '#/leaderboards', count: 0 }
]);
```

When building crumbs inside `resolveRoute()`, use `HOME_CRUMB.value` and `t(...)`:

```ts
return { component: LoginPage, props: {}, navKey: '', tier: 'public', crumbs: [{ label: t('auth.signIn') }] };
```

Use these translations for recurring crumb labels:

- `t('nav.datasets')`
- `t('nav.runs')`
- `t('nav.report')`
- `t('nav.leaderboards')`
- `t('artifacts.datasetManifest')`
- `t('artifacts.loadJob')`
- `t('artifacts.shard')`
- `t('artifacts.track')`

- [ ] **Step 5: Add app-shell language-switch test**

In `frontend/src/tests/AppRoutes.test.ts`, add:

```ts
it('switches app shell navigation language without changing the route', async () => {
  mockFetch();
  window.location.hash = '#/leaderboards';

  render(App);

  expect(await screen.findByText('Leaderboards')).toBeTruthy();
  await screen.findByRole('button', { name: '中文' }).click();

  expect(await screen.findByText('排行榜')).toBeTruthy();
  expect(window.location.hash).toBe('#/leaderboards');
});
```

If the button accessible name resolves differently because the visual text is `中`, query by title:

```ts
const zhButton = await screen.findByTitle('中文');
await zhButton.click();
```

- [ ] **Step 6: Run app-shell tests**

Run:

```bash
cd frontend && npm test -- src/tests/AppRoutes.test.ts
```

Expected:

- Existing route tests pass.
- New language-switch test passes.

- [ ] **Step 7: Commit app shell**

Run:

```bash
git add frontend/src/App.vue frontend/src/styles.css frontend/src/tests/AppRoutes.test.ts
git commit -m "add frontend language switch"
```

Expected:

- Commit contains only app shell, style, and app route test updates.

---

## Task 4: Guided Wizard Main Path

**Files:**

- Modify: `frontend/src/pages/EvaluationWizardPage.vue`
- Modify: `frontend/src/components/wizard/UploadStep.vue`
- Modify: `frontend/src/components/wizard/ColumnAndSplitStep.vue`
- Modify: `frontend/src/components/wizard/LoadShardStep.vue`
- Modify: `frontend/src/components/wizard/TrackStep.vue`
- Modify: `frontend/src/components/wizard/RunStep.vue`
- Modify: `frontend/src/components/wizard/ResultStep.vue`
- Modify: `frontend/src/tests/e2e-smoke.test.ts`
- Modify: focused wizard tests in `frontend/src/tests/*Step.test.ts`

- [ ] **Step 1: Convert wizard page headings and step definitions**

In `EvaluationWizardPage.vue`, import:

```ts
import { useI18n } from 'vue-i18n';
```

Add:

```ts
const { t } = useI18n();
```

Replace `stepDefs` with a computed:

```ts
const stepDefs = computed(() => [
  { title: t('wizard.steps.uploadCsv.title'), kicker: t('wizard.steps.uploadCsv.kicker'), description: t('wizard.steps.uploadCsv.description'), component: UploadStep, complete: () => Boolean(wizardState.preview) },
  { title: t('wizard.steps.configureSplit.title'), kicker: t('wizard.steps.configureSplit.kicker'), description: t('wizard.steps.configureSplit.description'), component: ColumnAndSplitStep, complete: () => Boolean(wizardState.shardId) },
  { title: t('wizard.steps.confirmShard.title'), kicker: t('wizard.steps.confirmShard.kicker'), description: t('wizard.steps.confirmShard.description'), component: LoadShardStep, complete: () => Boolean(wizardState.shardId) },
  { title: t('wizard.steps.createTrack.title'), kicker: t('wizard.steps.createTrack.kicker'), description: t('wizard.steps.createTrack.description'), component: TrackStep, complete: () => Boolean(wizardState.trackId) },
  { title: t('wizard.steps.runModels.title'), kicker: t('wizard.steps.runModels.kicker'), description: t('wizard.steps.runModels.description'), component: RunStep, complete: () => Boolean(wizardState.reportId) },
  { title: t('wizard.steps.openReport.title'), kicker: t('wizard.steps.openReport.kicker'), description: t('wizard.steps.openReport.description'), component: ResultStep, complete: () => Boolean(wizardState.reportId) }
]);
```

Update code that maps `stepDefs`:

```ts
const steps = computed(() => {
  const defs = stepDefs.value;
  const flags = defs.map((s) => s.complete());
  return defs.map((s, i) => {
    const complete = flags[i];
    const reachable = i === 0 || flags[i - 1];
    const stateLabel = complete ? t('common.done') : i === current.value ? t('common.current') : reachable ? t('common.ready') : t('common.locked');
    return { ...s, complete, reachable, stateLabel };
  });
});
```

Update template literals:

```vue
<p class="eyebrow">{{ t('wizard.eyebrow') }}</p>
<h1>{{ t('wizard.title') }}</h1>
<p class="page-sub">{{ t('wizard.subtitle') }}</p>
<Icon name="refresh" :size="15" /> {{ t('common.reset') }}
<p class="nav-group-label" style="margin:0 0 8px">{{ t('wizard.progress', { done: completedCount, total: steps.length }) }}</p>
<p class="nav-group-label" style="margin:0 0 10px">{{ t('artifacts.createdArtifacts') }}</p>
<StatusBadge :status="active.complete ? 'complete' : 'pending'" :label="active.complete ? t('common.done') : t('common.inProgress')" />
<Icon name="chevronLeft" :size="16" /> {{ t('common.back') }}
```

Update footer hint:

```ts
const footHint = computed(() => active.value.complete ? t('wizard.footComplete') : t('wizard.footIncomplete'));
```

Update artifacts:

```ts
if (wizardState.manifestId) out.push({ name: t('artifacts.datasetManifest'), href: `#/datasets/${wizardState.manifestId}`, icon: 'database' });
if (wizardState.loadJobId) out.push({ name: t('artifacts.loadJob'), href: `#/load-jobs/${wizardState.loadJobId}`, icon: 'file' });
if (wizardState.shardId) out.push({ name: t('artifacts.shard'), href: `#/shards/${wizardState.shardId}`, icon: 'layers' });
if (wizardState.trackId) out.push({ name: t('artifacts.track'), href: `#/tracks/${wizardState.trackId}`, icon: 'target' });
if (wizardState.runId) out.push({ name: t('artifacts.run'), href: `#/runs/${wizardState.runId}`, icon: 'activity' });
if (wizardState.reportId) out.push({ name: t('artifacts.report'), href: `#/reports/${wizardState.reportId}`, icon: 'barChart' });
```

- [ ] **Step 2: Convert `RunStep.vue`**

Import:

```ts
import { useI18n } from 'vue-i18n';
import { displayError } from '../../lib/errors';
```

Add:

```ts
const { t, te } = useI18n();
```

Replace visible literals:

```vue
<span class="label">{{ t('wizard.runStep.modelAdapters') }}</span>
{{ allSelected ? t('wizard.runStep.clearAll') : t('wizard.runStep.selectAll') }}
<div v-if="models.length" class="choice-grid" :aria-label="t('wizard.runStep.availableModels')">
<p v-else class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.runStep.loadingModels') }}</p>
<p v-if="isPreparingModels" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.runStep.loadingSelectedModels') }}</p>
<span class="status-line">{{ t('wizard.runStep.percentComplete', { percent: runningPct }) }}</span>
<span class="badge"><Icon name="layers" :size="13" />{{ t('wizard.runStep.modelsProgress', { done: progress.progress.completed_models ?? 0, total: progress.progress.total_models ?? selectedIds.length }) }}</span>
<span class="badge"><Icon name="list" :size="13" />{{ t('wizard.runStep.tasksProgress', { done: progress.progress.completed_tasks ?? 0, total: progress.progress.total_tasks ?? 0 }) }}</span>
<span class="badge"><Icon name="gauge" :size="13" />{{ t('wizard.runStep.samplesProgress', { done: progress.progress.completed_samples ?? 0, total: progress.progress.total_samples ?? 0 }) }}</span>
<a v-if="runId" class="btn secondary sm" :href="`#/runs/${runId}`"><Icon name="external" :size="15" /> {{ t('wizard.runStep.openRun') }}</a>
<button v-if="isRunning" class="btn danger sm" type="button" @click="onCancel"><Icon name="ban" :size="15" /> {{ t('common.cancel') }}</button>
<span v-if="isPreparingModels || isRunning" class="spinner" /> <Icon v-else name="play" :size="16" /> {{ t('wizard.runStep.run') }}
```

Replace error assignments:

```ts
error.value = displayError(e, t, te, 'errors.failedToLoadModels');
error.value = displayError(e, t, te, 'errors.failedToStartRun');
error.value = displayError(e, t, te, 'errors.failedToReadProgress');
error.value = displayError(e, t, te, 'errors.failedToCancelRun');
```

Replace `modelStateLabel`:

```ts
function modelStateLabel(model: ModelDTO) {
  if (model.loaded === true) return ` · ${t('wizard.runStep.loaded')}`;
  if (model.loading === true) return ` · ${t('wizard.runStep.loading')}`;
  if (model.loaded === false) return ` · ${t('wizard.runStep.notLoaded')}`;
  return '';
}
```

- [ ] **Step 3: Convert the remaining wizard components**

For each of `UploadStep.vue`, `ColumnAndSplitStep.vue`, `LoadShardStep.vue`, `TrackStep.vue`, and `ResultStep.vue`:

1. Import `useI18n`.
2. Add `const { t, te } = useI18n();`.
3. Replace user-visible literal text with existing or new keys under `wizard.<componentArea>`.
4. Replace API catch fallbacks with `displayError(e, t, te, '<specific errors key>')`.
5. Add the new keys to both locale files in the same task.

Use these key namespaces:

```ts
wizard.uploadStep
wizard.columnAndSplitStep
wizard.loadShardStep
wizard.trackStep
wizard.resultStep
```

For each component, keep API DTO fields and CSV column names unmodified.

- [ ] **Step 4: Update wizard tests**

Keep default-locale assertions in English where possible. Add one focused Chinese test in `frontend/src/tests/AppRoutes.test.ts` or a wizard-specific test:

```ts
it('renders wizard step names in Chinese after switching locale', async () => {
  resetWizard();
  render(EvaluationWizardPage);

  expect(screen.getByText('Upload CSV')).toBeTruthy();
  setLocale('zh-CN');

  expect(await screen.findByText('上传 CSV')).toBeTruthy();
});
```

Import `setLocale` in that test:

```ts
import { setLocale } from '../i18n';
```

Reset locale in `beforeEach`:

```ts
setLocale('en-US');
```

- [ ] **Step 5: Run wizard tests**

Run:

```bash
cd frontend && npm test -- src/tests/UploadStep.test.ts src/tests/ColumnAndSplitStep.test.ts src/tests/LoadShardStep.test.ts src/tests/RunStep.test.ts src/tests/e2e-smoke.test.ts
```

Expected:

- Existing English tests pass.
- New Chinese wizard test passes.

- [ ] **Step 6: Commit wizard migration**

Run:

```bash
git add frontend/src/i18n/locales frontend/src/pages/EvaluationWizardPage.vue frontend/src/components/wizard frontend/src/tests
git commit -m "translate evaluation wizard"
```

Expected:

- Commit contains wizard-facing translation keys, wizard components, and relevant tests.

---

## Task 5: Workspace Pages

**Files:**

- Modify: `frontend/src/pages/HomePage.vue`
- Modify: `frontend/src/pages/DatasetsPage.vue`
- Modify: `frontend/src/pages/RunsPage.vue`
- Modify: `frontend/src/pages/RunDetailPage.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Modify: `frontend/src/tests/WorkspacePages.test.ts`

- [ ] **Step 1: Convert `HomePage.vue`**

Import:

```ts
import { useI18n } from 'vue-i18n';
import { useFormat } from '../composables/useFormat';
import { displayError } from '../lib/errors';
```

Add:

```ts
const { t, te } = useI18n();
const { timeAgo } = useFormat();
```

Replace `steps` with:

```ts
const steps = computed(() => [
  { t: t('home.steps.upload.title'), d: t('home.steps.upload.desc') },
  { t: t('home.steps.shard.title'), d: t('home.steps.shard.desc') },
  { t: t('home.steps.run.title'), d: t('home.steps.run.desc') },
  { t: t('home.steps.review.title'), d: t('home.steps.review.desc') }
]);
```

Replace recent activity titles:

```ts
merged.push({ kind: 'shard', id: sh.shard_id, title: t('artifacts.shardTitle', { target: sh.target_columns?.[0] ?? 'target' }), href: `#/shards/${sh.shard_id}`, createdAt: sh.created_at ?? '' });
merged.push({ kind: 'run', id: run.benchmarking_run_id, title: t('artifacts.runTitle', { count: run.model_count || run.model_ids?.length || 0 }), href: `#/runs/${run.benchmarking_run_id}`, createdAt: run.created_at ?? '' });
merged.push({ kind: 'report', id: rep.report_id, title: t('artifacts.report'), href: `#/reports/${rep.report_id}`, createdAt: rep.created_at ?? '' });
```

Replace `humanize(item.kind)` in the template with:

```vue
{{ t(`home.kind.${item.kind}`) }} · {{ timeAgo(item.createdAt) }}
```

Replace catch fallback:

```ts
activityError.value = displayError(e, t, te, 'errors.failedToLoadActivity');
```

Replace visible template literals with keys already present in Task 1 under `home.*`.

- [ ] **Step 2: Convert `DatasetsPage.vue`**

Import:

```ts
import { useI18n } from 'vue-i18n';
import { useFormat } from '../composables/useFormat';
import { displayError } from '../lib/errors';
```

Add:

```ts
const { t, te } = useI18n();
const { formatDateTime, timeAgo } = useFormat();
```

Replace `shardSubtitle`:

```ts
function shardSubtitle(targetColumns: string[], rowCount: number): string {
  const cols = targetColumns.length ? targetColumns.join(', ') : 'shard';
  return rowCount ? `${cols} · ${t('datasets.rows', { count: rowCount })}` : cols;
}
```

Replace row title for shards:

```ts
title: t('artifacts.shardTitle', { target: s.target_columns?.[0] ?? 'target' }),
```

Replace catch fallback:

```ts
error.value = displayError(e, t, te, 'errors.failedToLoadDatasets');
```

Replace visible template literals with `datasets.*`, `nav.newEvaluation`, and `artifacts.*` keys.

- [ ] **Step 3: Convert `RunsPage.vue`**

Import:

```ts
import { useI18n } from 'vue-i18n';
import { useFormat } from '../composables/useFormat';
import { displayError } from '../lib/errors';
```

Add:

```ts
const { t, te } = useI18n();
const { formatDateTime, timeAgo } = useFormat();
```

Replace run title in the template:

```vue
{{ t('artifacts.runTitle', { count: run.model_count || run.model_ids?.length || 0 }) }}
```

Replace catch fallback:

```ts
error.value = displayError(e, t, te, 'errors.failedToLoadRuns');
```

Replace visible template literals with `runs.*`, `nav.newEvaluation`, and `common.open`.

- [ ] **Step 4: Convert `RunDetailPage.vue`**

Use the same imports as `RunsPage.vue`. Add missing locale keys under `runs.detail.*` for:

- Benchmarking run
- Live progress and execution units for this run.
- Models
- Tasks
- Unit
- Capability
- Samples
- Cancel run
- Recent events

Keep `recent_events[].message` as backend/event data and do not translate it.

- [ ] **Step 5: Update workspace tests**

In `frontend/src/tests/WorkspacePages.test.ts`:

- Keep English assertions for default locale.
- Add one Chinese smoke assertion after `setLocale('zh-CN')`, then reset to `en-US`.

Use:

```ts
import { setLocale } from '../i18n';
```

Add in `beforeEach`:

```ts
setLocale('en-US');
```

- [ ] **Step 6: Run workspace tests**

Run:

```bash
cd frontend && npm test -- src/tests/WorkspacePages.test.ts
```

Expected:

- Workspace page tests pass.

- [ ] **Step 7: Commit workspace pages**

Run:

```bash
git add frontend/src/i18n/locales frontend/src/pages/HomePage.vue frontend/src/pages/DatasetsPage.vue frontend/src/pages/RunsPage.vue frontend/src/pages/RunDetailPage.vue frontend/src/tests/WorkspacePages.test.ts
git commit -m "translate workspace pages"
```

Expected:

- Commit contains workspace page translations and tests.

---

## Task 6: Results, Details, Auth, and Admin Pages

**Files:**

- Modify: `frontend/src/pages/LeaderboardsPage.vue`
- Modify: `frontend/src/pages/RankingPage.vue`
- Modify: `frontend/src/pages/ReportPage.vue`
- Modify: `frontend/src/pages/TrackPage.vue`
- Modify: `frontend/src/pages/SampleForecastPage.vue`
- Modify: `frontend/src/pages/LoginPage.vue`
- Modify: `frontend/src/pages/ForbiddenPage.vue`
- Modify: `frontend/src/pages/LoadJobPage.vue`
- Modify: `frontend/src/pages/DatasetManifestPage.vue`
- Modify: `frontend/src/pages/ShardPage.vue`
- Modify: `frontend/src/pages/admin/UsersPage.vue`
- Modify: `frontend/src/pages/admin/RolesPage.vue`
- Modify: `frontend/src/pages/admin/ProfilePage.vue`
- Modify: `frontend/src/components/results/*.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Modify: relevant tests in `frontend/src/tests/*.test.ts`

- [ ] **Step 1: Add locale keys for remaining pages**

Add these key groups to both `frontend/src/i18n/locales/en-US.ts` and `frontend/src/i18n/locales/zh-CN.ts`. The English values below are the current UI copy. Use the Chinese values shown in the right column.

| Key | en-US | zh-CN |
|---|---|---|
| `leaderboards.eyebrow` | Results | 结果 |
| `leaderboards.title` | Leaderboards | 排行榜 |
| `leaderboards.subtitle` | Compare model performance across all benchmark tracks. | 比较所有评测赛道上的模型表现。 |
| `leaderboards.search` | Search | 搜索 |
| `leaderboards.searchPlaceholder` | Filter by track name... | 按赛道名称筛选... |
| `leaderboards.trackType` | Track type | 赛道类型 |
| `leaderboards.allTypes` | All types | 全部类型 |
| `leaderboards.boardCount` | {count} boards | {count} 个榜单 |
| `leaderboards.noLeaderboards` | No leaderboards yet | 暂无排行榜 |
| `leaderboards.noLeaderboardsDesc` | Run models on a track to populate one. | 在赛道上运行模型后会生成排行榜。 |
| `ranking.eyebrow` | Results | 结果 |
| `ranking.title` | Track ranking | 赛道排名 |
| `ranking.subtitle` | Compare model performance on the selected benchmark track. | 比较所选评测赛道上的模型表现。 |
| `ranking.trackDetail` | Track detail | 赛道详情 |
| `ranking.metric` | Metric | 指标 |
| `ranking.policy` | Policy | 策略 |
| `ranking.modelsRanked` | {count} models ranked | {count} 个模型已排名 |
| `ranking.noRanking` | No ranking yet | 暂无排名 |
| `ranking.noRankingDesc` | Run models on this track to populate the leaderboard. | 在此赛道上运行模型后会生成排行榜。 |
| `report.eyebrow` | Report | 报告 |
| `report.title` | Benchmark report | 评测报告 |
| `report.subtitle` | Model metrics, task outcomes, and per-sample forecast links for this run. | 本次运行的模型指标、任务结果和逐样本预测链接。 |
| `track.eyebrow` | Track | 赛道 |
| `track.title` | Track detail | 赛道详情 |
| `track.subtitle` | Landing page for a benchmark track, with ranking controls and links into result artifacts. | 评测赛道详情页，包含排名控件和结果产物链接。 |
| `track.standaloneRanking` | Standalone ranking | 独立排名 |
| `track.metadata` | Track metadata | 赛道元数据 |
| `track.trackId` | Track ID | 赛道 ID |
| `track.rankingRoute` | Ranking route | 排名路由 |
| `track.openStandaloneRanking` | Open standalone ranking | 打开独立排名 |
| `track.ranking` | Ranking | 排名 |
| `sampleForecast.eyebrow` | Sample forecast | 样本预测 |
| `sampleForecast.title` | Forecast detail | 预测详情 |
| `sampleForecast.failedModels` | Failed model outputs | 失败的模型输出 |
| `sampleForecast.history` | History | 历史 |
| `sampleForecast.future` | Future | 未来 |
| `loadJob.eyebrow` | Dataset load | 数据集加载 |
| `loadJob.title` | Dataset load job | 数据集加载任务 |
| `loadJob.status` | Status | 状态 |
| `loadJob.error` | Error | 错误 |
| `datasetManifest.eyebrow` | Dataset | 数据集 |
| `datasetManifest.title` | Dataset manifest | 数据集清单 |
| `datasetManifest.status` | Status | 状态 |
| `datasetManifest.sourceUri` | Source URI | 来源 URI |
| `datasetManifest.valueColumns` | Value columns | 数值列 |
| `shard.eyebrow` | Shard | 分片 |
| `shard.title` | Shard detail | 分片详情 |
| `shard.status` | Status | 状态 |
| `shard.samples` | Samples | 样本 |
| `shard.sample` | Sample | 样本 |
| `shard.index` | Index | 索引 |
| `shard.context` | Context | 上下文 |
| `shard.horizon` | Horizon | 预测步长 |
| `shard.noSampleIndex` | No sample index loaded. | 未加载样本索引。 |
| `login.title` | Sign in | 登录 |
| `login.username` | Username | 用户名 |
| `login.password` | Password | 密码 |
| `login.invalidCredentials` | Invalid username or password. | 用户名或密码无效。 |
| `forbidden.title` | Forbidden | 无权访问 |
| `forbidden.subtitle` | You do not have permission to view this page. | 你没有权限查看此页面。 |
| `admin.tabsLabel` | Administration sections | 管理区导航 |
| `admin.users.title` | Users | 用户 |
| `admin.users.newUser` | New user | 新建用户 |
| `admin.users.createUser` | Create user | 创建用户 |
| `admin.users.close` | Close | 关闭 |
| `admin.users.username` | Username | 用户名 |
| `admin.users.email` | Email | 邮箱 |
| `admin.users.password` | Password | 密码 |
| `admin.users.role` | Role | 角色 |
| `admin.users.status` | Status | 状态 |
| `admin.users.active` | Active | 启用 |
| `admin.users.disabled` | Disabled | 禁用 |
| `admin.users.actions` | Actions | 操作 |
| `admin.users.noUsers` | No users | 暂无用户 |
| `admin.users.noUsersDesc` | Use "New user" to add the first account. | 使用“新建用户”添加第一个账号。 |
| `admin.users.minimumPassword` | Minimum 6 characters. | 至少 6 个字符。 |
| `admin.roles.title` | Roles | 角色 |
| `admin.roles.noRoles` | No roles | 暂无角色 |
| `admin.roles.permissionsFor` | Permissions for "{name}" | “{name}”的权限 |
| `admin.roles.systemRole` | System role - managed by the platform, not editable here. | 系统角色由平台管理，不能在这里编辑。 |
| `admin.roles.noPermissions` | No permissions assigned. | 未分配权限。 |
| `admin.profile.title` | My profile | 我的资料 |
| `admin.profile.account` | Account | 账号 |
| `admin.profile.accountDetails` | Account details | 账号详情 |
| `admin.profile.privileges` | Privileges | 特权 |
| `admin.profile.superuser` | Superuser | 超级用户 |
| `admin.profile.changePassword` | Change password | 修改密码 |
| `admin.profile.currentPassword` | Current password | 当前密码 |
| `admin.profile.newPassword` | New password | 新密码 |
| `admin.profile.confirmPassword` | Confirm new password | 确认新密码 |
| `admin.profile.updatePassword` | Update password | 更新密码 |
| `admin.profile.updating` | Updating... | 更新中... |
| `results.lowerIsBetter` | Lower is better | 越低越好 |
| `results.rank` | Rank | 排名 |
| `results.model` | Model | 模型 |
| `results.relative` | Relative | 相对值 |
| `results.modelMetrics` | Model metrics | 模型指标 |
| `results.noModelMetrics` | No model metrics recorded. | 暂无模型指标记录。 |
| `results.bestHighlighted` | Lower is better · best per metric highlighted | 越低越好 · 每个指标的最佳值已高亮 |
| `results.taskOutcomes` | Task outcomes | 任务结果 |
| `results.noTasks` | No tasks recorded. | 暂无任务记录。 |
| `results.sampleForecasts` | Sample forecasts | 样本预测 |
| `results.noSampleForecasts` | No per-sample forecasts available. | 暂无逐样本预测。 |
| `results.sampleForecastLinks` | Sample forecast links | 样本预测链接 |
| `results.perModelMetrics` | Per-model metrics for this sample · best highlighted | 此样本的逐模型指标 · 最佳值已高亮 |
| `results.status` | Status | 状态 |
| `results.forecastVsTruth` | Forecast vs. ground truth | 预测与真实值对比 |
| `results.dimension` | Dimension | 维度 |
| `results.targetDimension` | Target dimension | 目标维度 |
| `results.noNumericSeries` | No numeric series to plot for this sample. | 此样本没有可绘制的数值序列。 |
| `results.seriesLegend` | Series legend | 序列图例 |
| `results.viewFullBoard` | View full board | 查看完整榜单 |
| `results.noRankedResults` | No ranked results yet | 暂无排名结果 |

- [ ] **Step 2: Migrate the listed pages**

For each page file in this task:

1. Import `useI18n`.
2. Import `useFormat` if the page calls `formatDateTime`, `formatInt`, `formatNumber`, or `timeAgo`.
3. Import `displayError` if the page assigns catch fallback text.
4. Replace visible template text with the key listed in Step 1.
5. Replace string-built display titles with translation params, for example `t('artifacts.runTitle', { count })`.
6. Keep DTO enum values, ids, metric keys, and server event messages unchanged.
7. Run the focused test for that page before moving to the next page.

Example catch replacement:

```ts
catch (e) {
  error.value = displayError(e, t, te, 'errors.failedToLoadRuns');
}
```

Example date replacement:

```ts
const { formatDateTime, timeAgo } = useFormat();
```

Template:

```vue
<td class="muted nowrap" :title="item.created_at ? formatDateTime(item.created_at) : ''">
  {{ item.created_at ? timeAgo(item.created_at) : '—' }}
</td>
```

- [ ] **Step 3: Migrate result components**

For `frontend/src/components/results/*.vue`:

1. Translate table headers, chart labels, empty helper text, and section titles using the `results.*` keys from Step 1.
2. Keep model names, metric keys, and numeric values unchanged.
3. Use `StatusBadge` for statuses where possible.
4. Use `useFormat` for numbers and dates when displayed.

- [ ] **Step 4: Update focused tests**

Run and update tests after each group:

```bash
cd frontend && npm test -- src/tests/RankingPage.test.ts
cd frontend && npm test -- src/tests/ReportPage.test.ts
cd frontend && npm test -- src/tests/SampleForecastPage.test.ts
cd frontend && npm test -- src/tests/LoadShardStep.test.ts
```

Expected:

- Tests pass in default English.
- Any new Chinese assertions pass after `setLocale('zh-CN')`.

- [ ] **Step 5: Commit remaining page migration**

Run:

```bash
git add frontend/src/i18n/locales frontend/src/pages frontend/src/components/results frontend/src/tests
git commit -m "translate remaining frontend pages"
```

Expected:

- Commit contains remaining page translations and test updates.

---

## Task 7: Final Sweep and Verification

**Files:**

- Modify only files where the sweep finds user-visible hardcoded text still requiring translation.

- [ ] **Step 1: Scan for likely remaining user-visible English**

Run:

```bash
rg -n "'[A-Z][^']*'|\"[A-Z][^\"]*\"|>[A-Z][^<]+<" frontend/src --glob '!**/tests/**'
```

Expected:

- Remaining matches are reviewed manually.
- API paths, TypeScript type names, icon names, component names, comments, test fixtures, and DTO enum values may remain.
- User-visible literals in templates and fallback errors are either translated or intentionally left because they are runtime data from the backend.

- [ ] **Step 2: Scan accessibility text**

Run:

```bash
rg -n "aria-label|title=|placeholder=" frontend/src --glob '!**/tests/**'
```

Expected:

- `aria-label`, `title`, and `placeholder` values are translated unless they display runtime data or ids.

- [ ] **Step 3: Verify locale key shape one final time**

Run:

```bash
cd frontend && npm test -- src/tests/i18n.test.ts
```

Expected:

- Locale key sets match.
- No empty translation strings are present.

- [ ] **Step 4: Run full frontend unit suite**

Run:

```bash
cd frontend && npm test
```

Expected:

- Vitest exits with status 0.
- All frontend tests pass.

- [ ] **Step 5: Run frontend smoke test**

Run:

```bash
cd frontend && npm run test:e2e
```

Expected:

- Smoke test exits with status 0.

- [ ] **Step 6: Commit final sweep**

If Step 1 or Step 2 required code changes, run:

```bash
git add frontend/src
git commit -m "finish frontend bilingual sweep"
```

Expected:

- Commit contains only final translation cleanup.

If Step 1 and Step 2 required no code changes, record the scan commands and test commands in the PR summary instead of creating an empty commit.

---

## Execution Notes

- Keep `en-US` as the default locale in tests to avoid broad assertion churn.
- Add Chinese assertions only for language switching and critical shared components.
- Use `t()` in templates and computed values. Avoid translating inside API client functions.
- Prefer translation parameters over string concatenation for user-visible phrases.
- Keep backend error `message` as a fallback only.
- If a new backend `error_code` is discovered during migration, add `errors.<error_code>` to both locale files in the same commit that surfaces it.
- Do not modify `scripts/start-system.sh`; it had unrelated local changes when this plan was written.

---

## Self-Review

Spec coverage:

- Frontend-only scope: covered by all tasks and execution notes.
- `vue-i18n` foundation: Task 1.
- Locale choice, localStorage, and `<html lang>`: Task 1.
- Shared state/status text: Task 2.
- Date, number, and relative-time formatting: Task 2.
- API error display from `error_code`: Task 2.
- App shell language switch, nav, breadcrumbs, and theme labels: Task 3.
- Wizard main flow: Task 4.
- Workspace pages: Task 5.
- Result/auth/admin/details pages: Task 6.
- Hardcoded text and accessibility sweep: Task 7.
- Tests and verification commands: each task includes focused commands; Task 7 includes full suite and smoke.

Each task has a concrete file set, commands, expected results, and commit point.
