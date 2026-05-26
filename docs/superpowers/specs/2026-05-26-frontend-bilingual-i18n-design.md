# TSBenchmark 前端中英双语改造设计

**日期：** 2026-05-26

**性质：** 前端显示层 i18n 改造方案。本文只定义架构、迁移步骤和验收标准，方便后续按文档实施；本轮不修改业务代码。

**结论：** 采用前端主导的双语方案。后端继续返回稳定结构化数据、错误码和英文 fallback message；用户可见文案统一在 Vue 前端翻译。

---

## 1. 目标与边界

### 目标

- 支持 `en-US` 与 `zh-CN` 两套前端显示文案。
- 用户可在界面内切换语言，选择持久化到本地。
- 导航、面包屑、wizard 主流程、公共组件、状态、错误提示、时间/数字格式跟随当前语言。
- 尽量保持现有自研 hash routing、组件结构和 API DTO 不变。

### 不做

- 不做后端按 `Accept-Language` 返回双语 message。
- 不翻译数据库字段、API 枚举值、权限 code、模型 id、数据集 id、CSV 原始列名、metric key。
- 不引入 vue-router 或重构整体页面结构。
- 不做服务端渲染、按语言拆包、在线翻译或 CMS 化文案管理。

---

## 2. 现状判断

前端当前文案主要是硬编码英文，集中和散落并存：

- `frontend/src/App.vue`：侧边栏、顶部按钮、面包屑、路由标题。
- `frontend/src/components/ui/StateBlock.vue`：loading、empty、error、retry 默认文案。
- `frontend/src/components/ui/StatusBadge.vue`：状态展示依赖 `humanize(status)`。
- `frontend/src/lib/format.ts`：日期、数字、相对时间、状态 humanize。
- `frontend/src/pages/*` 与 `frontend/src/components/wizard/*`：页面标题、说明、按钮、错误 fallback。
- `frontend/src/api/client.ts`：`ApiError` 使用后端 message 作为展示 fallback。

后端已经具备适合前端翻译的错误结构：

```json
{
  "error_code": "auth_required",
  "message": "login required",
  "details": {}
}
```

因此双语改造应把后端 `message` 视为调试/fallback，不作为主展示文案。

---

## 3. 推荐架构

### 3.1 技术选型

新增依赖：

```bash
cd frontend && npm install vue-i18n
```

理由：

- Vue 3 生态成熟，适合当前 Vite + Vue 组合。
- 支持 Composition API。
- 支持参数插值、复数、fallback locale。
- 比手写 `t()` 更适合后续规模扩大和测试。

### 3.2 目录结构

新增：

```text
frontend/src/i18n/
  index.ts
  locales/
    en-US.ts
    zh-CN.ts
  keys.ts
```

职责：

- `index.ts`：创建 i18n 实例、读取/保存 locale、暴露语言切换 helper。
- `locales/en-US.ts`：英文文案源。
- `locales/zh-CN.ts`：中文文案源。
- `keys.ts`：定义 `LocaleCode`、可选的 message 类型、工具函数。

`main.ts` 中注册 i18n：

```ts
app.use(i18n);
```

组件内使用：

```ts
const { t, locale } = useI18n();
```

模板内使用：

```vue
{{ t('nav.runs') }}
```

### 3.3 locale 选择规则

启动时按以下优先级决定当前语言：

1. URL query：`?lang=zh-CN` 或 `?lang=en-US`
2. `localStorage['tsbenchmark.locale']`
3. `navigator.language`
4. 默认 `en-US`

只支持 `en-US`、`zh-CN`。浏览器语言为 `zh`、`zh-CN`、`zh-Hans` 时归一到 `zh-CN`；其他语言归一到 `en-US`。

用户在界面切换语言时：

- 更新 i18n locale。
- 写入 localStorage。
- 同步 `<html lang="...">`。
- 不强制刷新页面。

URL query 只用于初始覆盖，不要求每次切换都改写 hash 或 query，避免影响现有 hash routing。

### 3.4 文案 key 分层

按产品域组织，不按文件机械切分：

```ts
export default {
  common: {
    loading: 'Loading...',
    retry: 'Try again',
    cancel: 'Cancel',
    open: 'Open',
  },
  nav: {
    workspace: 'Workspace',
    administration: 'Administration',
    overview: 'Overview',
    newEvaluation: 'New evaluation',
    datasets: 'Datasets',
    runs: 'Runs',
    leaderboards: 'Leaderboards',
  },
  auth: {
    signIn: 'Sign in',
    signOut: 'Sign out',
    noRole: 'no role',
  },
  wizard: {
    uploadCsv: {
      title: 'Upload CSV',
      kicker: 'Data source',
      description: 'Pick a local CSV and inspect detected columns before configuring a benchmark.',
    },
    runModels: {
      title: 'Run models',
      kicker: 'Execution',
      description: 'Select model adapters and start the run; progress updates until completion.',
    },
  },
  status: {
    queued: 'Queued',
    running: 'Running',
    succeeded: 'Succeeded',
    failed: 'Failed',
    cancelled: 'Cancelled',
  },
  errors: {
    apiError: 'Request failed',
    authRequired: 'Please sign in to continue.',
    forbidden: 'You do not have permission to perform this action.',
  },
};
```

中文包保持同构 key。后续新增 key 必须同时补齐两种语言。

---

## 4. 改造设计

### 4.1 应用外壳与语言切换

`App.vue` 增加语言切换入口，建议放在顶部右侧 actions 内，与主题切换同级。

交互建议：

- 用一个小型按钮或 segmented control 显示 `EN` / `中文`。
- 当前语言清晰可见。
- 点击后立即切换，无需刷新。
- `aria-label`、`title` 也使用 i18n 文案。

需要迁移的 `App.vue` 文案：

- 侧边栏分组：Workspace、Administration、Public。
- 导航项：Overview、New evaluation、Datasets、Runs、Leaderboards。
- admin 项：Users、Roles、My profile。
- 登录/登出按钮。
- 顶部按钮：New evaluation。
- 主题按钮 aria-label/title。
- 面包屑文案。
- 登录重定向相关可见文案。

### 4.2 公共 UI 组件

`StateBlock.vue`：

- 默认 `loadingText`、`errorTitle`、`emptyTitle`、retry 按钮改为 i18n。
- 继续允许调用方传入 props 覆盖。
- 对于调用方传入的 `emptyDesc`，第一阶段仍可传普通字符串；迁移页面时逐步改成 `t(...)`。

`StatusBadge.vue`：

- 状态 code 保持原值。
- 展示文案优先使用显式 `label` prop。
- 没有 `label` 时查 `status.<normalizedStatus>`。
- 查不到 key 时再 fallback 到 `humanize(status)`。

### 4.3 格式化工具

`format.ts` 调整为 locale-aware：

- `formatNumber(value, digits, locale?)`
- `formatInt(value, locale?)`
- `formatDateTime(value, locale?)`
- `timeAgo(value, locale?)`

相对时间建议使用 `Intl.RelativeTimeFormat`，避免英文硬编码 `just now`、`seconds ago`。

组件调用方式有两种：

- 简单页面：组件内 `const { locale } = useI18n()`，调用 `formatDateTime(value, locale.value)`。
- 频繁使用场景：新增 `frontend/src/composables/useFormat.ts`，集中包装当前 locale 下的格式化函数。

推荐新增 `useFormat.ts`，避免每个页面重复处理 locale。

### 4.4 API 错误展示

`api/client.ts` 仍保留 `ApiError(error_code, message, details, status)`。

新增前端错误展示 helper：

```text
frontend/src/lib/errors.ts
```

职责：

- 输入 `unknown` error。
- 如果是 `ApiError`，优先用 `errors.api.<error_code>` 翻译。
- 如果没有对应 key，使用 `ApiError.message`。
- 非 ApiError 使用调用方 fallback 或 `errors.apiError`。

建议函数：

```ts
displayError(error, t, fallbackKey?)
```

迁移页面时把：

```ts
error.value = e instanceof Error ? e.message : 'Failed to load runs';
```

改为：

```ts
error.value = displayError(e, t, 'errors.failedToLoadRuns');
```

### 4.5 页面迁移优先级

按用户主路径优先迁移：

1. 应用外壳：`App.vue`、公共组件、格式化和错误 helper。
2. 新建评测 wizard：`EvaluationWizardPage.vue`、`components/wizard/*`。
3. 核心工作区：`HomePage.vue`、`DatasetsPage.vue`、`RunsPage.vue`、`RunDetailPage.vue`。
4. 结果页面：`LeaderboardsPage.vue`、`RankingPage.vue`、`ReportPage.vue`、`TrackPage.vue`、`SampleForecastPage.vue`。
5. admin 页面：`pages/admin/*`。
6. 测试中的硬编码断言同步为双语或默认英文断言。

第一阶段不要求一次性消灭所有英文硬编码，但每个迁移批次必须保持 key 完整和测试通过。

---

## 5. 翻译规则

### 必须翻译

- 导航、按钮、表头、说明文字、空态、错误提示。
- wizard 步骤标题、kicker、描述。
- 状态 badge 展示值。
- aria-label、title、placeholder。
- 日期相对时间中的自然语言。

### 不翻译

- `model_id`、`run_id`、`track_id`、`dataset_manifest_id` 等实体 ID。
- 权限 code：例如 `run.execute`、`role.manage`。
- API error code：例如 `auth_required`、`forbidden`。
- metric key：例如 `mae`、`mse`、未来的 `mase`。
- CSV 上传文件名、CSV 原始列名。
- 后端日志和开发调试信息。

### 避免字符串拼接

不要写：

```ts
`Run · ${count} models`
```

应写：

```ts
t('runs.modelCount', { count })
```

中英文分别决定语序：

```ts
// en-US
modelCount: 'Run · {count} models'

// zh-CN
modelCount: '运行 · {count} 个模型'
```

---

## 6. 测试方案

### 单元测试

新增测试：

- `frontend/src/tests/i18n.test.ts`
  - `en-US` 与 `zh-CN` key 集合完全一致。
  - 不存在空字符串翻译。
  - 不支持的 locale fallback 到 `en-US`。

- `frontend/src/tests/format.test.ts`
  - `formatDateTime`、`formatInt`、`timeAgo` 在 `en-US` / `zh-CN` 下输出非空且不抛错。

- `frontend/src/tests/errors.test.ts`
  - `ApiError('auth_required')` 能映射到当前语言文案。
  - 未知 `error_code` fallback 到后端 message。

### 组件测试

更新现有测试：

- 默认 locale 使用 `en-US`，减少一次性断言改动。
- 新增一条语言切换测试，覆盖 `App.vue` 导航或 wizard 标题从英文切到中文。
- 对 `StatusBadge` 增加中文状态展示测试。

### 验证命令

每个迁移批次至少运行：

```bash
cd frontend && npm test
```

如果改动涉及页面 shell 或主流程，再运行：

```bash
cd frontend && npm run test:e2e
```

---

## 7. 实施步骤

### Phase 1：基础设施

1. 安装 `vue-i18n`。
2. 新增 `src/i18n/*`。
3. 在 `main.ts` 注册 i18n。
4. 新增 locale 选择、持久化和 `<html lang>` 同步。
5. 新增 key 完整性测试。

验收：

- 应用仍可启动。
- 默认语言为英文。
- `localStorage` 可控制启动语言。
- `npm test` 通过。

### Phase 2：公共层

1. 迁移 `StateBlock.vue` 默认文案。
2. 迁移 `StatusBadge.vue` 状态文案。
3. 新增 `useFormat.ts` 或让 `format.ts` 支持显式 locale。
4. 新增 `lib/errors.ts`。

验收：

- 公共空态、错误、状态 badge 可随语言变化。
- 日期/数字格式不再依赖浏览器默认 locale。
- 相关单元测试通过。

### Phase 3：应用外壳

1. 迁移 `App.vue` 导航、面包屑、按钮、aria-label。
2. 增加语言切换入口。
3. 保持现有 hash routing 行为不变。

验收：

- 顶部语言切换立即生效。
- 刷新后保留语言选择。
- 登录/未登录外壳文案都能切换。

### Phase 4：wizard 主流程

1. 迁移 `EvaluationWizardPage.vue` 步骤文案。
2. 迁移 `components/wizard/*`。
3. 处理错误 fallback 和 loading 文案。

验收：

- 新建评测完整主流程无硬编码用户可见英文。
- 现有 wizard 测试通过。
- e2e smoke 通过。

### Phase 5：页面补齐

按优先级迁移：

1. Home、Datasets、Runs、RunDetail。
2. Leaderboards、Ranking、Report、Track、SampleForecast。
3. admin 页面。

验收：

- `rg -n "'[A-Z][^']*'|\"[A-Z][^\"]*\"" frontend/src` 人工检查无明显遗漏用户可见英文。
- 代码注释、API path、测试 fixture 可保留英文。
- `npm test` 通过。

---

## 8. 风险与处理

### 风险：测试断言大量依赖英文

处理：

- 默认测试 locale 保持 `en-US`。
- 不在第一批强制把所有测试改成 key 断言。
- 只为语言切换新增少量中文断言。

### 风险：遗漏 aria-label、title、placeholder

处理：

- 迁移时用 `rg "aria-label|title=|placeholder=" frontend/src` 做专项检查。
- 公共组件优先迁移，减少重复遗漏。

### 风险：后端 message 与前端翻译不一致

处理：

- 用户可见文案以前端 `error_code` 映射为准。
- 后端 message 只作为 fallback 和调试信息。
- 新增后端错误码时，同步补前端 `errors.api.<code>`。

### 风险：语言切换影响布局

处理：

- 中英文长度差异较大处优先检查按钮、表头、侧边栏、wizard 步骤条。
- 不用固定宽度承载长文本；必要时允许换行。
- 语言切换后跑一次主要页面视觉 smoke。

---

## 9. 完成标准

双语改造完成时应满足：

- 用户可在 UI 中切换 `English` / `中文`。
- 刷新后语言选择仍保留。
- 主流程和核心页面无明显硬编码用户可见英文。
- 状态、错误、空态、loading、日期/数字格式跟随语言。
- 后端 API 合约无变化。
- `cd frontend && npm test` 通过。
- 涉及主流程时 `cd frontend && npm run test:e2e` 通过。

---

## 10. 建议后续任务拆分

推荐按以下任务开工，每个任务独立提交：

1. `frontend-i18n-foundation`：依赖、i18n 初始化、locale 持久化、key 完整性测试。
2. `frontend-i18n-common-components`：StateBlock、StatusBadge、format、errors helper。
3. `frontend-i18n-app-shell`：App.vue、导航、面包屑、语言切换控件。
4. `frontend-i18n-wizard`：EvaluationWizardPage 与 wizard 组件。
5. `frontend-i18n-workspace-pages`：Home、Datasets、Runs、RunDetail。
6. `frontend-i18n-results-admin`：结果页和 admin 页补齐。

每个任务都应在 PR 或提交说明中列出：

- 已迁移文件。
- 新增/修改的 locale key。
- 执行过的测试命令。

