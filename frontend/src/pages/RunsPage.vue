<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('nav.workspace') }}</p>
        <h1>{{ t('runs.title') }}</h1>
        <p class="page-sub">{{ t('runs.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> {{ t('nav.newEvaluation') }}</a>
      </div>
    </header>

    <section class="card pad">
      <div class="toolbar">
        <label class="choice compact">
          <input v-model="showArchived" type="checkbox" :aria-label="t('lifecycle.showArchived')" @change="load" />
          {{ t('lifecycle.showArchived') }}
        </label>
      </div>
      <StateBlock
        :loading="loading"
        :error="error || ''"
        :empty="!loading && !error && items.length === 0"
        empty-icon="activity"
        :empty-title="t('runs.noRuns')"
        :empty-desc="t('runs.noRunsDesc')"
        @retry="load"
      >
        <template #empty-action>
          <a class="btn sm" href="#/new"><Icon name="play" :size="15" /> {{ t('runs.startRun') }}</a>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>{{ t('runs.run') }}</th><th>{{ t('runs.lastStatus') }}</th><th>{{ t('runs.created') }}</th><th>{{ t('common.actions') }}</th></tr></thead>
            <tbody>
              <tr v-for="run in items" :key="run.benchmarking_run_id">
                <td>
                  <a class="text-link" :href="`#/runs/${run.benchmarking_run_id}`">
                    <Icon name="activity" :size="14" style="vertical-align:-2px;margin-right:6px" />{{ runTitle(run) }}
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(run.benchmarking_run_id) }}</div>
                </td>
                <td>
                  <div class="pill-row">
                    <StatusBadge :status="run.activity_status || run.status" />
                    <span v-if="run.archived_at" class="badge warning">{{ t('lifecycle.archived') }}</span>
                  </div>
                </td>
                <td class="muted nowrap" :title="run.created_at ? formatDateTime(run.created_at) : ''">{{ run.created_at ? timeAgo(run.created_at) : t('common.notAvailable') }}</td>
                <td>
                  <div class="pill-row" style="justify-content:flex-end">
                    <a class="btn secondary sm" :href="`#/runs/${run.benchmarking_run_id}`">{{ t('common.open') }} <Icon name="arrowRight" :size="14" /></a>
                    <button v-if="!run.archived_at" class="btn secondary sm" type="button" @click="openLifecycle('archive', run.benchmarking_run_id)">{{ t('lifecycle.archive') }}</button>
                    <button v-else class="btn secondary sm" type="button" @click="openLifecycle('restore', run.benchmarking_run_id)">{{ t('lifecycle.restore') }}</button>
                    <button class="btn danger sm" type="button" @click="openLifecycle('purge', run.benchmarking_run_id)">{{ t('lifecycle.permanentDelete') }}</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
    <ResourceActionDialog
      :open="dialog.open"
      resource-type="benchmarking_run"
      :resource-id="dialog.resourceId"
      :action="dialog.action"
      @close="dialog.open = false"
      @done="load"
    />
  </main>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import { listRuns } from '../api/runs';
import type { BenchmarkingRunSummaryDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';
import { useI18n } from 'vue-i18n';

const items = ref<BenchmarkingRunSummaryDTO[]>([]);
const loading = ref(true);
const showArchived = ref(false);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const dialog = reactive<{ open: boolean; action: LifecycleAction; resourceId: string }>({
  open: false,
  action: 'archive',
  resourceId: '',
});

function runTitle(run: BenchmarkingRunSummaryDTO): string {
  const count = run.model_count || run.model_ids?.length || 0;
  return t(count === 1 ? 'artifacts.runTitleOne' : 'artifacts.runTitleOther', { count: formatInt(count) });
}

async function load() {
  loading.value = true;
  clearError();
  try {
    const res = await listRuns({ limit: 200, includeArchived: showArchived.value });
    items.value = res.items;
  } catch (e) {
    setError(e, 'errors.failedToLoadRuns');
    items.value = [];
  } finally {
    loading.value = false;
  }
}

function openLifecycle(action: LifecycleAction, resourceId: string) {
  dialog.action = action;
  dialog.resourceId = resourceId;
  dialog.open = true;
}

onMounted(load);
</script>
