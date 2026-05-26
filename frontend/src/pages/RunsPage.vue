<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Workspace</p>
        <h1>Runs</h1>
        <p class="page-sub">Benchmarking runs launched in this workspace. Open one for live progress and results.</p>
      </div>
      <div class="head-actions">
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> New evaluation</a>
      </div>
    </header>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="error || ''"
        :empty="!loading && !error && items.length === 0"
        empty-icon="activity"
        empty-title="No runs yet"
        empty-desc="Create a track and execute model adapters in a new evaluation to see runs here."
        @retry="load"
      >
        <template #empty-action>
          <a class="btn sm" href="#/new"><Icon name="play" :size="15" /> Start a run</a>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>Run</th><th>Last status</th><th>Created</th><th></th></tr></thead>
            <tbody>
              <tr v-for="run in items" :key="run.benchmarking_run_id">
                <td>
                  <a class="text-link" :href="`#/runs/${run.benchmarking_run_id}`">
                    <Icon name="activity" :size="14" style="vertical-align:-2px;margin-right:6px" />Run · {{ run.model_count || run.model_ids?.length || 0 }} models
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(run.benchmarking_run_id) }}</div>
                </td>
                <td><StatusBadge :status="run.status" /></td>
                <td class="muted nowrap" :title="run.created_at ? formatDateTime(run.created_at) : ''">{{ run.created_at ? timeAgo(run.created_at) : '—' }}</td>
                <td style="text-align:right"><a class="btn secondary sm" :href="`#/runs/${run.benchmarking_run_id}`">Open <Icon name="arrowRight" :size="14" /></a></td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { listRuns } from '../api/runs';
import type { BenchmarkingRunSummaryDTO } from '../api/types';
import { formatDateTime, shortId, timeAgo } from '../lib/format';

const items = ref<BenchmarkingRunSummaryDTO[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    const res = await listRuns({ limit: 200 });
    items.value = res.items;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load runs';
    items.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>
