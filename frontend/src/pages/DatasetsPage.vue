<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('nav.workspace') }}</p>
        <h1>{{ t('datasets.title') }}</h1>
        <p class="page-sub">{{ t('datasets.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> {{ t('nav.newEvaluation') }}</a>
      </div>
    </header>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="error || ''"
        :empty="!loading && !error && items.length === 0"
        empty-icon="database"
        :empty-title="t('datasets.noDatasets')"
        :empty-desc="t('datasets.noDatasetsDesc')"
        @retry="load"
      >
        <template #empty-action>
          <a class="btn sm" href="#/new"><Icon name="upload" :size="15" /> {{ t('datasets.uploadCsv') }}</a>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>{{ t('datasets.artifact') }}</th><th>{{ t('datasets.type') }}</th><th>{{ t('datasets.detail') }}</th><th>{{ t('datasets.created') }}</th></tr></thead>
            <tbody>
              <tr v-for="item in displayItems" :key="`${item.kind}-${item.id}`">
                <td>
                  <a class="text-link" :href="item.href">
                    <Icon :name="kindIcon(item.kind)" :size="14" style="vertical-align:-2px;margin-right:6px" />{{ item.title }}
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(item.id) }}</div>
                </td>
                <td><span class="badge" :class="item.kind === 'shard' ? 'primary' : ''">{{ t(`datasets.kind.${item.kind}`) }}</span></td>
                <td class="muted">{{ item.subtitle || t('common.notAvailable') }}</td>
                <td class="muted nowrap" :title="item.createdAt ? formatDateTime(item.createdAt) : ''">{{ item.createdAt ? timeAgo(item.createdAt) : t('common.notAvailable') }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import { listDatasetManifests, listShards } from '../api/datasets';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';
import { useI18n } from 'vue-i18n';

type Kind = 'dataset' | 'shard';
interface Row {
  kind: Kind;
  id: string;
  name?: string;
  targetColumns?: string[];
  rowCount?: number;
  title?: string;
  subtitle?: string;
  href: string;
  createdAt?: string;
}

const items = ref<Row[]>([]);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();

const ICONS: Record<Kind, string> = { dataset: 'database', shard: 'layers' };
const kindIcon = (k: Kind) => ICONS[k] || 'file';

const displayItems = computed(() => items.value.map((item) => ({
  ...item,
  title: rowTitle(item),
  subtitle: rowSubtitle(item),
})));

function rowTitle(item: Row): string {
  if (item.kind === 'dataset') return item.name ?? '';
  return t('artifacts.shardTitle', { target: item.targetColumns?.[0] ?? t('artifacts.unknownTarget') });
}

function rowSubtitle(item: Row): string | undefined {
  if (item.kind === 'dataset') return item.subtitle;
  return shardSubtitle(item.targetColumns ?? [], item.rowCount ?? 0);
}

function shardSubtitle(targetColumns: string[], rowCount: number): string {
  const cols = targetColumns.length ? targetColumns.join(', ') : t('artifacts.shard');
  if (!rowCount) return cols;
  const key = rowCount === 1 ? 'datasets.rowCountOne' : 'datasets.rowCountOther';
  return `${cols} · ${t(key, { count: formatInt(rowCount) })}`;
}

async function load() {
  loading.value = true;
  clearError();
  try {
    const [manifests, shards] = await Promise.all([
      listDatasetManifests({ limit: 200 }),
      listShards({ limit: 200 })
    ]);
    const rows: Row[] = [];
    for (const m of manifests.items) {
      rows.push({
        kind: 'dataset',
        id: m.dataset_manifest_id,
        name: m.name,
        subtitle: m.domain,
        href: `#/datasets/${m.dataset_manifest_id}`,
        createdAt: m.created_at
      });
    }
    for (const s of shards.items) {
      rows.push({
        kind: 'shard',
        id: s.shard_id,
        targetColumns: s.target_columns ?? [],
        rowCount: s.row_count ?? 0,
        href: `#/shards/${s.shard_id}`,
        createdAt: s.created_at
      });
    }
    rows.sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
    items.value = rows;
  } catch (e) {
    setError(e, 'errors.failedToLoadDatasets');
    items.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>
