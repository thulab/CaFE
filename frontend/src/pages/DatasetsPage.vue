<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Workspace</p>
        <h1>Datasets</h1>
        <p class="page-sub">Dataset manifests and evaluation shards stored in this workspace.</p>
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
        empty-icon="database"
        empty-title="No datasets yet"
        empty-desc="Upload a CSV in a new evaluation to create your first dataset manifest and shard."
        @retry="load"
      >
        <template #empty-action>
          <a class="btn sm" href="#/new"><Icon name="upload" :size="15" /> Upload a CSV</a>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>Artifact</th><th>Type</th><th>Detail</th><th>Created</th></tr></thead>
            <tbody>
              <tr v-for="item in items" :key="`${item.kind}-${item.id}`">
                <td>
                  <a class="text-link" :href="item.href">
                    <Icon :name="kindIcon(item.kind)" :size="14" style="vertical-align:-2px;margin-right:6px" />{{ item.title }}
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(item.id) }}</div>
                </td>
                <td><span class="badge" :class="item.kind === 'shard' ? 'primary' : ''">{{ humanize(item.kind) }}</span></td>
                <td class="muted">{{ item.subtitle || '—' }}</td>
                <td class="muted nowrap" :title="item.createdAt ? formatDateTime(item.createdAt) : ''">{{ item.createdAt ? timeAgo(item.createdAt) : '—' }}</td>
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
import { listDatasetManifests, listShards } from '../api/datasets';
import { formatDateTime, humanize, shortId, timeAgo } from '../lib/format';

type Kind = 'dataset' | 'shard';
interface Row {
  kind: Kind;
  id: string;
  title: string;
  subtitle?: string;
  href: string;
  createdAt?: string;
}

const items = ref<Row[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

const ICONS: Record<Kind, string> = { dataset: 'database', shard: 'layers' };
const kindIcon = (k: Kind) => ICONS[k] || 'file';

function shardSubtitle(targetColumns: string[], rowCount: number): string {
  const cols = targetColumns.length ? targetColumns.join(', ') : 'shard';
  return rowCount ? `${cols} · ${rowCount} rows` : cols;
}

async function load() {
  loading.value = true;
  error.value = null;
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
        title: m.name,
        subtitle: m.domain,
        href: `#/datasets/${m.dataset_manifest_id}`,
        createdAt: m.created_at
      });
    }
    for (const s of shards.items) {
      rows.push({
        kind: 'shard',
        id: s.shard_id,
        title: `Shard · ${s.target_columns?.[0] ?? 'target'}`,
        subtitle: shardSubtitle(s.target_columns ?? [], s.row_count ?? 0),
        href: `#/shards/${s.shard_id}`,
        createdAt: s.created_at
      });
    }
    rows.sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
    items.value = rows;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load datasets';
    items.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>
