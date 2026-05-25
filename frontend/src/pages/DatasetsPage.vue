<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Workspace</p>
        <h1>Datasets</h1>
        <p class="page-sub">Dataset manifests and evaluation shards created in this workspace.</p>
      </div>
      <div class="head-actions">
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> New evaluation</a>
      </div>
    </header>

    <section class="card pad">
      <StateBlock
        :empty="items.length === 0"
        empty-icon="database"
        empty-title="No datasets yet"
        empty-desc="Upload a CSV in a new evaluation to create your first dataset manifest and shard."
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
                <td class="muted nowrap" :title="formatDateTime(item.createdAt)">{{ timeAgo(item.createdAt) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import { listRecents, type RecentKind } from '../stores/recents';
import { formatDateTime, humanize, shortId, timeAgo } from '../lib/format';

const items = computed(() => listRecents(['dataset', 'shard']));
const ICONS: Record<string, string> = { dataset: 'database', shard: 'layers' };
const kindIcon = (k: RecentKind) => ICONS[k] || 'file';
</script>
