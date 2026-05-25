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
        :empty="items.length === 0"
        empty-icon="activity"
        empty-title="No runs yet"
        empty-desc="Create a track and execute model adapters in a new evaluation to see runs here."
      >
        <template #empty-action>
          <a class="btn sm" href="#/new"><Icon name="play" :size="15" /> Start a run</a>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>Run</th><th>Last status</th><th>Created</th><th></th></tr></thead>
            <tbody>
              <tr v-for="item in items" :key="item.id">
                <td>
                  <a class="text-link" :href="item.href">
                    <Icon name="activity" :size="14" style="vertical-align:-2px;margin-right:6px" />{{ item.title }}
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(item.id) }}</div>
                </td>
                <td><StatusBadge :status="item.subtitle" /></td>
                <td class="muted nowrap" :title="formatDateTime(item.createdAt)">{{ timeAgo(item.createdAt) }}</td>
                <td style="text-align:right"><a class="btn secondary sm" :href="item.href">Open <Icon name="arrowRight" :size="14" /></a></td>
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
import StatusBadge from '../components/ui/StatusBadge.vue';
import { listRecents } from '../stores/recents';
import { formatDateTime, shortId, timeAgo } from '../lib/format';

const items = computed(() => listRecents('run'));
</script>
