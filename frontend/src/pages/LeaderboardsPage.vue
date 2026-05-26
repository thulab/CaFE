<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Results</p>
        <h1>Leaderboards</h1>
        <p class="page-sub">Compare model performance across all benchmark tracks.</p>
      </div>
    </header>

    <section class="card pad">
      <div class="toolbar">
        <div class="field">
          <label class="label" for="lb-search">Search</label>
          <input id="lb-search" v-model="query" type="search" placeholder="Filter by track name…" />
        </div>
        <div class="field">
          <label class="label" for="lb-type">Track type</label>
          <select id="lb-type" v-model="typeFilter">
            <option value="">All types</option>
            <option v-for="t in trackTypes" :key="t" :value="t">{{ t }}</option>
          </select>
        </div>
        <span class="spacer" />
        <span v-if="!loading && !error" class="badge"><Icon name="trophy" :size="13" /> {{ filtered.length }} board{{ filtered.length === 1 ? '' : 's' }}</span>
      </div>

      <StateBlock
        :loading="loading"
        :error="error"
        :empty="!loading && !error && filtered.length === 0"
        empty-icon="trophy"
        empty-title="No leaderboards yet"
        empty-desc="Run models on a track to populate one."
        @retry="load"
      >
        <div class="grid-auto">
          <LeaderboardCard v-for="item in filtered" :key="item.ranking_list_id" :item="item" />
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import LeaderboardCard from '../components/results/LeaderboardCard.vue';
import { listRankingLists, type LeaderboardItem } from '../api/results';

const items = ref<LeaderboardItem[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);
const query = ref('');
const typeFilter = ref('');

const trackTypes = computed(() => Array.from(new Set(items.value.map((i) => i.track_type))).sort());

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase();
  return items.value.filter((i) => {
    if (typeFilter.value && i.track_type !== typeFilter.value) return false;
    if (q && !i.track_name.toLowerCase().includes(q)) return false;
    return true;
  });
});

onMounted(load);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    items.value = (await listRankingLists()).items;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load leaderboards';
  } finally {
    loading.value = false;
  }
}
</script>
