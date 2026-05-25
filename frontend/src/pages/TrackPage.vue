<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Track</p>
        <h1>Track detail</h1>
        <p class="page-sub">Landing page for a benchmark track, with ranking controls and links into result artifacts.</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="`#/tracks/${trackId}/ranking`"><Icon name="trophy" :size="15" /> Standalone ranking</a>
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> New evaluation</a>
      </div>
    </header>

    <section class="stack">
      <article class="card">
        <header class="card-head"><h2 class="card-title">Track metadata</h2></header>
        <div class="card-body">
          <dl class="detail-grid">
            <div class="detail-item"><dt>Track ID</dt><dd class="mono">{{ trackId }}</dd></div>
            <div class="detail-item"><dt>Ranking route</dt><dd><a class="text-link" :href="`#/tracks/${trackId}/ranking`">Open standalone ranking</a></dd></div>
          </dl>
        </div>
      </article>

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">Ranking</h2>
        </header>
        <div class="card-body">
          <div class="toolbar">
            <div class="field">
              <label class="label">Metric</label>
              <select v-model="metric" aria-label="Metric" @change="load">
                <option value="mase">MASE</option>
                <option value="mse">MSE</option>
                <option value="mae">MAE</option>
              </select>
            </div>
            <div class="field">
              <label class="label">Policy</label>
              <select v-model="policy" aria-label="Policy" @change="load">
                <option value="latest_valid_result">latest_valid_result</option>
                <option value="best_result">best_result</option>
              </select>
            </div>
          </div>

          <StateBlock
            :loading="loading"
            :error="error"
            :empty="!loading && !error && items.length === 0"
            empty-title="No ranking yet"
            empty-desc="Run models on this track to populate the leaderboard."
            @retry="load"
          >
            <div class="stack">
              <RankingChart :items="items" />
              <hr class="divider" />
              <RankingTable :items="items" :metric-label="metric.toUpperCase()" />
            </div>
          </StateBlock>
        </div>
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import RankingTable from '../components/results/RankingTable.vue';
import RankingChart from '../components/results/RankingChart.vue';
import { getRanking } from '../api/results';

const props = defineProps<{ trackId: string }>();
const metric = ref('mase');
const policy = ref('latest_valid_result');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const loading = ref(true);
const error = ref<string | null>(null);

onMounted(load);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    items.value = (await getRanking(props.trackId, metric.value, policy.value)).items;
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load ranking';
  } finally {
    loading.value = false;
  }
}
</script>
