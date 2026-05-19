<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Track</p>
        <h1>Track detail</h1>
        <p class="page-subtitle">Use this view as the landing page for a benchmark track, with ranking controls and links into result artifacts.</p>
      </div>
      <a class="button-link secondary" href="#/">Create workflow</a>
    </header>

    <section class="report-sections">
      <article class="page-card">
        <h2 class="section-title">Track metadata</h2>
        <dl class="detail-grid">
          <div class="detail-item">
            <dt>Track ID</dt>
            <dd class="mono-text">{{ trackId }}</dd>
          </div>
          <div class="detail-item">
            <dt>Ranking route</dt>
            <dd><a class="text-link" :href="`#/tracks/${trackId}/ranking`">Open standalone ranking</a></dd>
          </div>
        </dl>
      </article>

      <article class="page-card">
        <div class="section-header">
          <h2 class="section-title">Ranking</h2>
          <span class="stat-pill">Lower is better</span>
        </div>
        <div class="filters">
          <label>Metric
            <select v-model="metric" aria-label="Metric" @change="load">
              <option value="mse">mse</option>
              <option value="mae">mae</option>
            </select>
          </label>
          <label>Policy
            <select v-model="policy" aria-label="Policy" @change="load">
              <option value="latest_valid_result">latest_valid_result</option>
              <option value="best_result">best_result</option>
            </select>
          </label>
        </div>
        <p v-if="loading" class="status-line">Loading ranking...</p>
        <RankingTable :items="items" />
      </article>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { getRanking } from '../api/results';
import RankingTable from '../components/results/RankingTable.vue';

const props = defineProps<{ trackId: string }>();
const metric = ref('mse');
const policy = ref('latest_valid_result');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const loading = ref(true);
const trackId = props.trackId;

onMounted(load);

async function load() {
  loading.value = true;
  try {
    items.value = (await getRanking(props.trackId, metric.value, policy.value)).items;
  } finally {
    loading.value = false;
  }
}
</script>
