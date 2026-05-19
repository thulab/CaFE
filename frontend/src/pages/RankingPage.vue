<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Results</p>
        <h1>Track ranking</h1>
        <p class="page-subtitle">Compare model performance on the selected benchmark track.</p>
      </div>
    </header>

    <section class="page-card">
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
