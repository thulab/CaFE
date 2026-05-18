<template>
  <main>
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
    <RankingTable :items="items" />
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

onMounted(load);

async function load() {
  items.value = (await getRanking(props.trackId, metric.value, policy.value)).items;
}
</script>
