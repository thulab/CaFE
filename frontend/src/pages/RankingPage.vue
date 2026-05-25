<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Results</p>
        <h1>Track ranking</h1>
        <p class="page-sub">Compare model performance on the selected benchmark track.</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="`#/tracks/${trackId}`"><Icon name="target" :size="15" /> Track detail</a>
      </div>
    </header>

    <section class="card pad">
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
        <span class="spacer" />
        <span v-if="items.length" class="badge"><Icon name="trophy" :size="13" /> {{ items.length }} models ranked</span>
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
