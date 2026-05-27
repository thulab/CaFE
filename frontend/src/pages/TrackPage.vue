<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('track.eyebrow') }}</p>
        <h1>{{ t('track.title') }}</h1>
        <p class="page-sub">{{ t('track.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="`#/tracks/${trackId}/ranking`"><Icon name="trophy" :size="15" /> {{ t('track.standaloneRanking') }}</a>
        <a class="btn accent sm" href="#/new"><Icon name="plus" :size="15" /> {{ t('track.newEvaluation') }}</a>
      </div>
    </header>

    <section class="stack">
      <article class="card">
        <header class="card-head"><h2 class="card-title">{{ t('track.metadata') }}</h2></header>
        <div class="card-body">
          <dl class="detail-grid">
            <div class="detail-item"><dt>{{ t('track.trackId') }}</dt><dd class="mono">{{ trackId }}</dd></div>
            <div class="detail-item"><dt>{{ t('track.rankingRoute') }}</dt><dd><a class="text-link" :href="`#/tracks/${trackId}/ranking`">{{ t('track.openStandaloneRanking') }}</a></dd></div>
          </dl>
        </div>
      </article>

      <article class="card">
        <header class="card-head">
          <h2 class="card-title">{{ t('track.ranking') }}</h2>
        </header>
        <div class="card-body">
          <div class="toolbar">
            <div class="field">
              <label class="label">{{ t('ranking.metric') }}</label>
              <select v-model="metric" :aria-label="t('ranking.metric')" @change="load">
                <option value="mase">MASE</option>
                <option value="mse">MSE</option>
                <option value="mae">MAE</option>
              </select>
            </div>
            <div class="field">
              <label class="label">{{ t('ranking.policy') }}</label>
              <select v-model="policy" :aria-label="t('ranking.policy')" @change="load">
                <option value="latest_valid_result">latest_valid_result</option>
                <option value="best_result">best_result</option>
              </select>
            </div>
          </div>

          <StateBlock
            :loading="loading"
            :error="error"
            :empty="!loading && !error && items.length === 0"
            :empty-title="t('ranking.noRanking')"
            :empty-desc="t('ranking.noRankingDesc')"
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
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import RankingTable from '../components/results/RankingTable.vue';
import RankingChart from '../components/results/RankingChart.vue';
import { getRanking } from '../api/results';
import { useDisplayMessage } from '../composables/useDisplayMessage';

const props = defineProps<{ trackId: string }>();
const metric = ref('mase');
const policy = ref('latest_valid_result');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { t } = useI18n();

onMounted(load);

async function load() {
  loading.value = true;
  clearError();
  try {
    items.value = (await getRanking(props.trackId, metric.value, policy.value)).items;
  } catch (e) {
    setError(e, 'ranking.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}
</script>
