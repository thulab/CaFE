<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('ranking.eyebrow') }}</p>
        <h1>{{ t('ranking.title') }}</h1>
        <p class="page-sub">{{ t('ranking.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <ResumeWizardButton resource-type="ranking" :resource-id="trackId" />
        <a class="btn secondary sm" :href="`#/tracks/${trackId}`"><Icon name="target" :size="15" /> {{ t('ranking.trackDetail') }}</a>
      </div>
    </header>

    <section class="card pad">
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
        <span class="spacer" />
        <span v-if="canManageRanking" class="badge" :class="publicVisible ? 'success' : 'warning'">
          {{ publicVisible ? t('leaderboards.public') : t('leaderboards.hidden') }}
        </span>
        <button v-if="canManageRanking" class="btn ghost sm" type="button" :disabled="visibilityBusy" @click="toggleVisibility">
          <span v-if="visibilityBusy" class="spinner" />
          <Icon v-else :name="publicVisible ? 'eyeOff' : 'eye'" :size="14" />
          {{ publicVisible ? t('leaderboards.hideFromPublic') : t('leaderboards.makePublic') }}
        </button>
        <span v-if="items.length" class="badge"><Icon name="trophy" :size="13" /> {{ rankedCountLabel }}</span>
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
          <p v-if="actionError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ actionError }}</p>
          <RankingChart :items="items" />
          <hr class="divider" />
          <RankingTable :items="items" :metric-label="metric.toUpperCase()" />
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import ResumeWizardButton from '../components/wizard/ResumeWizardButton.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import RankingTable from '../components/results/RankingTable.vue';
import RankingChart from '../components/results/RankingChart.vue';
import { getRanking, setRankingPublicVisible } from '../api/results';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { has } from '../stores/auth';

const props = defineProps<{ trackId: string }>();
const metric = ref('mase');
const policy = ref('latest_valid_result');
const items = ref<Array<{ model_id: string; rank: number; metric_value: number }>>([]);
const rankingListId = ref('');
const publicVisible = ref(true);
const visibilityBusy = ref(false);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: actionError, clear: clearActionError, setError: setActionError } = useDisplayMessage();
const { t } = useI18n();
const rankedCountLabel = computed(() =>
  t(items.value.length === 1 ? 'ranking.modelsRankedOne' : 'ranking.modelsRankedOther', { count: items.value.length })
);
const canManageRanking = computed(() => Boolean(rankingListId.value) && has('track.manage'));

onMounted(load);

async function load() {
  loading.value = true;
  clearError();
  try {
    const ranking = await getRanking(props.trackId, metric.value, policy.value);
    items.value = ranking.items;
    rankingListId.value = ranking.ranking_list_id || '';
    publicVisible.value = ranking.public_visible !== false;
  } catch (e) {
    setError(e, 'ranking.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

async function toggleVisibility() {
  if (!rankingListId.value) return;
  visibilityBusy.value = true;
  clearActionError();
  try {
    const updated = await setRankingPublicVisible(rankingListId.value, !publicVisible.value);
    publicVisible.value = updated.public_visible;
  } catch (e) {
    setActionError(e, 'leaderboards.errors.failedToUpdateVisibility');
  } finally {
    visibilityBusy.value = false;
  }
}
</script>
