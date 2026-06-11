<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('leaderboards.eyebrow') }}</p>
        <h1>{{ t('leaderboards.title') }}</h1>
        <p class="page-sub">{{ t('leaderboards.subtitle') }}</p>
      </div>
    </header>

    <section class="card pad">
      <div class="toolbar">
        <div class="field">
          <label class="label" for="lb-search">{{ t('leaderboards.search') }}</label>
          <input id="lb-search" v-model="query" type="search" :placeholder="t('leaderboards.searchPlaceholder')" />
        </div>
        <div class="field">
          <label class="label" for="lb-type">{{ t('leaderboards.trackType') }}</label>
          <select id="lb-type" v-model="typeFilter">
            <option value="">{{ t('leaderboards.allTypes') }}</option>
            <option v-for="t in trackTypes" :key="t" :value="t">{{ t }}</option>
          </select>
        </div>
        <span class="spacer" />
        <span v-if="!loading && !error" class="badge"><Icon name="trophy" :size="13" /> {{ boardCountLabel }}</span>
      </div>

      <StateBlock
        :loading="loading"
        :error="error"
        :empty="!loading && !error && filtered.length === 0"
        empty-icon="trophy"
        :empty-title="t('leaderboards.noLeaderboards')"
        :empty-desc="t('leaderboards.noLeaderboardsDesc')"
        @retry="load"
      >
        <div class="stack">
          <p v-if="actionError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ actionError }}</p>
          <div class="grid-auto">
            <LeaderboardCard
              v-for="item in filtered"
              :key="item.ranking_list_id"
              :item="item"
              :manageable="canManageRanking"
              :busy="busyRankingListId === item.ranking_list_id"
              @visibility-change="setPublicVisibility(item, $event)"
            />
          </div>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import LeaderboardCard from '../components/results/LeaderboardCard.vue';
import { listRankingLists, setRankingPublicVisible, type LeaderboardItem } from '../api/results';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { has } from '../stores/auth';

const items = ref<LeaderboardItem[]>([]);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: actionError, clear: clearActionError, setError: setActionError } = useDisplayMessage();
const query = ref('');
const typeFilter = ref('');
const busyRankingListId = ref('');
const { t } = useI18n();

const canManageRanking = computed(() => has('track.manage'));
const trackTypes = computed(() => Array.from(new Set(items.value.map((i) => i.track_type))).sort());

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase();
  return items.value.filter((i) => {
    if (typeFilter.value && i.track_type !== typeFilter.value) return false;
    if (q && !i.track_name.toLowerCase().includes(q)) return false;
    return true;
  });
});
const boardCountLabel = computed(() =>
  t(filtered.value.length === 1 ? 'leaderboards.boardCountOne' : 'leaderboards.boardCountOther', { count: filtered.value.length })
);

onMounted(load);

async function load() {
  loading.value = true;
  clearError();
  try {
    items.value = (await listRankingLists()).items;
  } catch (e) {
    setError(e, 'leaderboards.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

async function setPublicVisibility(item: LeaderboardItem, publicVisible: boolean) {
  busyRankingListId.value = item.ranking_list_id;
  clearActionError();
  try {
    const updated = await setRankingPublicVisible(item.ranking_list_id, publicVisible);
    items.value = items.value.map((candidate) =>
      candidate.ranking_list_id === item.ranking_list_id
        ? { ...candidate, public_visible: updated.public_visible, updated_at: updated.updated_at }
        : candidate
    );
  } catch (e) {
    setActionError(e, 'leaderboards.errors.failedToUpdateVisibility');
  } finally {
    busyRankingListId.value = '';
  }
}
</script>
