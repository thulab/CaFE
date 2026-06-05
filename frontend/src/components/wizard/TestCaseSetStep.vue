<template>
  <section class="step-body">
    <p class="field-help">
      {{ t('wizard.testCaseSetStep.description') }}
    </p>

    <div v-if="generatedShardLink" class="note-success" style="justify-content:space-between;gap:12px;flex-wrap:wrap">
      <span><Icon name="checkCircle" :size="16" />{{ t('wizard.testCaseSetStep.generatedReady', { name: generatedShardName }) }}</span>
      <a class="btn secondary sm" :href="generatedShardLink" :aria-label="t('wizard.testCaseSetStep.openGeneratedSet')">
        <Icon name="layers" :size="15" /> {{ t('wizard.testCaseSetStep.openGeneratedSet') }}
      </a>
    </div>

    <SelectablePagedList
      v-model:selected-ids="wizardState.selectedShardIds"
      :items="items"
      :total="total"
      :limit="limit"
      :offset="offset"
      :search="search"
      :labels="listLabels"
      :loading="loading"
      search-id="test-case-set-search"
      @update:search="updateSearch"
      @page="changePage"
    />

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="wizardState.trackId" class="note-success"><Icon name="checkCircle" :size="16" />{{ t('wizard.testCaseSetStep.trackReady') }}</p>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('wizard.testCaseSetStep.selectedCount', { count: wizardState.selectedShardIds.length }) }}</span>
      <button class="btn" type="button" :disabled="wizardState.selectedShardIds.length === 0 || busy" @click="createTrack">
        <span v-if="busy" class="spinner" />
        {{ wizardState.trackId ? t('wizard.testCaseSetStep.recreateTrack') : t('wizard.testCaseSetStep.createTrack') }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { listShards } from '../../api/datasets';
import { createTrackFromShards } from '../../api/tracks';
import type { ShardDTO } from '../../api/types';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { formatInt } from '../../lib/format';
import { goNext, resetWizard, wizardState } from '../../stores/wizard';
import Icon from '../ui/Icon.vue';
import SelectablePagedList from '../ui/SelectablePagedList.vue';

type SelectablePagedListItem = {
  id: string;
  title: string;
  href?: string;
  description?: string;
  meta?: string[];
  status?: string | null;
};

const { t, locale } = useI18n();
const props = withDefaults(defineProps<{ completion?: 'next' | 'open-track' }>(), {
  completion: 'next',
});
const limit = 10;
const search = ref('');
const offset = ref(0);
const total = ref(0);
const shards = ref<ShardDTO[]>([]);
const loading = ref(false);
const busy = ref(false);
const { text: error, clear: clearError, setError } = useDisplayMessage();

onMounted(() => {
  if (wizardState.shardId && !wizardState.selectedShardIds.includes(wizardState.shardId)) {
    wizardState.selectedShardIds = [wizardState.shardId, ...wizardState.selectedShardIds];
  }
});

watch([search, offset], loadShards, { immediate: true });

const items = computed<SelectablePagedListItem[]>(() => shards.value.map((shard) => {
  const meta = [
    shard.shard_type === 'synthetic' ? t('wizard.testCaseSetStep.synthetic') : t('wizard.testCaseSetStep.real'),
    t('wizard.testCaseSetStep.samples', { count: formatInt(shard.sample_count, locale.value) }),
    t('wizard.testCaseSetStep.window', { context: shard.context_length, horizon: shard.horizon }),
    t('wizard.testCaseSetStep.targets', { targets: shard.target_columns.join(', ') || '—' }),
  ];
  if (shard.covariate_columns?.length) {
    meta.push(t('wizard.testCaseSetStep.covariates', { covariates: shard.covariate_columns.join(', ') }));
  }
  if (shard.shard_type === 'synthetic') {
    const config = shard.generation_config || {};
    const capability = stringConfig(config.capability_label) || stringConfig(config.capability_id) || shard.capability_type || '';
    if (capability) meta.push(t('wizard.testCaseSetStep.capability', { capability }));
    if (config.difficulty) meta.push(t('wizard.testCaseSetStep.difficulty', { difficulty: config.difficulty }));
    if (config.seed !== undefined) meta.push(t('wizard.testCaseSetStep.seed', { seed: config.seed }));
  }
  return {
    id: shard.shard_id,
    title: shard.name || shardTitle(shard),
    href: `#/shards/${shard.shard_id}`,
    description: shard.dataset_name || shard.source_uri,
    meta,
    status: shard.status,
  };
}));

const generatedShardName = computed(() => wizardState.shardName || wizardState.shardId);
const generatedShardLink = computed(() => wizardState.shardId ? `#/shards/${wizardState.shardId}` : '');

const listLabels = computed(() => ({
  search: t('wizard.testCaseSetStep.search'),
  searchPlaceholder: t('wizard.testCaseSetStep.searchPlaceholder'),
  item: t('wizard.testCaseSetStep.item'),
  details: t('wizard.testCaseSetStep.details'),
  status: t('common.status'),
  loading: t('wizard.testCaseSetStep.loading'),
  empty: t('wizard.testCaseSetStep.empty'),
  previousPage: t('common.previousPage'),
  nextPage: t('common.nextPage'),
  selectItem: t('wizard.testCaseSetStep.selectItem'),
  togglePageSelection: t('wizard.testCaseSetStep.togglePageSelection'),
}));

async function loadShards() {
  loading.value = true;
  clearError();
  try {
    const result = await listShards({ limit, offset: offset.value, query: search.value });
    shards.value = result.items;
    total.value = result.total;
  } catch (caught) {
    setError(caught, 'wizard.testCaseSetStep.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

function updateSearch(value: string) {
  search.value = value;
  offset.value = 0;
}

function changePage(nextOffset: number) {
  offset.value = nextOffset;
}

async function createTrack() {
  clearError();
  busy.value = true;
  try {
    const result = await createTrackFromShards({
      name: wizardState.trackName,
      shard_ids: wizardState.selectedShardIds,
      primary_metric_id: wizardState.primaryMetric
    });
    wizardState.trackId = result.track_id;
    wizardState.capabilityBlockId = result.capability_block_id || result.capability_block_ids?.[0] || '';
    wizardState.rankingListId = result.ranking_list_id;
    if (props.completion === 'open-track') {
      const trackId = result.track_id;
      resetWizard('evaluation');
      window.location.hash = `#/tracks/${trackId}`;
      return;
    }
    goNext();
  } catch (caught) {
    setError(caught, 'wizard.testCaseSetStep.errors.failedToCreateTrack');
  } finally {
    busy.value = false;
  }
}

function shardTitle(shard: ShardDTO) {
  return t('wizard.testCaseSetStep.fallbackTitle', { target: shard.target_columns?.[0] || shard.shard_id });
}

function stringConfig(value: unknown) {
  return typeof value === 'string' ? value : '';
}
</script>
