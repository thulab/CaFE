<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('nav.workspace') }}</p>
        <h1>{{ t('tracks.title') }}</h1>
        <p class="page-sub">{{ t('tracks.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" href="#/datasets"><Icon name="database" :size="15" /> {{ t('nav.datasets') }}</a>
      </div>
    </header>

    <section class="card pad" style="display:grid;gap:16px">
      <header class="card-head" style="padding:0;border:0">
        <div>
          <h2 class="card-title">{{ t('tracks.createTitle') }}</h2>
          <p class="muted" style="margin:4px 0 0">{{ t('tracks.createDesc') }}</p>
        </div>
      </header>

      <div class="grid-auto">
        <div class="field">
          <label class="label" for="track-name">{{ t('tracks.trackName') }}</label>
          <input id="track-name" v-model.trim="trackName" :placeholder="t('tracks.trackNamePlaceholder')" />
        </div>
        <div class="field">
          <label class="label" for="track-primary-metric">{{ t('tracks.primaryMetric') }}</label>
          <select id="track-primary-metric" v-model="primaryMetric">
            <option value="mase">MASE</option>
            <option value="mse">MSE</option>
            <option value="mae">MAE</option>
          </select>
        </div>
      </div>

      <fieldset class="field" style="border:0;padding:0;margin:0">
        <legend class="label" style="padding:0;margin-bottom:6px">{{ t('tracks.shards') }}</legend>
        <div v-if="shards.length" class="choice-grid">
          <label v-for="shard in shards" :key="shard.shard_id" class="choice">
            <input v-model="selectedShardIds" type="checkbox" :value="shard.shard_id" :aria-label="shardTitle(shard)" />
            <span style="display:grid;gap:1px;min-width:0">
              <span>{{ shardTitle(shard) }}</span>
              <span class="faint" style="font-size:0.74rem">{{ shardSubtitle(shard) }}</span>
            </span>
          </label>
        </div>
        <p v-else class="status-line">{{ t('tracks.noReusableShards') }}</p>
      </fieldset>

      <p v-if="createdTrackId" class="note-success">
        <Icon name="checkCircle" :size="16" />{{ t('tracks.trackReady', { id: createdTrackId }) }}
        <a class="text-link" :href="`#/tracks/${createdTrackId}`">{{ t('tracks.openTrack') }}</a>
      </p>
      <p v-if="createError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ createError }}</p>

      <div class="wizard-foot" style="padding:0;border:0">
        <span class="status-line">{{ t('tracks.selectedShards', { count: selectedShardIds.length }) }}</span>
        <button class="btn" type="button" :disabled="creating || selectedShardIds.length === 0 || !trackName" @click="createTrack">
          <span v-if="creating" class="spinner" /> <Icon v-else name="target" :size="16" /> {{ t('tracks.createTrack') }}
        </button>
      </div>
    </section>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="error"
        :empty="!loading && !error && tracks.length === 0"
        empty-icon="target"
        :empty-title="t('tracks.noTracks')"
        :empty-desc="t('tracks.noTracksDesc')"
        @retry="load"
      >
        <div class="table-wrap">
          <table class="data">
            <thead>
              <tr>
                <th>{{ t('tracks.track') }}</th>
                <th>{{ t('tracks.metric') }}</th>
                <th>{{ t('tracks.shards') }}</th>
                <th>{{ t('tracks.samples') }}</th>
                <th>{{ t('tracks.status') }}</th>
                <th>{{ t('tracks.created') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="track in tracks" :key="track.track_id">
                <td>
                  <a class="text-link" :href="`#/tracks/${track.track_id}`">{{ track.name }}</a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(track.track_id) }}</div>
                </td>
                <td><span class="badge primary">{{ track.primary_metric_id.toUpperCase() }}</span></td>
                <td class="muted">{{ formatInt(track.shard_count ?? track.shard_ids?.length ?? 0) }}</td>
                <td class="muted">{{ formatInt(track.sample_count ?? 0) }}</td>
                <td><StatusBadge :status="track.status" /></td>
                <td class="muted nowrap" :title="track.created_at ? formatDateTime(track.created_at) : ''">{{ track.created_at ? timeAgo(track.created_at) : t('common.notAvailable') }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { listShards } from '../api/datasets';
import { createRealDatasetTrack, listTracks } from '../api/tracks';
import type { ShardDTO, TrackDTO } from '../api/types';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import Icon from '../components/ui/Icon.vue';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';

const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const tracks = ref<TrackDTO[]>([]);
const shards = ref<ShardDTO[]>([]);
const loading = ref(true);
const creating = ref(false);
const trackName = ref('');
const primaryMetric = ref('mase');
const selectedShardIds = ref<string[]>([]);
const createdTrackId = ref('');
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: createError, clear: clearCreateError, setError: setCreateError } = useDisplayMessage();

onMounted(load);

async function load() {
  loading.value = true;
  clearError();
  try {
    const [trackResp, shardResp] = await Promise.all([
      listTracks(),
      listShards({ limit: 200 })
    ]);
    tracks.value = trackResp.items ?? [];
    shards.value = (shardResp.items ?? []).filter((shard) => shard.status === 'ready');
  } catch (caught) {
    setError(caught, 'errors.failedToLoadTracks');
  } finally {
    loading.value = false;
  }
}

async function createTrack() {
  creating.value = true;
  createdTrackId.value = '';
  clearCreateError();
  try {
    const created = await createRealDatasetTrack({
      name: trackName.value,
      shard_ids: selectedShardIds.value,
      primary_metric_id: primaryMetric.value
    });
    createdTrackId.value = created.track_id;
    selectedShardIds.value = [];
    trackName.value = '';
    await load();
  } catch (caught) {
    setCreateError(caught, 'tracks.errors.failedToCreate');
  } finally {
    creating.value = false;
  }
}

function shardTitle(shard: ShardDTO) {
  return t('artifacts.shardTitle', { target: shard.target_columns?.[0] ?? t('artifacts.unknownTarget') });
}

function shardSubtitle(shard: ShardDTO) {
  const rows = t(shard.row_count === 1 ? 'datasets.rowCountOne' : 'datasets.rowCountOther', { count: formatInt(shard.row_count ?? 0) });
  return `${rows} · ${shard.context_length} ${t('wizard.columnAndSplitStep.context')} · ${shard.horizon} ${t('wizard.columnAndSplitStep.horizon')}`;
}
</script>
