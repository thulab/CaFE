<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('nav.workspace') }}</p>
        <h1>{{ t('tracks.title') }}</h1>
        <p class="page-sub">{{ t('tracks.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <button class="btn sm" type="button" @click="startCreateTrack">
          <Icon name="plus" :size="15" /> {{ t('tracks.newTrack') }}
        </button>
        <a class="btn secondary sm" href="#/datasets"><Icon name="database" :size="15" /> {{ t('nav.datasets') }}</a>
      </div>
    </header>

    <section class="card pad">
      <div class="toolbar">
        <label class="choice compact">
          <input v-model="showArchived" type="checkbox" :aria-label="t('lifecycle.showArchived')" @change="load" />
          {{ t('lifecycle.showArchived') }}
        </label>
      </div>
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
                <th>{{ t('common.actions') }}</th>
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
                <td>
                  <div class="pill-row">
                    <StatusBadge :status="track.status" />
                    <span v-if="track.archived_at" class="badge warning">{{ t('lifecycle.archived') }}</span>
                  </div>
                </td>
                <td class="muted nowrap" :title="track.created_at ? formatDateTime(track.created_at) : ''">{{ track.created_at ? timeAgo(track.created_at) : t('common.notAvailable') }}</td>
                <td>
                  <div class="pill-row">
                    <button v-if="!track.archived_at" class="btn secondary sm" type="button" @click="openLifecycle('archive', 'track', track.track_id)">{{ t('lifecycle.archive') }}</button>
                    <button v-else class="btn secondary sm" type="button" @click="openLifecycle('restore', 'track', track.track_id)">{{ t('lifecycle.restore') }}</button>
                    <button class="btn danger sm" type="button" @click="openLifecycle('purge', 'track', track.track_id)">{{ t('lifecycle.permanentDelete') }}</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
    <ResourceActionDialog
      :open="dialog.open"
      :resource-type="dialog.resourceType"
      :resource-id="dialog.resourceId"
      :action="dialog.action"
      @close="dialog.open = false"
      @done="load"
    />
  </main>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { listTracks } from '../api/tracks';
import type { ResourceType, TrackDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import StateBlock from '../components/ui/StateBlock.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import Icon from '../components/ui/Icon.vue';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';
import { resetWizard, wizardState } from '../stores/wizard';

const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const tracks = ref<TrackDTO[]>([]);
const loading = ref(true);
const showArchived = ref(false);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const dialog = reactive<{ open: boolean; action: LifecycleAction; resourceType: ResourceType; resourceId: string }>({
  open: false,
  action: 'archive',
  resourceType: 'track',
  resourceId: '',
});

onMounted(load);

async function load() {
  loading.value = true;
  clearError();
  try {
    const trackResp = await listTracks({ includeArchived: showArchived.value });
    tracks.value = trackResp.items ?? [];
  } catch (caught) {
    setError(caught, 'errors.failedToLoadTracks');
  } finally {
    loading.value = false;
  }
}

function openLifecycle(action: LifecycleAction, resourceType: ResourceType, resourceId: string) {
  dialog.action = action;
  dialog.resourceType = resourceType;
  dialog.resourceId = resourceId;
  dialog.open = true;
}

function startCreateTrack() {
  resetWizard('track');
  wizardState.entryMode = 'new-track';
  window.location.hash = `#/tracks/new?session=${Date.now()}`;
}
</script>
