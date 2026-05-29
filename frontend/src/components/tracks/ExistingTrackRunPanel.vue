<template>
  <section class="card pad" style="display:grid;gap:14px">
    <header class="card-head" style="padding:0;border:0">
      <div>
        <h2 class="card-title">{{ t('runPanel.existingTrackTitle') }}</h2>
        <p class="muted" style="margin:4px 0 0">{{ t('runPanel.existingTrackDesc') }}</p>
      </div>
      <a class="btn secondary sm" href="#/tracks"><Icon name="target" :size="15" />{{ t('nav.tracks') }}</a>
    </header>

    <StateBlock
      :loading="loading"
      :error="error"
      :empty="!loading && !error && tracks.length === 0"
      empty-icon="target"
      :empty-title="t('runPanel.noTracks')"
      :empty-desc="t('runPanel.noTracksDesc')"
      @retry="load"
    >
      <div class="field">
        <label class="label" for="existing-track">{{ t('runPanel.track') }}</label>
        <select id="existing-track" v-model="selectedTrackId">
          <option v-for="track in tracks" :key="track.track_id" :value="track.track_id">
            {{ track.name }} · {{ track.primary_metric_id.toUpperCase() }}
          </option>
        </select>
      </div>
      <TrackRunPanel v-if="selectedTrackId" :track-id="selectedTrackId" @run-created="onRunCreated" />
    </StateBlock>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { listTracks } from '../../api/tracks';
import type { TrackDTO } from '../../api/types';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { wizardState } from '../../stores/wizard';
import Icon from '../ui/Icon.vue';
import StateBlock from '../ui/StateBlock.vue';
import TrackRunPanel from './TrackRunPanel.vue';

const { t } = useI18n();
const tracks = ref<TrackDTO[]>([]);
const selectedTrackId = ref('');
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();

onMounted(load);
watch(selectedTrackId, (trackId) => {
  wizardState.entryMode = 'existing-track';
  wizardState.trackId = trackId;
});

async function load() {
  loading.value = true;
  clearError();
  try {
    tracks.value = (await listTracks()).items ?? [];
    selectedTrackId.value = tracks.value.some((track) => track.track_id === wizardState.trackId)
      ? wizardState.trackId
      : tracks.value[0]?.track_id || '';
  } catch (caught) {
    setError(caught, 'errors.failedToLoadTracks');
  } finally {
    loading.value = false;
  }
}

function onRunCreated(runId: string) {
  wizardState.runId = runId;
}
</script>
