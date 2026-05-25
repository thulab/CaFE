<template>
  <section class="step-body">
    <p class="field-help">
      A track bundles the loaded shard into a benchmark target with a ranking list. MASE is the primary metric
      (lower is better).
    </p>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="wizardState.trackId" class="note-success"><Icon name="checkCircle" :size="16" />Track ready.</p>
    <p v-else-if="!wizardState.shardId" class="status-line">Load a shard before creating a track.</p>

    <div v-if="wizardState.trackId" class="pill-row">
      <a class="btn secondary sm" :href="`#/tracks/${wizardState.trackId}`"><Icon name="target" :size="15" /> View track</a>
      <a class="btn secondary sm" :href="`#/tracks/${wizardState.trackId}/ranking`"><Icon name="trophy" :size="15" /> View ranking</a>
    </div>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">Primary metric: MASE</span>
      <button class="btn" type="button" :disabled="!wizardState.shardId || busy" @click="createTrack">
        <span v-if="busy" class="spinner" /> {{ wizardState.trackId ? 'Recreate track' : 'Create track' }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Icon from '../ui/Icon.vue';
import { createRealDatasetTrack } from '../../api/tracks';
import { goNext, wizardState } from '../../stores/wizard';
import { recordRecent } from '../../stores/recents';

const error = ref('');
const busy = ref(false);

async function createTrack() {
  busy.value = true;
  error.value = '';
  try {
    const result = await createRealDatasetTrack({
      name: 'Real dataset track',
      shard_ids: [wizardState.shardId],
      primary_metric_id: 'mase'
    });
    wizardState.trackId = result.track_id;
    wizardState.capabilityBlockId = result.capability_block_id;
    wizardState.rankingListId = result.ranking_list_id;
    recordRecent({ kind: 'track', id: result.track_id, title: 'Real dataset track', subtitle: 'MASE primary', href: `#/tracks/${result.track_id}` });
    goNext();
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to create track';
  } finally {
    busy.value = false;
  }
}
</script>
