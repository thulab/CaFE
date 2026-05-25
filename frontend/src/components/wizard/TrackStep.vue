<template>
  <section class="step-body">
    <div class="action-row">
      <button :disabled="!wizardState.shardId" @click="createTrack">Create track</button>
      <p v-if="!wizardState.shardId" class="status-line">Load a shard before creating a track.</p>
      <p v-if="wizardState.trackId" class="success-note">Track ready</p>
    </div>
    <div v-if="wizardState.trackId" class="resource-links" aria-label="Track view links">
      <a class="text-link" :href="`#/tracks/${wizardState.trackId}`">Track</a>
      <a class="text-link" :href="`#/tracks/${wizardState.trackId}/ranking`">Ranking</a>
    </div>
  </section>
</template>

<script setup lang="ts">
import { createRealDatasetTrack } from '../../api/tracks';
import { wizardState } from '../../stores/wizard';

async function createTrack() {
  const result = await createRealDatasetTrack({ name: 'Real dataset track', shard_ids: [wizardState.shardId], primary_metric_id: 'mase' });
  wizardState.trackId = result.track_id;
  wizardState.capabilityBlockId = result.capability_block_id;
  wizardState.rankingListId = result.ranking_list_id;
}
</script>
