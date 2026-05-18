<template>
  <section>
    <button :disabled="!wizardState.shardId" @click="createTrack">Create track</button>
    <p v-if="wizardState.trackId">Track ready</p>
  </section>
</template>

<script setup lang="ts">
import { createRealDatasetTrack } from '../../api/tracks';
import { wizardState } from '../../stores/wizard';

async function createTrack() {
  const result = await createRealDatasetTrack({ name: 'Real dataset track', shard_ids: [wizardState.shardId], primary_metric_id: 'mse' });
  wizardState.trackId = result.track_id;
}
</script>
