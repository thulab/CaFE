<template>
  <section class="step-body">
    <p class="status-line">{{ status }}</p>
    <div v-if="wizardState.shardId" class="resource-links" aria-label="Shard view links">
      <a class="text-link" :href="`#/shards/${wizardState.shardId}`">Shard</a>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { getShardSamples } from '../../api/datasets';
import { wizardState } from '../../stores/wizard';

const status = ref('Waiting');

onMounted(async () => {
  await loadSamples();
});

watch(() => wizardState.shardId, loadSamples);

async function loadSamples() {
  if (!wizardState.shardId) return;
  const samples = await getShardSamples(wizardState.shardId);
  status.value = `${samples.items.length} samples`;
}
</script>
