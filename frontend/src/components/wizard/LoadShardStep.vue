<template>
  <section>
    <p>{{ status }}</p>
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
