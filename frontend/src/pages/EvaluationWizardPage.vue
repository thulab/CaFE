<template>
  <main class="shell">
    <UploadStep />
    <ColumnAndSplitStep />
    <LoadShardStep />
    <TrackStep />
    <RunStep />
    <ResultStep />
  </main>
</template>

<script setup lang="ts">
import UploadStep from '../components/wizard/UploadStep.vue';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import LoadShardStep from '../components/wizard/LoadShardStep.vue';
import TrackStep from '../components/wizard/TrackStep.vue';
import RunStep from '../components/wizard/RunStep.vue';
import ResultStep from '../components/wizard/ResultStep.vue';
</script>

<style scoped>
.shell {
  max-width: 1120px;
  margin: 0 auto;
  padding: 24px;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}
</style>
