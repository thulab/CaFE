<template>
  <section>
    <a v-if="wizardState.reportId" :href="`#/reports/${wizardState.reportId}`">Report</a>
  </section>
</template>

<script setup lang="ts">
import { wizardState } from '../../stores/wizard';
</script>
