<template>
  <section class="step-body">
    <a v-if="wizardState.reportId" class="button-link" :href="`#/reports/${wizardState.reportId}`">Report</a>
    <p v-else class="empty-state">A report link will appear after the run finishes.</p>
  </section>
</template>

<script setup lang="ts">
import { wizardState } from '../../stores/wizard';
</script>
