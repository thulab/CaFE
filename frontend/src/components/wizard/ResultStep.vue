<template>
  <section class="step-body">
    <template v-if="wizardState.reportId">
      <p class="note-success"><Icon name="checkCircle" :size="16" />Run finished — the benchmark report is ready.</p>
      <div class="pill-row">
        <a class="btn lg" :href="`#/reports/${wizardState.reportId}`"><Icon name="barChart" :size="17" /> Open report</a>
        <a v-if="wizardState.trackId" class="btn secondary" :href="`#/tracks/${wizardState.trackId}/ranking`">
          <Icon name="trophy" :size="16" /> View ranking
        </a>
        <a v-if="wizardState.runId" class="btn secondary" :href="`#/runs/${wizardState.runId}`">
          <Icon name="activity" :size="16" /> Run detail
        </a>
      </div>
    </template>
    <div v-else class="state-block">
      <span class="state-icon"><Icon name="clock" :size="24" /></span>
      <p class="state-title">Waiting for results</p>
      <p class="state-desc">A report link appears here once the run reaches a terminal status.</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import Icon from '../ui/Icon.vue';
import { wizardState } from '../../stores/wizard';
</script>
