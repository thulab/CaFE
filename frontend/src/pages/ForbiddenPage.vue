<template>
  <main class="page" style="max-width: 600px; margin: 5vh auto">
    <header class="page-head" style="text-align:center; display:block">
      <p class="eyebrow">403</p>
      <h1>Access denied</h1>
      <p class="page-sub">
        You are signed in but lack the required permission to view this page.
        <span v-if="needCode"> Required: <code class="mono">{{ needCode }}</code></span>
      </p>
    </header>
    <div class="head-actions" style="justify-content:center">
      <a class="btn secondary" href="#/">
        <Icon name="dashboard" :size="16" /> Back to overview
      </a>
      <a class="btn ghost" href="#/leaderboards">
        <Icon name="trophy" :size="16" /> Browse leaderboards
      </a>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../components/ui/Icon.vue';

const needCode = computed(() => {
  const hash = window.location.hash.replace(/^#/, '');
  const qIdx = hash.indexOf('?');
  if (qIdx < 0) return '';
  const params = new URLSearchParams(hash.slice(qIdx + 1));
  return params.get('need') || '';
});
</script>
