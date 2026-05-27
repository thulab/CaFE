<template>
  <main class="page" style="max-width: 600px; margin: 5vh auto">
    <header class="page-head" style="text-align:center; display:block">
      <p class="eyebrow">{{ t('forbidden.eyebrow') }}</p>
      <h1>{{ t('forbidden.title') }}</h1>
      <p class="page-sub">
        {{ t('forbidden.message') }}
        <span v-if="needCode"> {{ t('forbidden.required') }} <code class="mono">{{ needCode }}</code></span>
      </p>
    </header>
    <div class="head-actions" style="justify-content:center">
      <a class="btn secondary" href="#/">
        <Icon name="dashboard" :size="16" /> {{ t('forbidden.backToOverview') }}
      </a>
      <a class="btn ghost" href="#/leaderboards">
        <Icon name="trophy" :size="16" /> {{ t('forbidden.browseLeaderboards') }}
      </a>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
const { t } = useI18n();

const needCode = computed(() => {
  const hash = window.location.hash.replace(/^#/, '');
  const qIdx = hash.indexOf('?');
  if (qIdx < 0) return '';
  const params = new URLSearchParams(hash.slice(qIdx + 1));
  return params.get('need') || '';
});
</script>
