<template>
  <nav class="head-actions" :aria-label="t('sampleNavigation.label')" style="gap:8px">
    <a v-if="previousSample" class="btn secondary sm" :href="hrefForSample(previousSample.sample_id)" :aria-label="t('sampleNavigation.previous')">
      <Icon name="chevronLeft" :size="15" /> {{ t('sampleNavigation.previous') }}
    </a>
    <button v-else class="btn secondary sm" type="button" disabled :aria-label="t('sampleNavigation.previous')">
      <Icon name="chevronLeft" :size="15" /> {{ t('sampleNavigation.previous') }}
    </button>
    <a v-if="nextSample" class="btn secondary sm" :href="hrefForSample(nextSample.sample_id)" :aria-label="t('sampleNavigation.next')">
      {{ t('sampleNavigation.next') }} <Icon name="chevronRight" :size="15" />
    </a>
    <button v-else class="btn secondary sm" type="button" disabled :aria-label="t('sampleNavigation.next')">
      {{ t('sampleNavigation.next') }} <Icon name="chevronRight" :size="15" />
    </button>
  </nav>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { getShardSamples } from '../../api/datasets';
import type { SampleIndexDTO } from '../../api/types';
import type { Ref } from 'vue';
import Icon from '../ui/Icon.vue';

const props = defineProps<{
  shardId?: string | null;
  sampleIndex?: number | null;
  hrefForSample: (sampleId: string) => string;
}>();

const { t } = useI18n();
const previousSample = ref<SampleIndexDTO | null>(null);
const nextSample = ref<SampleIndexDTO | null>(null);

watch(() => [props.shardId, props.sampleIndex] as const, loadNeighbors, { immediate: true });

async function loadNeighbors() {
  previousSample.value = null;
  nextSample.value = null;
  if (!props.shardId || typeof props.sampleIndex !== 'number') return;
  const requests: Array<Promise<void>> = [];
  if (props.sampleIndex > 0) {
    requests.push(loadNeighbor(props.sampleIndex - 1, previousSample));
  }
  requests.push(loadNeighbor(props.sampleIndex + 1, nextSample));
  await Promise.all(requests);
}

async function loadNeighbor(offset: number, target: Ref<SampleIndexDTO | null>) {
  try {
    const result = await getShardSamples(String(props.shardId), { limit: 1, offset });
    target.value = result.items[0] ?? null;
  } catch {
    target.value = null;
  }
}
</script>
