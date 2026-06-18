<template>
  <nav class="head-actions" :aria-label="t('sampleNavigation.label')" style="gap:8px">
    <a v-if="previousSample" class="btn secondary sm" :href="hrefForSample(previousSample.sample_id, previousSample.cursor_offset)" :aria-label="t('sampleNavigation.previous')">
      <Icon name="chevronLeft" :size="15" /> {{ t('sampleNavigation.previous') }}
    </a>
    <button v-else class="btn secondary sm" type="button" disabled :aria-label="t('sampleNavigation.previous')">
      <Icon name="chevronLeft" :size="15" /> {{ t('sampleNavigation.previous') }}
    </button>
    <a v-if="nextSample" class="btn secondary sm" :href="hrefForSample(nextSample.sample_id, nextSample.cursor_offset)" :aria-label="t('sampleNavigation.next')">
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
import { getReport } from '../../api/results';
import type { SampleForecastSort, SampleIndexDTO } from '../../api/types';
import type { Ref } from 'vue';
import Icon from '../ui/Icon.vue';

type NeighborSample = SampleIndexDTO & { cursor_offset?: number };

const props = defineProps<{
  shardId?: string | null;
  sampleIndex?: number | null;
  reportId?: string | null;
  sampleCursorOffset?: number | null;
  sampleCapabilityBlockId?: string | null;
  sampleModelId?: string | null;
  sampleMetric?: string | null;
  sampleSort?: SampleForecastSort | null;
  hrefForSample: (sampleId: string, cursorOffset?: number | null) => string;
}>();

const { t } = useI18n();
const previousSample = ref<NeighborSample | null>(null);
const nextSample = ref<NeighborSample | null>(null);

watch(
  () => [
    props.shardId,
    props.sampleIndex,
    props.reportId,
    props.sampleCursorOffset,
    props.sampleCapabilityBlockId,
    props.sampleModelId,
    props.sampleMetric,
    props.sampleSort,
  ] as const,
  loadNeighbors,
  { immediate: true }
);

async function loadNeighbors() {
  previousSample.value = null;
  nextSample.value = null;
  if (props.reportId && typeof props.sampleCursorOffset === 'number' && Number.isFinite(props.sampleCursorOffset)) {
    await loadReportNeighbors(Math.max(0, props.sampleCursorOffset));
    return;
  }
  if (!props.shardId || typeof props.sampleIndex !== 'number') return;
  const requests: Array<Promise<void>> = [];
  if (props.sampleIndex > 0) {
    requests.push(loadNeighbor(props.sampleIndex - 1, previousSample));
  }
  requests.push(loadNeighbor(props.sampleIndex + 1, nextSample));
  await Promise.all(requests);
}

async function loadReportNeighbors(cursorOffset: number) {
  const requests: Array<Promise<void>> = [];
  if (cursorOffset > 0) {
    requests.push(loadReportNeighbor(cursorOffset - 1, previousSample));
  }
  requests.push(loadReportNeighbor(cursorOffset + 1, nextSample));
  await Promise.all(requests);
}

async function loadReportNeighbor(offset: number, target: Ref<NeighborSample | null>) {
  try {
    const result = await getReport(String(props.reportId), {
      sampleLinkLimit: 1,
      sampleLinkOffset: offset,
      sampleLinkCapabilityBlockId: props.sampleCapabilityBlockId || undefined,
      sampleLinkMetric: props.sampleMetric || undefined,
      sampleLinkModelId: props.sampleModelId || undefined,
      sampleLinkSort: props.sampleSort || 'sample_index',
    });
    const link = result.sample_forecast_links[0];
    target.value = link ? { ...link, cursor_offset: offset } : null;
  } catch {
    target.value = null;
  }
}

async function loadNeighbor(offset: number, target: Ref<NeighborSample | null>) {
  try {
    const result = await getShardSamples(String(props.shardId), { limit: 1, offset });
    target.value = result.items[0] ? { ...result.items[0] } : null;
  } catch {
    target.value = null;
  }
}
</script>
