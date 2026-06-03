<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('sampleWindow.eyebrow') }}</p>
        <h1>{{ t('sampleWindow.title') }}</h1>
        <p class="page-sub">{{ t('sampleWindow.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" :href="`#/shards/${shardId}`">
          <Icon name="chevronLeft" :size="15" /> {{ t('sampleWindow.backToShard') }}
        </a>
        <SampleNeighborNav
          v-if="sample"
          :shard-id="sample.shard_id || shardId"
          :sample-index="sample.sample_index"
          :href-for-sample="sampleWindowHref"
        />
        <span class="badge mono">{{ sampleTitle }}</span>
      </div>
    </header>

    <StateBlock :loading="loading" :error="error" @retry="run">
      <section v-if="sample" class="stack">
        <div class="card pad">
          <SampleWindowChart :sample="sample" />
        </div>
      </section>
    </StateBlock>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import SampleWindowChart from '../components/results/SampleWindowChart.vue';
import SampleNeighborNav from '../components/results/SampleNeighborNav.vue';
import { getSamplePreview } from '../api/samples';
import type { SamplePreviewDTO } from '../api/types';
import { useAsyncData } from '../composables/useAsync';
import { shortId } from '../lib/format';

const props = defineProps<{ shardId: string; sampleId: string }>();
const { t } = useI18n();
const { data: sample, loading, error, run } = useAsyncData<SamplePreviewDTO>(() => getSamplePreview(props.sampleId));

const sampleTitle = computed(() => {
  if (typeof sample.value?.sample_index === 'number') return t('sampleWindow.windowLabel', { index: sample.value.sample_index + 1 });
  return shortId(props.sampleId);
});

function sampleWindowHref(sampleId: string) {
  return `#/shards/${props.shardId}/samples/${encodeURIComponent(sampleId)}`;
}

onMounted(run);
</script>
