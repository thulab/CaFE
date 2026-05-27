<template>
  <section class="step-body">
    <div v-if="loading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.loadShardStep.loadingSamples') }}</div>
    <p v-else-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>

    <template v-else>
      <div class="grid-auto">
        <div class="stat-tile">
          <span class="stat-label">{{ t('wizard.loadShardStep.samples') }}</span>
          <span class="stat-value">{{ count }}</span>
          <span class="stat-foot">{{ t('wizard.loadShardStep.samplesCount', { count }) }}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">{{ t('wizard.loadShardStep.shard') }}</span>
          <span class="stat-value mono" style="font-size:1rem">{{ short }}</span>
          <span class="stat-foot">{{ t('wizard.loadShardStep.materializedSet') }}</span>
        </div>
      </div>
      <p class="note-success"><Icon name="checkCircle" :size="16" />{{ t('wizard.loadShardStep.readyForBenchmarking') }}</p>
    </template>

    <div class="wizard-foot" style="padding:0;border:0">
      <a v-if="wizardState.shardId" class="btn secondary sm" :href="`#/shards/${wizardState.shardId}`">
        <Icon name="layers" :size="15" /> {{ t('wizard.loadShardStep.inspectShard') }}
      </a>
      <button class="btn" type="button" :disabled="loading || !wizardState.shardId" @click="goNext">
        {{ t('wizard.loadShardStep.continue') }} <Icon name="arrowRight" :size="16" />
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { getShardSamples } from '../../api/datasets';
import { goNext, wizardState } from '../../stores/wizard';
import { shortId } from '../../lib/format';
import { displayError } from '../../lib/errors';

const { t, te } = useI18n();
const loading = ref(true);
const error = ref('');
const count = ref(0);
const short = computed(() => shortId(wizardState.shardId));

onMounted(loadSamples);
watch(() => wizardState.shardId, loadSamples);

async function loadSamples() {
  if (!wizardState.shardId) {
    loading.value = false;
    return;
  }
  loading.value = true;
  error.value = '';
  try {
    const samples = await getShardSamples(wizardState.shardId);
    count.value = samples.total ?? samples.items.length;
  } catch (e) {
    error.value = displayError(e, t, te, 'wizard.loadShardStep.errors.failedToLoadSamples');
  } finally {
    loading.value = false;
  }
}
</script>
