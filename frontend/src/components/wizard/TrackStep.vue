<template>
  <section class="step-body">
    <p class="field-help">
      {{ t('wizard.trackStep.description', { metric: 'MASE' }) }}
    </p>

    <p v-if="error" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ error }}</p>
    <p v-if="wizardState.trackId" class="note-success"><Icon name="checkCircle" :size="16" />{{ t('wizard.trackStep.trackReady') }}</p>
    <p v-else-if="!wizardState.shardId" class="status-line">{{ t('wizard.trackStep.loadShardFirst') }}</p>

    <div v-if="wizardState.trackId" class="pill-row">
      <a class="btn secondary sm" :href="`#/tracks/${wizardState.trackId}`"><Icon name="target" :size="15" /> {{ t('wizard.trackStep.viewTrack') }}</a>
      <a class="btn secondary sm" :href="`#/tracks/${wizardState.trackId}/ranking`"><Icon name="trophy" :size="15" /> {{ t('wizard.trackStep.viewRanking') }}</a>
    </div>

    <div class="wizard-foot" style="padding:0;border:0">
      <span class="status-line">{{ t('wizard.trackStep.primaryMetric', { metric: 'MASE' }) }}</span>
      <button class="btn" type="button" :disabled="!wizardState.shardId || busy" @click="createTrack">
        <span v-if="busy" class="spinner" /> {{ wizardState.trackId ? t('wizard.trackStep.recreateTrack') : t('wizard.trackStep.createTrack') }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { createRealDatasetTrack } from '../../api/tracks';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { goNext, wizardState } from '../../stores/wizard';

const { t } = useI18n();
const { text: error, clear: clearError, setError } = useDisplayMessage();
const busy = ref(false);

async function createTrack() {
  busy.value = true;
  clearError();
  try {
    const result = await createRealDatasetTrack({
      name: 'Real dataset track',
      shard_ids: [wizardState.shardId],
      primary_metric_id: 'mase'
    });
    wizardState.trackId = result.track_id;
    wizardState.capabilityBlockId = result.capability_block_id;
    wizardState.rankingListId = result.ranking_list_id;
    goNext();
  } catch (e) {
    setError(e, 'wizard.trackStep.errors.failedToCreateTrack');
  } finally {
    busy.value = false;
  }
}
</script>
