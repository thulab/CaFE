<template>
  <div v-if="open" class="modal-backdrop" @click.self="close">
    <section class="modal-panel" role="dialog" aria-modal="true" :aria-label="title">
      <header class="card-head">
        <div>
          <p class="eyebrow">{{ t(`lifecycle.resource.${resourceType}`) }}</p>
          <h2 class="card-title">{{ title }}</h2>
        </div>
        <button class="btn ghost sm" type="button" :aria-label="t('common.cancel')" @click="close">
          <Icon name="x" :size="16" />
        </button>
      </header>
      <div class="card-body stack">
        <p class="muted">{{ description }}</p>
        <StateBlock :loading="loading" :error="error" @retry="loadImpact">
          <div class="stack">
            <p v-if="action === 'purge' && impact?.cascade_required" class="alert" role="note">
              <Icon class="alert-ico" name="alert" :size="16" />{{ t('lifecycle.cascadeWarning') }}
            </p>
            <div>
              <h3 class="compact-title">{{ t('lifecycle.affectedResources') }}</h3>
              <ul v-if="impactRows.length" class="impact-list">
                <li v-for="row in impactRows" :key="row.key">
                  <span>{{ row.label }}</span>
                  <strong>{{ formatInt(row.count) }}</strong>
                </li>
              </ul>
              <p v-else class="empty-state">{{ t('lifecycle.noAffectedResources') }}</p>
            </div>
          </div>
        </StateBlock>
        <div class="head-actions" style="justify-content:flex-end">
          <button class="btn secondary sm" type="button" @click="close">{{ t('common.cancel') }}</button>
          <button class="btn sm" :class="action === 'purge' ? 'danger' : ''" type="button" :disabled="busy || loading" @click="confirm">
            <span v-if="busy" class="spinner" />{{ confirmLabel }}
          </button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { archiveResource, getDeletionImpact, purgeResource, restoreResource, type LifecycleAction } from '../../api/lifecycle';
import type { DeletionImpactDTO, ResourceType } from '../../api/types';
import { useDisplayMessage } from '../../composables/useDisplayMessage';
import { useFormat } from '../../composables/useFormat';
import Icon from './Icon.vue';
import StateBlock from './StateBlock.vue';

const props = defineProps<{
  open: boolean;
  resourceType: ResourceType;
  resourceId: string;
  action: LifecycleAction;
}>();

const emit = defineEmits<{ (event: 'close'): void; (event: 'done'): void }>();
const { t, te } = useI18n();
const { formatInt } = useFormat();
const impact = ref<DeletionImpactDTO | null>(null);
const loading = ref(false);
const busy = ref(false);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const SINGULAR_AFFECTED_KEYS: Record<string, string> = {
  dataset_manifests: 'dataset_manifest',
  load_jobs: 'load_job',
  shards: 'shard',
  series_points: 'series_point',
  sample_indices: 'sample_index',
  capability_blocks: 'capability_block',
  tracks: 'track',
  benchmarking_runs: 'benchmarking_run',
  units: 'unit',
  tasks: 'task',
  reports: 'report',
  ranking_lists: 'ranking_list',
  ranking_entries: 'ranking_entry',
  forecast_artifacts: 'forecast_artifact',
  metric_results: 'metric_result',
  run_events: 'run_event',
};

const title = computed(() => t(`lifecycle.dialog.${props.action}.title`));
const description = computed(() => t(`lifecycle.dialog.${props.action}.description`));
const confirmLabel = computed(() => t(`lifecycle.dialog.${props.action}.confirm`));
const impactRows = computed(() => {
  const affected = impact.value?.affected ?? {};
  return Object.entries(affected)
    .filter(([, count]) => count > 0)
    .map(([key, count]) => ({ key, count, label: affectedLabel(key, count) }));
});

watch(() => [props.open, props.resourceType, props.resourceId, props.action] as const, () => {
  if (props.open) void loadImpact();
}, { immediate: true });

async function loadImpact() {
  if (!props.open || !props.resourceId) return;
  loading.value = true;
  clearError();
  try {
    impact.value = await getDeletionImpact(props.resourceType, props.resourceId);
  } catch (caught) {
    setError(caught, 'lifecycle.errors.failedToLoadImpact');
  } finally {
    loading.value = false;
  }
}

async function confirm() {
  busy.value = true;
  clearError();
  try {
    if (props.action === 'archive') await archiveResource(props.resourceType, props.resourceId);
    if (props.action === 'restore') await restoreResource(props.resourceType, props.resourceId);
    if (props.action === 'purge') await purgeResource(props.resourceType, props.resourceId, true);
    emit('done');
    close();
  } catch (caught) {
    setError(caught, `lifecycle.errors.failedTo${capitalize(props.action)}`);
  } finally {
    busy.value = false;
  }
}

function close() {
  emit('close');
}

function affectedLabel(key: string, count: number) {
  const normalized = count === 1 ? (SINGULAR_AFFECTED_KEYS[key] ?? key.replace(/s$/, '')) : key;
  const i18nKey = `lifecycle.affected.${normalized}`;
  if (te(i18nKey)) return t(i18nKey, { count: formatInt(count) });
  return `${formatInt(count)} ${key.replaceAll('_', ' ')}`;
}

function capitalize(value: string) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}
</script>
