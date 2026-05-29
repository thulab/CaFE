<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('nav.workspace') }}</p>
        <h1>{{ t('datasets.title') }}</h1>
        <p class="page-sub">{{ t('datasets.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" href="#/tracks"><Icon name="target" :size="15" /> {{ t('nav.tracks') }}</a>
      </div>
    </header>

    <section class="card pad" style="display:grid;gap:16px">
      <header class="card-head" style="padding:0;border:0">
        <div>
          <h2 class="card-title">{{ t('datasets.uploadTitle') }}</h2>
          <p class="muted" style="margin:4px 0 0">{{ t('datasets.uploadDesc') }}</p>
        </div>
      </header>

      <label
        class="dropzone"
        :class="{ 'is-drag': uploadDragging }"
        for="dataset-file"
        @dragover.prevent="uploadDragging = true"
        @dragleave.prevent="uploadDragging = false"
        @drop.prevent="onUploadDrop"
      >
        <Icon class="dz-icon" name="upload" :size="26" />
        <div>
          <strong>{{ uploadFileName || t('datasets.dropOrBrowse') }}</strong>
          <p class="field-help" style="margin-top:4px">{{ t('datasets.fileHelp') }}</p>
        </div>
        <input id="dataset-file" :aria-label="t('datasets.dataFile')" type="file" accept=".csv,.tsfile,text/csv,application/octet-stream" style="display:none" @change="onUploadChange" />
        <span class="btn secondary sm" style="justify-self:center;pointer-events:none">
          <Icon name="file" :size="15" /> {{ t('wizard.uploadStep.chooseFile') }}
        </span>
      </label>

      <p v-if="uploading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ t('wizard.uploadStep.parsingFile') }}</p>
      <p v-if="uploadError" class="alert" role="alert"><Icon class="alert-ico" name="alert" :size="16" />{{ uploadError }}</p>
      <p v-if="createdShardId" class="note-success">
        <Icon name="checkCircle" :size="16" />{{ t('datasets.shardReady', { id: createdShardId }) }}
        <a class="text-link" :href="`#/shards/${createdShardId}`">{{ t('datasets.openShard') }}</a>
      </p>

      <div v-if="uploadPreview" class="stack">
        <div class="pill-row">
          <span class="badge primary"><Icon name="table" :size="13" />{{ t('wizard.uploadStep.columns', { count: uploadPreview.columns.length }) }}</span>
          <span class="badge">{{ uploadFileFormat }}</span>
          <span v-if="uploadPreview.detected_delimiter" class="badge">{{ t('wizard.uploadStep.delimiter', { delimiter: uploadPreview.detected_delimiter }) }}</span>
        </div>

        <div class="grid-2">
          <div class="field">
            <label class="label" for="dataset-upload-name">{{ t('wizard.columnAndSplitStep.datasetName') }}</label>
            <input id="dataset-upload-name" v-model.trim="uploadDatasetName" />
            <p class="hint">{{ t('wizard.columnAndSplitStep.datasetNameHint') }}</p>
          </div>
          <div class="field">
            <label class="label" for="dataset-shard-name">{{ t('wizard.columnAndSplitStep.shardName') }}</label>
            <input id="dataset-shard-name" v-model.trim="uploadShardName" />
            <p class="hint">{{ t('wizard.columnAndSplitStep.shardNameHint') }}</p>
          </div>
        </div>

        <div class="grid-2">
          <div v-if="!uploadIsTsFile" class="field">
            <label class="label" for="dataset-time-column">{{ t('wizard.columnAndSplitStep.timeColumn') }}</label>
            <select id="dataset-time-column" v-model="uploadTimeColumn">
              <option v-for="column in uploadColumns" :key="column" :value="column">{{ column }}</option>
            </select>
            <p class="hint">{{ t('wizard.columnAndSplitStep.timeColumnHint') }}</p>
          </div>
          <div v-else class="field">
            <span class="label">{{ t('wizard.columnAndSplitStep.timeColumn') }}</span>
            <p class="status-line">{{ t('wizard.columnAndSplitStep.tsfileTimeColumn') }}</p>
            <p class="hint">{{ t('wizard.columnAndSplitStep.tsfileHint') }}</p>
          </div>

          <div class="field">
            <label class="label" for="dataset-target">{{ t('wizard.columnAndSplitStep.targetColumn') }}</label>
            <select id="dataset-target" v-model="uploadTarget" :aria-label="t('wizard.columnAndSplitStep.target')">
              <option value="">{{ t('wizard.columnAndSplitStep.selectTarget') }}</option>
              <option v-for="column in uploadValueColumns" :key="column" :value="column">{{ column }}</option>
            </select>
            <p class="hint">{{ t('wizard.columnAndSplitStep.targetHint') }}</p>
          </div>
        </div>

        <fieldset class="field" style="border:0;padding:0;margin:0">
          <legend class="label" style="padding:0;margin-bottom:6px">{{ t('wizard.columnAndSplitStep.valueColumns') }}</legend>
          <div class="choice-grid">
            <label v-for="column in uploadNonTimeColumns" :key="column" class="choice">
              <input v-model="uploadValueColumns" type="checkbox" :value="column" :aria-label="column" />
              {{ column }}
            </label>
          </div>
        </fieldset>

        <div class="grid-auto">
          <div class="field">
            <label class="label" for="dataset-context">{{ t('wizard.columnAndSplitStep.context') }}</label>
            <input id="dataset-context" v-model.number="uploadContext" type="number" min="1" />
          </div>
          <div class="field">
            <label class="label" for="dataset-horizon">{{ t('wizard.columnAndSplitStep.horizon') }}</label>
            <input id="dataset-horizon" v-model.number="uploadHorizon" type="number" min="1" />
          </div>
          <div class="field">
            <label class="label" for="dataset-stride">{{ t('wizard.columnAndSplitStep.stride') }}</label>
            <input id="dataset-stride" v-model.number="uploadStride" type="number" min="1" />
          </div>
          <div class="field">
            <label class="label" for="dataset-max-samples">{{ t('wizard.columnAndSplitStep.maxSamples') }}</label>
            <input id="dataset-max-samples" v-model.number="uploadMaxSamples" type="number" min="1" :placeholder="t('wizard.columnAndSplitStep.noCap')" />
          </div>
        </div>

        <div class="wizard-foot" style="padding:0;border:0">
          <span class="status-line">{{ t('wizard.columnAndSplitStep.windowStatus', { context: uploadContext, horizon: uploadHorizon, stride: uploadStride }) }}</span>
          <button class="btn" type="button" :disabled="sliceBusy" @click="createShard">
            <span v-if="sliceBusy" class="spinner" /> <Icon v-else name="layers" :size="16" /> {{ t('datasets.createShard') }}
          </button>
        </div>
      </div>
    </section>

    <section class="card pad">
      <div class="toolbar">
        <label class="choice compact">
          <input v-model="showArchived" type="checkbox" :aria-label="t('lifecycle.showArchived')" @change="load" />
          {{ t('lifecycle.showArchived') }}
        </label>
      </div>
      <StateBlock
        :loading="loading"
        :error="error || ''"
        :empty="!loading && !error && items.length === 0"
        empty-icon="database"
        :empty-title="t('datasets.noDatasets')"
        :empty-desc="t('datasets.noDatasetsDesc')"
        @retry="load"
      >
        <template #empty-action>
          <span class="status-line">{{ t('datasets.emptyUploadHint') }}</span>
        </template>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>{{ t('datasets.artifact') }}</th><th>{{ t('datasets.type') }}</th><th>{{ t('datasets.detail') }}</th><th>{{ t('datasets.created') }}</th><th>{{ t('common.actions') }}</th></tr></thead>
            <tbody>
              <tr v-for="item in displayItems" :key="`${item.kind}-${item.id}`">
                <td>
                  <a class="text-link" :href="item.href">
                    <Icon :name="kindIcon(item.kind)" :size="14" style="vertical-align:-2px;margin-right:6px" />{{ item.title }}
                  </a>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(item.id) }}</div>
                </td>
                <td>
                  <div class="pill-row">
                    <span class="badge" :class="item.kind === 'shard' ? 'primary' : ''">{{ t(`datasets.kind.${item.kind}`) }}</span>
                    <span v-if="item.archivedAt" class="badge warning">{{ t('lifecycle.archived') }}</span>
                  </div>
                </td>
                <td class="muted">{{ item.subtitle || t('common.notAvailable') }}</td>
                <td class="muted nowrap" :title="item.createdAt ? formatDateTime(item.createdAt) : ''">{{ item.createdAt ? timeAgo(item.createdAt) : t('common.notAvailable') }}</td>
                <td>
                  <div class="pill-row">
                    <button v-if="!item.archivedAt" class="btn secondary sm" type="button" @click="openLifecycle('archive', item)">{{ t('lifecycle.archive') }}</button>
                    <button v-else class="btn secondary sm" type="button" @click="openLifecycle('restore', item)">{{ t('lifecycle.restore') }}</button>
                    <button class="btn danger sm" type="button" @click="openLifecycle('purge', item)">{{ t('lifecycle.permanentDelete') }}</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>
    <ResourceActionDialog
      :open="dialog.open"
      :resource-type="dialog.resourceType"
      :resource-id="dialog.resourceId"
      :action="dialog.action"
      @close="dialog.open = false"
      @done="load"
    />
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StateBlock from '../components/ui/StateBlock.vue';
import ResourceActionDialog from '../components/ui/ResourceActionDialog.vue';
import { createDatasetManifest, createLoadJob, listDatasetManifests, listShards, uploadDataset } from '../api/datasets';
import type { ResourceType, UploadPreviewDTO } from '../api/types';
import type { LifecycleAction } from '../api/lifecycle';
import { useDisplayMessage } from '../composables/useDisplayMessage';
import { useFormat } from '../composables/useFormat';
import { shortId } from '../lib/format';
import { useI18n } from 'vue-i18n';

type Kind = 'dataset' | 'shard';
interface Row {
  kind: Kind;
  id: string;
  name?: string;
  targetColumns?: string[];
  rowCount?: number;
  title?: string;
  subtitle?: string;
  href: string;
  createdAt?: string;
  archivedAt?: string | null;
}

const items = ref<Row[]>([]);
const loading = ref(true);
const { text: error, clear: clearError, setError } = useDisplayMessage();
const { text: uploadError, clear: clearUploadError, setKey: setUploadErrorKey, setError: setUploadError } = useDisplayMessage();
const { t } = useI18n();
const { formatDateTime, formatInt, timeAgo } = useFormat();
const uploadDragging = ref(false);
const uploading = ref(false);
const sliceBusy = ref(false);
const showArchived = ref(false);
const uploadFileName = ref('');
const uploadPreview = ref<UploadPreviewDTO | null>(null);
const createdShardId = ref('');
const uploadDatasetName = ref('');
const uploadShardName = ref('');
const uploadTimeColumn = ref('time');
const uploadValueColumns = ref<string[]>([]);
const uploadTarget = ref('');
const uploadContext = ref(6);
const uploadHorizon = ref(3);
const uploadStride = ref(3);
const uploadMaxSamples = ref<number | undefined>(undefined);
const dialog = reactive<{ open: boolean; action: LifecycleAction; resourceType: ResourceType; resourceId: string }>({
  open: false,
  action: 'archive',
  resourceType: 'dataset_manifest',
  resourceId: '',
});

const uploadColumns = computed(() => uploadPreview.value?.columns.map((column) => column.name) || []);
const uploadFileFormat = computed(() => uploadPreview.value?.file_format === 'tsfile' ? 'tsfile' : 'csv');
const uploadIsTsFile = computed(() => uploadFileFormat.value === 'tsfile');
const uploadNonTimeColumns = computed(() => uploadIsTsFile.value ? uploadColumns.value : uploadColumns.value.filter((column) => column !== uploadTimeColumn.value));

const ICONS: Record<Kind, string> = { dataset: 'database', shard: 'layers' };
const kindIcon = (k: Kind) => ICONS[k] || 'file';

const displayItems = computed(() => items.value.map((item) => ({
  ...item,
  title: rowTitle(item),
  subtitle: rowSubtitle(item),
})));

watch(uploadNonTimeColumns, (cols) => {
  if (cols.length > 0 && uploadValueColumns.value.length === 0) {
    uploadValueColumns.value = [...cols];
  }
}, { immediate: true });

watch(uploadValueColumns, (cols) => {
  if (uploadTarget.value && !cols.includes(uploadTarget.value)) {
    uploadTarget.value = '';
  }
});

function rowTitle(item: Row): string {
  if (item.kind === 'dataset') return item.name ?? '';
  if (item.name) return item.name;
  return t('artifacts.shardTitle', { target: item.targetColumns?.[0] ?? t('artifacts.unknownTarget') });
}

function rowSubtitle(item: Row): string | undefined {
  if (item.kind === 'dataset') return item.subtitle;
  return shardSubtitle(item.targetColumns ?? [], item.rowCount ?? 0);
}

function shardSubtitle(targetColumns: string[], rowCount: number): string {
  const cols = targetColumns.length ? targetColumns.join(', ') : t('artifacts.shard');
  if (!rowCount) return cols;
  const key = rowCount === 1 ? 'datasets.rowCountOne' : 'datasets.rowCountOther';
  return `${cols} · ${t(key, { count: formatInt(rowCount) })}`;
}

async function load() {
  loading.value = true;
  clearError();
  try {
    const [manifests, shards] = await Promise.all([
      listDatasetManifests({ limit: 200, includeArchived: showArchived.value }),
      listShards({ limit: 200, includeArchived: showArchived.value })
    ]);
    const rows: Row[] = [];
    for (const m of manifests.items) {
      rows.push({
        kind: 'dataset',
        id: m.dataset_manifest_id,
        name: m.name,
        subtitle: m.domain,
        href: `#/datasets/${m.dataset_manifest_id}`,
        createdAt: m.created_at,
        archivedAt: m.archived_at
      });
    }
    for (const s of shards.items) {
      rows.push({
        kind: 'shard',
        id: s.shard_id,
        name: s.name ?? undefined,
        targetColumns: s.target_columns ?? [],
        rowCount: s.row_count ?? 0,
        href: `#/shards/${s.shard_id}`,
        createdAt: s.created_at,
        archivedAt: s.archived_at
      });
    }
    rows.sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
    items.value = rows;
  } catch (e) {
    setError(e, 'errors.failedToLoadDatasets');
    items.value = [];
  } finally {
    loading.value = false;
  }
}

function openLifecycle(action: LifecycleAction, item: Row) {
  dialog.action = action;
  dialog.resourceType = item.kind === 'dataset' ? 'dataset_manifest' : 'shard';
  dialog.resourceId = item.id;
  dialog.open = true;
}

async function onUploadChange(event: Event) {
  const input = event.target as HTMLInputElement;
  await handleUpload(input.files?.[0]);
}

async function onUploadDrop(event: DragEvent) {
  uploadDragging.value = false;
  await handleUpload(event.dataTransfer?.files?.[0]);
}

async function handleUpload(file?: File) {
  if (!file) return;
  uploadFileName.value = file.name;
  uploading.value = true;
  createdShardId.value = '';
  clearUploadError();
  try {
    uploadPreview.value = await uploadDataset(file);
    const baseName = baseNameFromFilename(uploadPreview.value.filename || file.name);
    uploadDatasetName.value = baseName;
    uploadShardName.value = `${baseName} shard`;
    uploadTimeColumn.value = uploadColumns.value.includes('time') ? 'time' : uploadColumns.value[0] ?? 'time';
    uploadValueColumns.value = [...uploadNonTimeColumns.value];
    uploadTarget.value = '';
  } catch (caught) {
    setUploadError(caught, 'wizard.uploadStep.errors.uploadFailed');
  } finally {
    uploading.value = false;
  }
}

async function createShard() {
  if (!uploadPreview.value) return;
  if (!uploadTarget.value || !uploadValueColumns.value.includes(uploadTarget.value)) {
    setUploadErrorKey('wizard.columnAndSplitStep.errors.selectExactlyOneTarget');
    return;
  }
  if (uploadContext.value <= 0 || uploadHorizon.value <= 0 || uploadStride.value <= 0) {
    setUploadErrorKey('wizard.columnAndSplitStep.errors.positiveSplitValues');
    return;
  }
  sliceBusy.value = true;
  clearUploadError();
  try {
    const manifest = await createDatasetManifest({
      name: uploadDatasetName.value || baseNameFromFilename(uploadPreview.value.filename || uploadFileName.value),
      domain: 'general',
      source_uri: uploadPreview.value.source_uri,
      file_format: uploadFileFormat.value,
      time_column: uploadIsTsFile.value ? 'time' : uploadTimeColumn.value,
      value_columns: uploadValueColumns.value
    });
    const splitConfig: { context_length: number; horizon: number; stride?: number; target_columns: string[]; shard_name?: string; max_samples?: number } = {
      context_length: uploadContext.value,
      horizon: uploadHorizon.value,
      stride: uploadStride.value,
      target_columns: [uploadTarget.value],
      shard_name: uploadShardName.value || `${uploadDatasetName.value || 'Uploaded dataset'} shard`
    };
    if (uploadMaxSamples.value != null && uploadMaxSamples.value > 0) splitConfig.max_samples = uploadMaxSamples.value;
    const job = await createLoadJob({ dataset_manifest_id: manifest.dataset_manifest_id, split_config: splitConfig });
    createdShardId.value = job.output_shard_id || '';
    await load();
  } catch (caught) {
    setUploadError(caught, 'wizard.columnAndSplitStep.errors.loadFailed');
  } finally {
    sliceBusy.value = false;
  }
}

function baseNameFromFilename(filename?: string | null): string {
  const clean = (filename || '').split(/[\\/]/).pop()?.trim() || '';
  if (!clean) return 'Uploaded dataset';
  const dot = clean.lastIndexOf('.');
  return dot > 0 ? clean.slice(0, dot) : clean;
}

onMounted(load);
</script>
