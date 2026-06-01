import { reactive, watch } from 'vue';
import type { UploadPreviewDTO } from '../api/types';
import type { MessageState } from '../lib/errors';

export type WizardEntryMode = '' | 'new-track' | 'existing-track';
export type WizardFlow = 'evaluation' | 'track';
export type WizardResourceType = 'dataset_manifest' | 'load_job' | 'shard' | 'track' | 'ranking' | 'run' | 'report' | 'sample_forecast';
export const WIZARD_STORAGE_KEY = 'tsbenchmark:wizard:draft:v1';

type WizardSnapshot = {
  flow: WizardFlow;
  entryMode: WizardEntryMode;
  step: number;
  preview: UploadPreviewDTO | null;
  dataUploadSkipped: boolean;
  sourceUri: string;
  datasetName: string;
  shardName: string;
  trackName: string;
  primaryMetric: string;
  selectedModelIds: string[];
  manifestId: string;
  loadJobId: string;
  shardId: string;
  selectedShardIds: string[];
  trackId: string;
  capabilityBlockId: string;
  rankingListId: string;
  runId: string;
  reportId: string;
};

export const wizardState = reactive({
  flow: 'evaluation' as WizardFlow,
  entryMode: '' as WizardEntryMode,
  step: 0,
  preview: null as UploadPreviewDTO | null,
  dataUploadSkipped: false,
  sourceUri: '',
  datasetName: '',
  shardName: '',
  trackName: '',
  primaryMetric: 'mase',
  selectedModelIds: [] as string[],
  manifestId: '',
  loadJobId: '',
  shardId: '',
  selectedShardIds: [] as string[],
  trackId: '',
  capabilityBlockId: '',
  rankingListId: '',
  runId: '',
  reportId: '',
  error: null as MessageState
});

export const STEP_COUNT = 6;

let suppressPersist = false;

restoreWizardDraft();

if (canUseSessionStorage()) {
  watch(
    wizardState,
    () => {
      if (!suppressPersist) persistWizardDraft();
    },
    { deep: true, flush: 'sync' }
  );
}

export function goToStep(index: number) {
  wizardState.step = Math.max(0, Math.min(STEP_COUNT - 1, index));
}

export function goNext() {
  goToStep(wizardState.step + 1);
}

export function goPrev() {
  goToStep(wizardState.step - 1);
}

export function resetWizard(flow: WizardFlow = 'evaluation') {
  suppressPersist = true;
  wizardState.flow = flow;
  wizardState.entryMode = '';
  wizardState.step = 0;
  wizardState.preview = null;
  wizardState.dataUploadSkipped = false;
  wizardState.sourceUri = '';
  wizardState.datasetName = '';
  wizardState.shardName = '';
  wizardState.trackName = '';
  wizardState.primaryMetric = 'mase';
  wizardState.selectedModelIds = [];
  wizardState.manifestId = '';
  wizardState.loadJobId = '';
  wizardState.shardId = '';
  wizardState.selectedShardIds = [];
  wizardState.trackId = '';
  wizardState.capabilityBlockId = '';
  wizardState.rankingListId = '';
  wizardState.runId = '';
  wizardState.reportId = '';
  wizardState.error = null;
  clearWizardDraft();
  suppressPersist = false;
}

export function defaultNameFromFilename(filename?: string | null) {
  const clean = (filename || '').split(/[\\/]/).pop()?.trim() || '';
  if (!clean) return 'Uploaded dataset';
  const dot = clean.lastIndexOf('.');
  return dot > 0 ? clean.slice(0, dot) : clean;
}

export function applyUploadNameDefaults(filename?: string | null) {
  const base = defaultNameFromFilename(filename);
  wizardState.datasetName = base;
  wizardState.shardName = `${base} test cases`;
}

export function hasWizardDraft() {
  return Boolean(
    wizardState.entryMode ||
    wizardState.preview ||
    wizardState.manifestId ||
    wizardState.loadJobId ||
    wizardState.shardId ||
    wizardState.selectedShardIds.length > 0 ||
    wizardState.trackId ||
    wizardState.runId ||
    wizardState.reportId
  );
}

export function hasIncompleteWizardDraft() {
  return wizardState.flow === 'evaluation' && hasWizardDraft() && !wizardState.reportId;
}

export function wizardMatchesResource(resourceType: WizardResourceType, resourceId: string) {
  if (wizardState.flow !== 'evaluation') return false;
  if (!resourceId || !hasWizardDraft()) return false;
  if (resourceType === 'dataset_manifest') return wizardState.manifestId === resourceId;
  if (resourceType === 'load_job') return wizardState.loadJobId === resourceId;
  if (resourceType === 'shard') return wizardState.shardId === resourceId;
  if (resourceType === 'track' || resourceType === 'ranking') return wizardState.trackId === resourceId;
  if (resourceType === 'run' || resourceType === 'sample_forecast') return wizardState.runId === resourceId;
  if (resourceType === 'report') return wizardState.reportId === resourceId;
  return false;
}

export function restoreWizardDraft() {
  if (!canUseSessionStorage()) return false;
  const raw = window.sessionStorage.getItem(WIZARD_STORAGE_KEY);
  if (!raw) return false;
  try {
    const parsed = JSON.parse(raw) as Partial<WizardSnapshot>;
    applySnapshot(parsed);
    return true;
  } catch {
    clearWizardDraft();
    return false;
  }
}

function persistWizardDraft() {
  if (!canUseSessionStorage()) return;
  if (!hasWizardDraft()) {
    clearWizardDraft();
    return;
  }
  window.sessionStorage.setItem(WIZARD_STORAGE_KEY, JSON.stringify(snapshot()));
}

function clearWizardDraft() {
  if (canUseSessionStorage()) window.sessionStorage.removeItem(WIZARD_STORAGE_KEY);
}

function applySnapshot(value: Partial<WizardSnapshot>) {
  suppressPersist = true;
  wizardState.flow = normalizeFlow(value.flow);
  wizardState.entryMode = normalizeEntryMode(value.entryMode);
  wizardState.step = clampStep(value.step);
  wizardState.preview = value.preview ?? null;
  wizardState.dataUploadSkipped = Boolean(value.dataUploadSkipped);
  wizardState.sourceUri = stringValue(value.sourceUri);
  wizardState.datasetName = stringValue(value.datasetName);
  wizardState.shardName = stringValue(value.shardName);
  wizardState.trackName = stringValue(value.trackName);
  wizardState.primaryMetric = stringValue(value.primaryMetric) || 'mase';
  wizardState.selectedModelIds = Array.isArray(value.selectedModelIds) ? value.selectedModelIds.filter((item) => typeof item === 'string') : [];
  wizardState.manifestId = stringValue(value.manifestId);
  wizardState.loadJobId = stringValue(value.loadJobId);
  wizardState.shardId = stringValue(value.shardId);
  wizardState.selectedShardIds = Array.isArray(value.selectedShardIds) ? value.selectedShardIds.filter((item) => typeof item === 'string') : [];
  wizardState.trackId = stringValue(value.trackId);
  wizardState.capabilityBlockId = stringValue(value.capabilityBlockId);
  wizardState.rankingListId = stringValue(value.rankingListId);
  wizardState.runId = stringValue(value.runId);
  wizardState.reportId = stringValue(value.reportId);
  wizardState.error = null;
  suppressPersist = false;
}

function snapshot(): WizardSnapshot {
  return {
    flow: wizardState.flow,
    entryMode: wizardState.entryMode,
    step: wizardState.step,
    preview: wizardState.preview,
    dataUploadSkipped: wizardState.dataUploadSkipped,
    sourceUri: wizardState.sourceUri,
    datasetName: wizardState.datasetName,
    shardName: wizardState.shardName,
    trackName: wizardState.trackName,
    primaryMetric: wizardState.primaryMetric,
    selectedModelIds: [...wizardState.selectedModelIds],
    manifestId: wizardState.manifestId,
    loadJobId: wizardState.loadJobId,
    shardId: wizardState.shardId,
    selectedShardIds: [...wizardState.selectedShardIds],
    trackId: wizardState.trackId,
    capabilityBlockId: wizardState.capabilityBlockId,
    rankingListId: wizardState.rankingListId,
    runId: wizardState.runId,
    reportId: wizardState.reportId,
  };
}

function canUseSessionStorage() {
  return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';
}

function clampStep(value: unknown) {
  return Math.max(0, Math.min(STEP_COUNT - 1, Number.isFinite(Number(value)) ? Number(value) : 0));
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value : '';
}

function normalizeEntryMode(value: unknown): WizardEntryMode {
  if (value === 'existing-track') return 'existing-track';
  if (value === 'new-track' || value === 'upload') return 'new-track';
  return '';
}

function normalizeFlow(value: unknown): WizardFlow {
  return value === 'track' ? 'track' : 'evaluation';
}
