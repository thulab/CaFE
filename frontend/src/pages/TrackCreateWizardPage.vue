<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('trackCreate.eyebrow') }}</p>
        <h1>{{ t('trackCreate.title') }}</h1>
        <p class="page-sub">{{ t('trackCreate.subtitle') }}</p>
      </div>
      <div class="head-actions">
        <a class="btn secondary sm" href="#/tracks"><Icon name="chevronLeft" :size="15" /> {{ t('trackCreate.backToTracks') }}</a>
        <button class="btn ghost sm" type="button" @click="onReset">
          <Icon name="refresh" :size="15" /> {{ t('common.reset') }}
        </button>
      </div>
    </header>

    <div class="wizard-layout">
      <aside class="wizard-aside">
        <div class="card pad">
          <p class="nav-group-label" style="margin:0 0 8px">{{ t('wizard.progress', { done: completedCount, total: steps.length }) }}</p>
          <div class="progress" style="margin-bottom:14px"><span :style="{ width: progressPct + '%' }" /></div>
          <ol class="stepper">
            <li
              v-for="(step, i) in steps"
              :key="step.title"
              :class="{ 'is-current': i === current, 'is-complete': step.complete }"
            >
              <button type="button" :disabled="!step.reachable" @click="goToStep(i)">
                <span class="step-index">
                  <Icon v-if="step.complete" name="check" :size="15" />
                  <template v-else>{{ i + 1 }}</template>
                </span>
                <span>
                  <span class="step-name">{{ step.title }}</span>
                  <span class="step-state">{{ step.stateLabel }}</span>
                </span>
              </button>
            </li>
          </ol>
        </div>

        <div v-if="artifacts.length" class="card pad">
          <p class="nav-group-label" style="margin:0 0 10px">{{ t('artifacts.createdArtifacts') }}</p>
          <ul class="artifact-list">
            <li v-for="art in artifacts" :key="art.name">
              <a :href="art.href">
                <Icon class="art-ico" :name="art.icon" :size="16" />
                <span>{{ art.name }}</span>
                <Icon name="arrowRight" :size="14" style="margin-left:auto;opacity:.5" />
              </a>
            </li>
          </ul>
        </div>
      </aside>

      <section class="card wizard-panel" aria-live="polite">
        <header class="step-card-head">
          <span class="step-badge">{{ String(current + 1).padStart(2, '0') }}</span>
          <div>
            <p class="step-kicker">{{ active.kicker }}</p>
            <h2 class="step-title">{{ active.title }}</h2>
            <p class="step-desc">{{ active.description }}</p>
          </div>
          <StatusBadge :status="active.complete ? 'complete' : 'pending'" :label="active.complete ? t('common.done') : t('common.inProgress')" />
        </header>

        <component :is="active.component" v-bind="active.props" />

        <footer class="wizard-foot">
          <button class="btn secondary" type="button" @click="goBack">
            <Icon name="chevronLeft" :size="16" /> {{ backLabel }}
          </button>
          <p class="status-line">{{ footHint }}</p>
        </footer>
      </section>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import UploadStep from '../components/wizard/UploadStep.vue';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import TrackStep from '../components/wizard/TrackStep.vue';
import TestCaseSetStep from '../components/wizard/TestCaseSetStep.vue';
import { goPrev, goToStep, resetWizard, wizardState } from '../stores/wizard';

const { t } = useI18n();

onMounted(() => {
  if (wizardState.flow !== 'track') resetWizard('track');
  wizardState.flow = 'track';
  wizardState.entryMode = 'new-track';
  if (wizardState.step >= stepDefs.value.length) goToStep(0);
});

const stepDefs = computed(() => [
  { title: t('wizard.steps.createTrack.title'), kicker: t('wizard.steps.createTrack.kicker'), description: t('wizard.steps.createTrack.description'), component: TrackStep, props: {}, complete: () => Boolean(wizardState.trackName && wizardState.primaryMetric) },
  { title: t('wizard.steps.uploadCsv.title'), kicker: t('wizard.steps.uploadCsv.kicker'), description: t('wizard.steps.uploadCsv.description'), component: UploadStep, props: {}, complete: () => Boolean(wizardState.preview || wizardState.dataUploadSkipped) },
  { title: t('wizard.steps.configureSplit.title'), kicker: t('wizard.steps.configureSplit.kicker'), description: t('wizard.steps.configureSplit.description'), component: ColumnAndSplitStep, props: {}, complete: () => Boolean(wizardState.shardId || wizardState.dataUploadSkipped) },
  { title: t('wizard.steps.selectTestCases.title'), kicker: t('wizard.steps.selectTestCases.kicker'), description: t('wizard.steps.selectTestCases.description'), component: TestCaseSetStep, props: { completion: 'open-track' }, complete: () => Boolean(wizardState.trackId) },
]);

const current = computed(() => Math.min(wizardState.step, stepDefs.value.length - 1));

const steps = computed(() => {
  const flags = stepDefs.value.map((s) => s.complete());
  return stepDefs.value.map((s, i) => {
    const complete = flags[i];
    const reachable = i === 0 || flags[i - 1];
    const stateLabel = complete ? t('common.done') : i === current.value ? t('common.current') : reachable ? t('common.ready') : t('common.locked');
    return { ...s, complete, reachable, stateLabel };
  });
});

const active = computed(() => steps.value[current.value]);
const completedCount = computed(() => steps.value.filter((s) => s.complete).length);
const progressPct = computed(() => Math.round((completedCount.value / steps.value.length) * 100));
const backLabel = computed(() => current.value === 0 ? t('trackCreate.backToTracks') : t('common.back'));

const footHint = computed(() => {
  if (active.value.complete) return t('wizard.footComplete');
  return t('wizard.footIncomplete');
});

const artifacts = computed(() => {
  const out: Array<{ name: string; href: string; icon: string }> = [];
  if (wizardState.manifestId) out.push({ name: t('artifacts.datasetManifest'), href: `#/datasets/${wizardState.manifestId}`, icon: 'database' });
  if (wizardState.loadJobId) out.push({ name: t('artifacts.loadJob'), href: `#/load-jobs/${wizardState.loadJobId}`, icon: 'file' });
  if (wizardState.shardId) out.push({ name: t('artifacts.shard'), href: `#/shards/${wizardState.shardId}`, icon: 'layers' });
  return out;
});

function onReset() {
  resetWizard('track');
  wizardState.entryMode = 'new-track';
}

function goBack() {
  if (current.value === 0) {
    window.location.hash = '#/tracks';
    return;
  }
  goPrev();
}
</script>
