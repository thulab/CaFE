<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Guided workflow</p>
        <h1>New evaluation</h1>
        <p class="page-sub">
          Take a time-series CSV from upload to a published benchmark report — configure the split,
          materialize an evaluation shard, run model adapters, and review results.
        </p>
      </div>
      <div class="head-actions">
        <button class="btn ghost sm" type="button" @click="onReset">
          <Icon name="refresh" :size="15" /> Reset
        </button>
      </div>
    </header>

    <div class="wizard-layout">
      <aside class="wizard-aside">
        <div class="card pad">
          <p class="nav-group-label" style="margin:0 0 8px">Progress · {{ completedCount }}/{{ steps.length }}</p>
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
          <p class="nav-group-label" style="margin:0 0 10px">Created artifacts</p>
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
          <StatusBadge :status="active.complete ? 'complete' : 'pending'" :label="active.complete ? 'Done' : 'In progress'" />
        </header>

        <component :is="active.component" />

        <footer class="wizard-foot">
          <button class="btn secondary" type="button" :disabled="current === 0" @click="goPrev">
            <Icon name="chevronLeft" :size="16" /> Back
          </button>
          <p class="status-line">{{ footHint }}</p>
        </footer>
      </section>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../components/ui/Icon.vue';
import StatusBadge from '../components/ui/StatusBadge.vue';
import UploadStep from '../components/wizard/UploadStep.vue';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import LoadShardStep from '../components/wizard/LoadShardStep.vue';
import TrackStep from '../components/wizard/TrackStep.vue';
import RunStep from '../components/wizard/RunStep.vue';
import ResultStep from '../components/wizard/ResultStep.vue';
import { goPrev, goToStep, resetWizard, wizardState } from '../stores/wizard';

const stepDefs = [
  { title: 'Upload CSV', kicker: 'Data source', description: 'Pick a local CSV and inspect detected columns before configuring a benchmark.', component: UploadStep, complete: () => Boolean(wizardState.preview) },
  { title: 'Configure split', kicker: 'Dataset manifest', description: 'Choose the time and target columns, then set context, horizon, and stride.', component: ColumnAndSplitStep, complete: () => Boolean(wizardState.shardId) },
  { title: 'Confirm shard', kicker: 'Sample load', description: 'Review the materialized shard so the run starts from a known evaluation set.', component: LoadShardStep, complete: () => Boolean(wizardState.shardId) },
  { title: 'Create track', kicker: 'Benchmark target', description: 'Bind the shard to a real-dataset track with MASE as the primary metric.', component: TrackStep, complete: () => Boolean(wizardState.trackId) },
  { title: 'Run models', kicker: 'Execution', description: 'Select model adapters and start the run; progress updates until completion.', component: RunStep, complete: () => Boolean(wizardState.reportId) },
  { title: 'Open report', kicker: 'Results', description: 'Jump into the generated report once the run publishes its identifier.', component: ResultStep, complete: () => Boolean(wizardState.reportId) }
];

const current = computed(() => wizardState.step);

const steps = computed(() => {
  const flags = stepDefs.map((s) => s.complete());
  return stepDefs.map((s, i) => {
    const complete = flags[i];
    const reachable = i === 0 || flags[i - 1];
    const stateLabel = complete ? 'Done' : i === current.value ? 'Current' : reachable ? 'Ready' : 'Locked';
    return { ...s, complete, reachable, stateLabel };
  });
});

const active = computed(() => steps.value[current.value]);
const completedCount = computed(() => steps.value.filter((s) => s.complete).length);
const progressPct = computed(() => Math.round((completedCount.value / steps.value.length) * 100));

const footHint = computed(() => {
  if (active.value.complete) return 'Step complete — pick the next step on the left.';
  return 'Finish this step to unlock the next one.';
});

const artifacts = computed(() => {
  const out: Array<{ name: string; href: string; icon: string }> = [];
  if (wizardState.manifestId) out.push({ name: 'Dataset manifest', href: `#/datasets/${wizardState.manifestId}`, icon: 'database' });
  if (wizardState.loadJobId) out.push({ name: 'Load job', href: `#/load-jobs/${wizardState.loadJobId}`, icon: 'file' });
  if (wizardState.shardId) out.push({ name: 'Shard', href: `#/shards/${wizardState.shardId}`, icon: 'layers' });
  if (wizardState.trackId) out.push({ name: 'Track', href: `#/tracks/${wizardState.trackId}`, icon: 'target' });
  if (wizardState.runId) out.push({ name: 'Run', href: `#/runs/${wizardState.runId}`, icon: 'activity' });
  if (wizardState.reportId) out.push({ name: 'Report', href: `#/reports/${wizardState.reportId}`, icon: 'barChart' });
  return out;
});

function onReset() {
  resetWizard();
}
</script>
