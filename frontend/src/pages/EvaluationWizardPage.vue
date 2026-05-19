<template>
  <main class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">TSBenchmark MVP</p>
        <h1>Guided benchmark workbench</h1>
        <p class="page-subtitle">
          Load a time-series CSV, configure the forecasting split, run baseline models, and open the generated report from one workflow.
        </p>
      </div>
      <div class="header-actions" aria-label="Workflow summary">
        <span class="stat-pill">{{ datasetSummary }}</span>
        <span class="stat-pill">{{ shardSummary }}</span>
        <span class="stat-pill">{{ reportSummary }}</span>
      </div>
    </header>

    <section class="wizard-grid" aria-label="Benchmark workflow">
      <aside class="progress-panel" aria-label="Workflow progress">
        <h2 class="progress-title">Workflow</h2>
        <p class="progress-caption">Complete each step from source data to report review.</p>
        <ol class="progress-list">
          <li v-for="(step, index) in steps" :key="step.title" class="progress-step" :class="`is-${step.state}`">
            <span class="step-dot">{{ index + 1 }}</span>
            <span class="progress-copy">
              <strong>{{ step.title }}</strong>
              <span>{{ step.statusLabel }}</span>
            </span>
          </li>
        </ol>
      </aside>

      <div class="workflow-stack">
        <article v-for="(step, index) in steps" :key="step.title" class="step-card" :aria-labelledby="`step-title-${index}`">
          <header class="step-card-header">
            <span class="step-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <div>
              <p class="step-kicker">{{ step.kicker }}</p>
              <h2 :id="`step-title-${index}`" class="step-title">{{ step.title }}</h2>
              <p class="step-description">{{ step.description }}</p>
            </div>
            <span class="status-badge" :class="`is-${step.state}`">{{ step.statusLabel }}</span>
          </header>

          <component :is="step.component" />
        </article>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import UploadStep from '../components/wizard/UploadStep.vue';
import ColumnAndSplitStep from '../components/wizard/ColumnAndSplitStep.vue';
import LoadShardStep from '../components/wizard/LoadShardStep.vue';
import TrackStep from '../components/wizard/TrackStep.vue';
import RunStep from '../components/wizard/RunStep.vue';
import ResultStep from '../components/wizard/ResultStep.vue';
import { wizardState } from '../stores/wizard';

const stepDefinitions = [
  {
    title: 'Upload CSV',
    kicker: 'Data source',
    description: 'Start with a local CSV and inspect the detected columns before configuring a benchmark.',
    component: UploadStep,
    isComplete: () => Boolean(wizardState.preview)
  },
  {
    title: 'Configure split',
    kicker: 'Dataset manifest',
    description: 'Choose the time and target columns, then set context, horizon, and stride for sample generation.',
    component: ColumnAndSplitStep,
    isComplete: () => Boolean(wizardState.shardId)
  },
  {
    title: 'Confirm shard',
    kicker: 'Sample load',
    description: 'Review the shard sample count so the run starts from a known evaluation set.',
    component: LoadShardStep,
    isComplete: () => Boolean(wizardState.shardId)
  },
  {
    title: 'Create track',
    kicker: 'Benchmark target',
    description: 'Bind the loaded shard to a real-dataset track with MSE as the primary metric.',
    component: TrackStep,
    isComplete: () => Boolean(wizardState.trackId)
  },
  {
    title: 'Run models',
    kicker: 'Execution',
    description: 'Select model adapters and start the run. Progress updates until a terminal status is reached.',
    component: RunStep,
    isComplete: () => Boolean(wizardState.reportId)
  },
  {
    title: 'Open report',
    kicker: 'Results',
    description: 'Jump into the generated report when the run publishes its report identifier.',
    component: ResultStep,
    isComplete: () => Boolean(wizardState.reportId)
  }
];

const steps = computed(() => {
  const completeFlags = stepDefinitions.map((step) => step.isComplete());
  const firstIncomplete = completeFlags.findIndex((complete) => !complete);
  const currentIndex = firstIncomplete === -1 ? stepDefinitions.length - 1 : firstIncomplete;

  return stepDefinitions.map((step, index) => {
    const state = completeFlags[index] ? 'complete' : index === currentIndex ? 'current' : 'pending';
    return {
      ...step,
      state,
      statusLabel: state === 'complete' ? 'Done' : state === 'current' ? 'Now' : 'Pending'
    };
  });
});

const datasetSummary = computed(() => {
  if (!wizardState.preview) return 'No dataset loaded';
  return `${wizardState.preview.columns.length} columns detected`;
});

const shardSummary = computed(() => (wizardState.shardId ? 'Shard ready' : 'Shard pending'));
const reportSummary = computed(() => (wizardState.reportId ? 'Report ready' : 'Report pending'));
</script>
