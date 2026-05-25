<template>
  <div class="stack">
    <article class="card">
      <header class="card-head">
        <h2 class="card-title"><Icon name="barChart" :size="18" style="vertical-align:-3px;margin-right:6px" />Model metrics</h2>
        <span class="badge">{{ report.model_metrics.length }} models</span>
      </header>
      <div class="card-body">
        <div v-if="!report.model_metrics.length" class="empty-state">No model metrics recorded.</div>
        <div v-else class="table-wrap">
          <table class="data">
            <caption>Lower is better · best per metric highlighted</caption>
            <thead>
              <tr>
                <th>Model</th>
                <th v-for="key in metricKeys" :key="key" class="num">{{ key.toUpperCase() }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="m in rows" :key="m.id">
                <td>
                  <span style="font-weight:600">{{ m.name }}</span>
                  <div v-if="m.name !== m.id" class="faint mono" style="font-size:0.74rem">{{ shortId(m.id) }}</div>
                </td>
                <td v-for="key in metricKeys" :key="key" class="num">
                  <span :style="m.best[key] ? 'font-weight:700;color:var(--success-text)' : ''">{{ formatNumber(m.metrics[key]) }}</span>
                  <Icon v-if="m.best[key]" name="check" :size="12" style="margin-left:4px;color:var(--success)" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </article>

    <div class="grid-2">
      <article class="card">
        <header class="card-head"><h2 class="card-title">Task outcomes</h2><span class="badge">{{ report.task_summaries.length }}</span></header>
        <div class="card-body stack">
          <div v-if="!report.task_summaries.length" class="empty-state">No tasks recorded.</div>
          <div v-for="task in report.task_summaries" :key="String(task.task_id)" class="detail-item" style="display:grid;gap:8px">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
              <span class="mono faint" style="font-size:0.78rem">{{ shortId(String(task.task_id)) }}</span>
              <StatusBadge :status="String(task.status)" />
            </div>
            <p v-if="task.error_message" class="alert" style="padding:8px 10px">{{ task.error_message }}</p>
          </div>
        </div>
      </article>

      <article class="card">
        <header class="card-head"><h2 class="card-title">Sample forecasts</h2><span class="badge">{{ report.sample_forecast_links.length }}</span></header>
        <div class="card-body">
          <p v-if="!report.sample_forecast_links.length" class="empty-state">No per-sample forecasts available.</p>
          <nav v-else class="pill-row" aria-label="Sample forecast links">
            <a
              v-for="link in report.sample_forecast_links"
              :key="link.sample_id"
              class="btn secondary sm"
              :href="`#/samples/${link.sample_id}?run_id=${link.run_id}`"
            >
              <Icon name="lineChart" :size="14" /> {{ link.sample_id }}
            </a>
          </nav>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from '../ui/Icon.vue';
import StatusBadge from '../ui/StatusBadge.vue';
import type { ReportDTO } from '../../api/types';
import { useModels } from '../../composables/useModels';
import { formatNumber, shortId } from '../../lib/format';

const props = defineProps<{ report: ReportDTO }>();
const { modelName } = useModels();

const KNOWN_ORDER = ['mase', 'mse', 'rmse', 'mae', 'smape', 'mape'];

function metricsOf(m: Record<string, unknown>): Record<string, number> {
  const raw = (m.metrics ?? {}) as Record<string, unknown>;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    const n = typeof v === 'number' ? v : Number(v);
    if (Number.isFinite(n)) out[k] = n;
  }
  return out;
}

const metricKeys = computed(() => {
  const keys = new Set<string>();
  props.report.model_metrics.forEach((m) => Object.keys(metricsOf(m)).forEach((k) => keys.add(k)));
  return [...keys].sort((a, b) => {
    const ia = KNOWN_ORDER.indexOf(a);
    const ib = KNOWN_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
});

const bestByMetric = computed(() => {
  const best: Record<string, number> = {};
  props.report.model_metrics.forEach((m) => {
    const mm = metricsOf(m);
    metricKeys.value.forEach((k) => {
      if (k in mm && (!(k in best) || mm[k] < best[k])) best[k] = mm[k];
    });
  });
  return best;
});

const rows = computed(() =>
  props.report.model_metrics.map((m) => {
    const id = String(m.model_id ?? '');
    const metrics = metricsOf(m);
    const best: Record<string, boolean> = {};
    metricKeys.value.forEach((k) => {
      best[k] = props.report.model_metrics.length > 1 && k in metrics && metrics[k] === bestByMetric.value[k];
    });
    return { id, name: modelName(id) || id, metrics, best };
  })
);
</script>
