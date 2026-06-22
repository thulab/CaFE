<template>
  <div class="table-wrap">
    <table class="data">
      <caption>{{ t('results.sampleMetricCaption') }}</caption>
      <thead>
        <tr>
          <th>{{ t('results.model') }}</th>
          <th>{{ t('results.status') }}</th>
          <th v-for="key in metricKeys" :key="key" class="num">{{ key.toUpperCase() }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="m in rows" :key="m.model_id" :class="{ 'is-winner': m.isBest }">
          <td><span class="cell-wrap" style="font-weight:600" :title="m.name">{{ m.name }}</span></td>
          <td><StatusBadge :status="m.status" /></td>
          <td v-for="key in metricKeys" :key="key" class="num">
            <span :style="m.best[key] ? 'font-weight:700;color:var(--success-text)' : ''">{{ format(m.metrics[key]) }}</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import StatusBadge from '../ui/StatusBadge.vue';
import type { SampleForecastDTO } from '../../api/types';
import { useFormat } from '../../composables/useFormat';

const props = defineProps<{ models: SampleForecastDTO['models'] }>();
const { t } = useI18n();
const { formatNumber } = useFormat();

const KNOWN_ORDER = ['mase', 'mse', 'rmse', 'mae', 'smape', 'mape'];

const metricKeys = computed(() => {
  const keys = new Set<string>();
  props.models.forEach((m) => Object.keys(m.metrics || {}).forEach((k) => keys.add(k)));
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
  props.models.forEach((m) => {
    if (m.status !== 'succeeded') return;
    metricKeys.value.forEach((k) => {
      const v = m.metrics?.[k];
      if (typeof v === 'number' && Number.isFinite(v) && (!(k in best) || v < best[k])) best[k] = v;
    });
  });
  return best;
});

const succeededCount = computed(() => props.models.filter((m) => m.status === 'succeeded').length);

const rows = computed(() =>
  props.models.map((m) => {
    const metrics = (m.metrics || {}) as Record<string, number>;
    const best: Record<string, boolean> = {};
    metricKeys.value.forEach((k) => {
      best[k] = succeededCount.value > 1 && m.status === 'succeeded' && metrics[k] === bestByMetric.value[k];
    });
    const primary = metricKeys.value[0];
    return {
      model_id: m.model_id,
      name: m.model_name || m.model_id,
      status: m.status,
      metrics,
      best,
      isBest: Boolean(primary && best[primary])
    };
  })
);

const format = (v: unknown) => formatNumber(v);
</script>
