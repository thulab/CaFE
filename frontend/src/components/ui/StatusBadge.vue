<template>
  <span class="badge" :class="[variant, big ? 'lg' : '']">
    <Icon v-if="icon" :name="icon" :size="big ? 14 : 12" />
    <span v-else class="dot" />
    {{ label }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from './Icon.vue';
import { humanize } from '../../lib/format';
import { i18n } from '../../i18n';

const props = withDefaults(defineProps<{ status?: string | null; big?: boolean; label?: string }>(), {
  big: false,
});

function useTranslations() {
  try {
    const { t, te } = useI18n();
    return { t, te };
  } catch (_error) {
    return {
      t: (key: string) => i18n.global.t(key),
      te: (key: string) => i18n.global.te(key),
    };
  }
}

const { t, te } = useTranslations();

type Variant = 'success' | 'warning' | 'danger' | 'info' | 'primary' | 'neutral';

const MAP: Record<string, { variant: Variant; icon?: string }> = {
  succeeded: { variant: 'success', icon: 'checkCircle' },
  success: { variant: 'success', icon: 'checkCircle' },
  ready: { variant: 'success', icon: 'checkCircle' },
  complete: { variant: 'success', icon: 'checkCircle' },
  completed: { variant: 'success', icon: 'checkCircle' },
  ready_to_load: { variant: 'success', icon: 'check' },
  partial_succeeded: { variant: 'warning', icon: 'alert' },
  running: { variant: 'primary', icon: 'refresh' },
  in_progress: { variant: 'primary', icon: 'refresh' },
  processing: { variant: 'primary', icon: 'refresh' },
  queued: { variant: 'info', icon: 'clock' },
  pending: { variant: 'neutral', icon: 'clock' },
  idle: { variant: 'neutral', icon: 'clock' },
  failed: { variant: 'danger', icon: 'x' },
  error: { variant: 'danger', icon: 'x' },
  cancelled: { variant: 'danger', icon: 'ban' },
  canceled: { variant: 'danger', icon: 'ban' },
};

const normalized = computed(() => (props.status || '').toLowerCase());
const entry = computed(() => MAP[normalized.value] ?? { variant: 'neutral' as Variant, icon: 'info' });
const variant = computed(() => entry.value.variant);
const icon = computed(() => entry.value.icon);
const label = computed(() => {
  if (props.label) return props.label;
  const key = `status.${normalized.value}`;
  if (normalized.value && te(key)) return t(key);
  return humanize(props.status) || '—';
});
</script>
