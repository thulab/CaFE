import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { formatDateTime, formatInt, formatNumber, timeAgo } from '../lib/format';

export function useFormat() {
  const { locale } = useI18n();
  const currentLocale = computed(() => String(locale.value));

  return {
    locale: currentLocale,
    formatNumber: (value: unknown, digits = 4) => formatNumber(value, digits, currentLocale.value),
    formatInt: (value: unknown) => formatInt(value, currentLocale.value),
    formatDateTime: (value?: string | null) => formatDateTime(value, currentLocale.value),
    timeAgo: (value?: string | null) => timeAgo(value, currentLocale.value),
  };
}
