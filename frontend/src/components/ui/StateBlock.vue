<template>
  <!-- Loading -->
  <div v-if="loading">
    <slot name="loading">
      <div class="state-block" aria-busy="true" aria-live="polite">
        <span class="spinner" />
        <p class="state-desc">{{ resolvedLoadingText }}</p>
      </div>
    </slot>
  </div>

  <!-- Error (announced) -->
  <div v-else-if="error" class="state-block error" role="alert">
    <span class="state-icon"><Icon name="alert" :size="24" /></span>
    <p class="state-title">{{ resolvedErrorTitle }}</p>
    <p class="state-desc">{{ error }}</p>
    <button class="btn secondary sm" type="button" @click="$emit('retry')">
      <Icon name="refresh" :size="15" /> {{ t('common.retry') }}
    </button>
  </div>

  <!-- Empty -->
  <div v-else-if="empty" class="state-block">
    <slot name="empty">
      <span class="state-icon"><Icon :name="emptyIcon" :size="24" /></span>
      <p class="state-title">{{ resolvedEmptyTitle }}</p>
      <p v-if="emptyDesc" class="state-desc">{{ emptyDesc }}</p>
      <slot name="empty-action" />
    </slot>
  </div>

  <!-- Content -->
  <slot v-else />
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { i18n } from '../../i18n';
import Icon from './Icon.vue';

const props = withDefaults(defineProps<{
  loading?: boolean;
  error?: string | null;
  empty?: boolean;
  loadingText?: string;
  errorTitle?: string;
  emptyTitle?: string;
  emptyDesc?: string;
  emptyIcon?: string;
}>(), {
  loading: false,
  error: null,
  empty: false,
  emptyDesc: '',
  emptyIcon: 'inbox',
});

function useTranslations() {
  try {
    const { t } = useI18n();
    return { t };
  } catch (_error) {
    return { t: (key: string) => i18n.global.t(key) };
  }
}

const { t } = useTranslations();

const resolvedLoadingText = computed(() => props.loadingText ?? t('common.loading'));
const resolvedErrorTitle = computed(() => props.errorTitle ?? t('state.somethingWentWrong'));
const resolvedEmptyTitle = computed(() => props.emptyTitle ?? t('state.nothingHereYet'));

defineEmits<{ retry: [] }>();
</script>
