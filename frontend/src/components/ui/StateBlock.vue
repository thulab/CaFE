<template>
  <!-- Loading -->
  <div v-if="loading">
    <slot name="loading">
      <div class="state-block" aria-busy="true" aria-live="polite">
        <span class="spinner" />
        <p class="state-desc">{{ loadingText }}</p>
      </div>
    </slot>
  </div>

  <!-- Error (announced) -->
  <div v-else-if="error" class="state-block error" role="alert">
    <span class="state-icon"><Icon name="alert" :size="24" /></span>
    <p class="state-title">{{ errorTitle }}</p>
    <p class="state-desc">{{ error }}</p>
    <button class="btn secondary sm" type="button" @click="$emit('retry')">
      <Icon name="refresh" :size="15" /> Try again
    </button>
  </div>

  <!-- Empty -->
  <div v-else-if="empty" class="state-block">
    <slot name="empty">
      <span class="state-icon"><Icon :name="emptyIcon" :size="24" /></span>
      <p class="state-title">{{ emptyTitle }}</p>
      <p v-if="emptyDesc" class="state-desc">{{ emptyDesc }}</p>
      <slot name="empty-action" />
    </slot>
  </div>

  <!-- Content -->
  <slot v-else />
</template>

<script setup lang="ts">
import Icon from './Icon.vue';

withDefaults(defineProps<{
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
  loadingText: 'Loading…',
  errorTitle: 'Something went wrong',
  emptyTitle: 'Nothing here yet',
  emptyDesc: '',
  emptyIcon: 'inbox',
});

defineEmits<{ retry: [] }>();
</script>
