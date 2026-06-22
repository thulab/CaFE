<template>
  <a v-if="visible" class="btn accent sm" href="#/new">
    <Icon name="arrowRight" :size="15" /> {{ t('wizard.resume.continueCurrent') }}
  </a>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../ui/Icon.vue';
import { wizardMatchesResource, type WizardResourceType } from '../../stores/wizard';

const props = defineProps<{ resourceType: WizardResourceType; resourceId: string }>();
const { t } = useI18n();
const visible = computed(() => wizardMatchesResource(props.resourceType, props.resourceId));
</script>
