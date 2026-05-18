<template>
  <section>
    <label v-for="model in models" :key="model.model_id">
      <input v-model="selectedIds" type="checkbox" :value="model.model_id" />
      {{ model.name }}
    </label>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { listModels, type ModelDTO } from '../../api/models';
import { wizardState } from '../../stores/wizard';

const models = ref<ModelDTO[]>([]);
const selectedIds = ref<string[]>([]);

onMounted(async () => {
  models.value = (await listModels()).items;
});

watch(selectedIds, () => {
  wizardState.selectedModels = models.value.filter((model) => selectedIds.value.includes(model.model_id));
});
</script>
