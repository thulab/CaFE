<template>
  <section class="stack" style="gap:8px">
    <SelectablePagedList
      :items="pageItems"
      :selected-ids="selected"
      :total="filteredColumns.length"
      :limit="limit"
      :offset="offset"
      :search="search"
      :labels="labels"
      :search-id="searchId"
      @update:selected-ids="$emit('update:selected', $event)"
      @update:search="onSearch"
      @page="offset = $event"
    />
    <span class="status-line">{{ selectedCountLabel }}</span>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import SelectablePagedList from './SelectablePagedList.vue';

interface ColumnSelectionLabels {
  search: string;
  searchPlaceholder: string;
  item: string;
  details: string;
  status: string;
  loading: string;
  empty: string;
  previousPage: string;
  nextPage: string;
  selectItem: string;
  togglePageSelection: string;
}

const props = withDefaults(defineProps<{
  columns: string[];
  selected: string[];
  labels: ColumnSelectionLabels;
  selectedCountLabel: string;
  searchId: string;
  limit?: number;
}>(), {
  limit: 8,
});

defineEmits<{
  (event: 'update:selected', value: string[]): void;
}>();

const search = ref('');
const offset = ref(0);

const filteredColumns = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return props.columns;
  return props.columns.filter((column) => column.toLowerCase().includes(q));
});

const pageItems = computed(() => filteredColumns.value.slice(offset.value, offset.value + props.limit).map((column) => ({
  id: column,
  title: column,
  status: 'ready',
})));

watch(() => props.columns, () => {
  offset.value = 0;
});

function onSearch(value: string) {
  search.value = value;
  offset.value = 0;
}
</script>
