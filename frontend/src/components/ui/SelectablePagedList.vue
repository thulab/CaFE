<template>
  <section class="selectable-list">
    <div class="field">
      <label class="label" :for="searchId">{{ labels.search }}</label>
      <input
        :id="searchId"
        type="search"
        :aria-label="labels.search"
        :value="search"
        :placeholder="labels.searchPlaceholder"
        @input="emitSearch"
      />
    </div>

    <div class="table-wrap">
      <table v-if="items.length" class="data selectable-data">
        <thead>
          <tr>
            <th class="select-col">
              <input type="checkbox" :aria-label="labels.togglePageSelection" :checked="allCurrentPageSelected" @change="togglePage" />
            </th>
            <th>{{ labels.item }}</th>
            <th>{{ labels.details }}</th>
            <th>{{ labels.status }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in items" :key="item.id" :class="{ 'is-selected': selectedIds.includes(item.id) }">
            <td class="select-col">
              <input
                type="checkbox"
                :aria-label="`${labels.selectItem} ${item.title}`"
                :checked="selectedIds.includes(item.id)"
                @change="toggleItem(item.id, $event)"
              />
            </td>
            <td>
              <div class="selectable-main">
                <span class="selectable-title">{{ item.title }}</span>
                <span v-if="item.description" class="faint">{{ item.description }}</span>
              </div>
            </td>
            <td>
              <div v-if="item.meta?.length" class="pill-row">
                <span v-for="meta in item.meta" :key="meta" class="badge">{{ meta }}</span>
              </div>
              <span v-else class="faint">—</span>
            </td>
            <td>
              <StatusBadge :status="item.status || 'ready'" />
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else-if="loading" class="status-line"><span class="spinner" style="vertical-align:-3px;margin-right:6px" />{{ labels.loading }}</p>
      <p v-else class="empty-state">{{ labels.empty }}</p>
    </div>

    <div class="sample-pager">
      <div class="head-actions" style="margin-left:auto">
        <button class="btn secondary sm" type="button" :disabled="offset <= 0 || loading" @click="$emit('page', previousOffset)">
          <Icon name="chevronLeft" :size="15" /> {{ labels.previousPage }}
        </button>
        <button class="btn secondary sm" type="button" :disabled="offset + limit >= total || loading" @click="$emit('page', nextOffset)">
          {{ labels.nextPage }} <Icon name="chevronRight" :size="15" />
        </button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Icon from './Icon.vue';
import StatusBadge from './StatusBadge.vue';

interface SelectablePagedListItem {
  id: string;
  title: string;
  description?: string;
  meta?: string[];
  status?: string | null;
}

interface SelectablePagedListLabels {
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
  items: SelectablePagedListItem[];
  selectedIds: string[];
  total: number;
  limit: number;
  offset: number;
  search: string;
  labels: SelectablePagedListLabels;
  loading?: boolean;
  searchId?: string;
}>(), {
  loading: false,
  searchId: 'selectable-list-search',
});

const emit = defineEmits<{
  (event: 'update:selectedIds', value: string[]): void;
  (event: 'update:search', value: string): void;
  (event: 'page', offset: number): void;
}>();

const currentIds = computed(() => props.items.map((item) => item.id));
const allCurrentPageSelected = computed(() => currentIds.value.length > 0 && currentIds.value.every((id) => props.selectedIds.includes(id)));
const previousOffset = computed(() => Math.max(0, props.offset - props.limit));
const nextOffset = computed(() => props.offset + props.limit);

function emitSearch(event: Event) {
  emit('update:search', (event.target as HTMLInputElement).value);
}

function toggleItem(id: string, event: Event) {
  const checked = (event.target as HTMLInputElement).checked;
  const next = checked
    ? [...props.selectedIds, id]
    : props.selectedIds.filter((item) => item !== id);
  emit('update:selectedIds', [...new Set(next)]);
}

function togglePage(event: Event) {
  const checked = (event.target as HTMLInputElement).checked;
  if (checked) {
    emit('update:selectedIds', [...new Set([...props.selectedIds, ...currentIds.value])]);
    return;
  }
  emit('update:selectedIds', props.selectedIds.filter((id) => !currentIds.value.includes(id)));
}
</script>
