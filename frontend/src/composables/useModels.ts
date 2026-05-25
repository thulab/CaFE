import { ref } from 'vue';
import { listModels, type ModelDTO } from '../api/models';

// Shared model registry. The ranking/report payloads reference models by id only,
// so we load the catalog once and resolve display names from it.
const models = ref<ModelDTO[]>([]);
const byId = ref<Record<string, ModelDTO>>({});
let loaded = false;
let inflight: Promise<void> | null = null;

async function ensureLoaded() {
  if (loaded) return;
  if (!inflight) {
    inflight = listModels()
      .then((res) => {
        models.value = res.items;
        byId.value = Object.fromEntries(res.items.map((m) => [m.model_id, m]));
        loaded = true;
      })
      .catch(() => {
        /* leave empty; callers fall back to ids */
      })
      .finally(() => {
        inflight = null;
      });
  }
  await inflight;
}

export function useModels() {
  void ensureLoaded();
  function modelName(id?: string | null): string {
    if (!id) return '—';
    return byId.value[id]?.name ?? id;
  }
  return { models, modelName, ensureLoaded };
}
