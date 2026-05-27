import { ref, type Ref } from 'vue';
import { i18n } from '../i18n';

// Minimal async-state helper: tracks loading/error and re-runs on demand so every
// page wires loading/empty/error consistently through <StateBlock>.
export function useAsyncData<T>(fetcher: () => Promise<T>) {
  const data = ref<T | null>(null) as Ref<T | null>;
  const loading = ref(true);
  const error = ref<string | null>(null);

  async function run() {
    loading.value = true;
    error.value = null;
    try {
      data.value = await fetcher();
    } catch (e) {
      error.value = e instanceof Error ? e.message : i18n.global.t('errors.apiError');
    } finally {
      loading.value = false;
    }
  }

  return { data, loading, error, run };
}
