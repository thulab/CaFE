import { computed, ref, type Ref } from 'vue';
import { i18n } from '../i18n';
import { messageFromError, renderMessage, type MessageState } from '../lib/errors';

// Minimal async-state helper: tracks loading/error and re-runs on demand so every
// page wires loading/empty/error consistently through <StateBlock>.
export function useAsyncData<T>(fetcher: () => Promise<T>) {
  const data = ref<T | null>(null) as Ref<T | null>;
  const loading = ref(true);
  const errorState = ref<MessageState>(null);
  const error = computed(() => renderMessage(errorState.value, translate));

  async function run() {
    loading.value = true;
    errorState.value = null;
    try {
      data.value = await fetcher();
    } catch (e) {
      errorState.value = messageFromError(e, keyExists);
    } finally {
      loading.value = false;
    }
  }

  return { data, loading, error, errorState, run };
}

function translate(key: string, params?: Record<string, unknown>) {
  return i18n.global.t(key, params ?? {});
}

function keyExists(key: string) {
  return i18n.global.te(key);
}
