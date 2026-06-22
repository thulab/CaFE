import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { messageFromError, renderMessage, type MessageState } from '../lib/errors';

export function useDisplayMessage() {
  const { t, te } = useI18n();
  const state = ref<MessageState>(null);
  const text = computed(() => renderMessage(state.value, t));

  function clear() {
    state.value = null;
  }

  function setKey(key: string, params?: Record<string, unknown>) {
    state.value = { key, params };
  }

  function setRaw(raw: string) {
    state.value = { raw };
  }

  function setError(error: unknown, fallbackKey = 'errors.apiError') {
    state.value = messageFromError(error, te, fallbackKey);
  }

  return { state, text, clear, setKey, setRaw, setError };
}
