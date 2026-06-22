<template>
  <main class="page" style="max-width: 460px; margin: 5vh auto">
    <header class="page-head" style="text-align:center; display:block">
      <p class="eyebrow">{{ t('login.eyebrow') }}</p>
      <h1>{{ t('login.title') }}</h1>
      <p class="page-sub">{{ t('login.subtitle') }}</p>
    </header>

    <form class="card pad stack" @submit.prevent="onSubmit">
      <div class="field">
        <label class="label" for="login-username">{{ t('login.username') }}</label>
        <input id="login-username" v-model="username" type="text" autocomplete="username" autofocus required />
      </div>
      <div class="field">
        <label class="label" for="login-password">{{ t('login.password') }}</label>
        <input id="login-password" v-model="password" type="password" autocomplete="current-password" required />
      </div>

      <div v-if="displayErrorMessage" class="banner danger" role="alert" style="padding:10px 12px; border-radius:8px; background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-border)">
        {{ displayErrorMessage }}
      </div>

      <div class="head-actions" style="justify-content:space-between; align-items:center">
        <a class="btn ghost sm" href="#/leaderboards">
          <Icon name="trophy" :size="14" /> {{ t('login.browseLeaderboards') }}
        </a>
        <button class="btn accent" type="submit" :disabled="busy">
          <Icon v-if="!busy" name="arrowRight" :size="16" />
          {{ busy ? t('login.signingIn') : t('login.signIn') }}
        </button>
      </div>
    </form>
  </main>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import { login } from '../stores/auth';
import { ApiError } from '../api/client';

type MessageState = { key: string; params?: Record<string, unknown> } | { raw: string } | null;

const username = ref('');
const password = ref('');
const busy = ref(false);
const errorMessage = ref<MessageState>(null);
const { t, te } = useI18n();
const displayErrorMessage = computed(() => renderMessage(errorMessage.value));

async function onSubmit() {
  errorMessage.value = null;
  busy.value = true;
  try {
    await login(username.value, password.value);
    // 登录成功后回跳：优先用 ?next=#/xxx，没有就回 #/
    const next = readNext();
    window.location.hash = next || '/';
  } catch (e) {
    if (e instanceof ApiError && e.error_code === 'invalid_credentials') {
      errorMessage.value = { key: 'login.invalidCredentials' };
    } else if (e instanceof ApiError) {
      errorMessage.value = messageFromError(e, 'login.failed');
    } else {
      errorMessage.value = messageFromError(e, 'login.failed');
    }
  } finally {
    busy.value = false;
  }
}

function readNext(): string | null {
  const hash = window.location.hash.replace(/^#/, '');
  const qIdx = hash.indexOf('?');
  if (qIdx < 0) return null;
  const params = new URLSearchParams(hash.slice(qIdx + 1));
  const next = params.get('next');
  if (!next) return null;
  // next is a fragment-style "#/somewhere", strip the leading '#'.
  try {
    const decoded = decodeURIComponent(next);
    return decoded.replace(/^#/, '');
  } catch {
    return null;
  }
}

function renderMessage(message: MessageState): string {
  if (!message) return '';
  if ('raw' in message) return message.raw;
  return t(message.key, message.params ?? {});
}

function messageFromError(error: unknown, fallbackKey: string): MessageState {
  if (error instanceof ApiError) {
    const key = `errors.${error.error_code}`;
    if (te(key)) return { key };
    return error.message ? { raw: error.message } : { key: fallbackKey };
  }
  if (error instanceof Error && error.message) return { raw: error.message };
  return { key: fallbackKey };
}
</script>
