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

      <div v-if="errorMessage" class="banner danger" role="alert" style="padding:10px 12px; border-radius:8px; background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-border)">
        {{ errorMessage }}
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
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../components/ui/Icon.vue';
import { login } from '../stores/auth';
import { ApiError } from '../api/client';
import { displayError } from '../lib/errors';

const username = ref('');
const password = ref('');
const busy = ref(false);
const errorMessage = ref<string | null>(null);
const { t, te } = useI18n();

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
      errorMessage.value = t('login.invalidCredentials');
    } else if (e instanceof ApiError) {
      errorMessage.value = displayError(e, t, te, 'login.failed');
    } else {
      errorMessage.value = displayError(e, t, te, 'login.failed');
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
</script>
