<template>
  <main class="page" style="max-width: 460px; margin: 5vh auto">
    <header class="page-head" style="text-align:center; display:block">
      <p class="eyebrow">TSBenchmark</p>
      <h1>Sign in</h1>
      <p class="page-sub">Sign in to access the workbench. Anonymous visitors can still browse leaderboards.</p>
    </header>

    <form class="card pad stack" @submit.prevent="onSubmit">
      <div class="field">
        <label class="label" for="login-username">Username</label>
        <input id="login-username" v-model="username" type="text" autocomplete="username" autofocus required />
      </div>
      <div class="field">
        <label class="label" for="login-password">Password</label>
        <input id="login-password" v-model="password" type="password" autocomplete="current-password" required />
      </div>

      <div v-if="errorMessage" class="banner danger" role="alert" style="padding:10px 12px; border-radius:8px; background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-border)">
        {{ errorMessage }}
      </div>

      <div class="head-actions" style="justify-content:space-between; align-items:center">
        <a class="btn ghost sm" href="#/leaderboards">
          <Icon name="trophy" :size="14" /> Browse leaderboards
        </a>
        <button class="btn accent" type="submit" :disabled="busy">
          <Icon v-if="!busy" name="arrowRight" :size="16" />
          {{ busy ? 'Signing in…' : 'Sign in' }}
        </button>
      </div>
    </form>
  </main>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Icon from '../components/ui/Icon.vue';
import { login } from '../stores/auth';
import { ApiError } from '../api/client';

const username = ref('');
const password = ref('');
const busy = ref(false);
const errorMessage = ref<string | null>(null);

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
      errorMessage.value = 'Invalid username or password.';
    } else if (e instanceof ApiError) {
      errorMessage.value = e.message;
    } else {
      errorMessage.value = 'Sign-in failed. Please try again.';
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
