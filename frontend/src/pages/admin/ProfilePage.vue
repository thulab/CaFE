<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('admin.profile.eyebrow') }}</p>
        <h1>{{ t('admin.profile.title') }}</h1>
      </div>
    </header>

    <nav class="admin-tabs" :aria-label="t('admin.sectionsLabel')" style="display:flex; gap:18px; margin: -6px 0 14px; border-bottom:1px solid var(--border)">
      <a
        v-for="tab in tabs"
        :key="tab.href"
        :href="tab.href"
        :style="tabStyle(tab.key === 'profile')"
      >{{ tab.label }}</a>
    </nav>

    <section v-if="user" class="card pad" style="margin-bottom:16px">
      <h2 style="margin:0 0 12px; font-size:1.05rem">{{ t('admin.profile.accountDetails') }}</h2>
      <dl style="display:grid; grid-template-columns: 140px 1fr; row-gap:10px; column-gap:16px; margin:0">
        <dt class="muted">{{ t('admin.profile.username') }}</dt>
        <dd style="margin:0; font-weight:600">{{ user.username }}</dd>

        <dt class="muted">{{ t('admin.profile.email') }}</dt>
        <dd style="margin:0">{{ user.email || t('common.notAvailable') }}</dd>

        <dt class="muted">{{ t('admin.profile.roles') }}</dt>
        <dd style="margin:0; display:flex; flex-wrap:wrap; gap:6px">
          <span v-if="user.roles.length === 0" class="faint">{{ t('admin.profile.noRole') }}</span>
          <span
            v-for="r in user.roles"
            :key="r"
            class="badge sm"
            :class="r === 'admin' ? 'accent' : 'neutral'"
          >{{ r }}</span>
        </dd>

        <template v-if="user.is_superuser">
          <dt class="muted">{{ t('admin.profile.privileges') }}</dt>
          <dd style="margin:0">
            <span class="badge sm primary">{{ t('admin.profile.superuser') }}</span>
          </dd>
        </template>
      </dl>
    </section>

    <section class="card pad">
      <h2 style="margin:0 0 12px; font-size:1.05rem">{{ t('admin.profile.changePassword') }}</h2>

      <div v-if="displaySuccessMessage" role="status" :style="successBannerStyle">{{ displaySuccessMessage }}</div>
      <div v-if="displayErrorMessage" role="alert" :style="errorBannerStyle">{{ displayErrorMessage }}</div>

      <form class="stack" style="display:grid; gap:12px; max-width:420px" @submit.prevent="onSubmit">
        <div class="field">
          <label class="label" for="pw-current">{{ t('admin.profile.currentPassword') }}</label>
          <input id="pw-current" v-model="form.current" type="password" autocomplete="current-password" required />
        </div>
        <div class="field">
          <label class="label" for="pw-new">{{ t('admin.profile.newPassword') }}</label>
          <input id="pw-new" v-model="form.next" type="password" autocomplete="new-password" required />
          <p class="field-help">{{ t('admin.users.minPassword') }}</p>
        </div>
        <div class="field">
          <label class="label" for="pw-confirm">{{ t('admin.profile.confirmNewPassword') }}</label>
          <input id="pw-confirm" v-model="form.confirm" type="password" autocomplete="new-password" required />
        </div>
        <div class="head-actions" style="justify-content:flex-end">
          <button class="btn accent sm" type="submit" :disabled="busy">
            {{ busy ? t('admin.profile.updating') : t('admin.profile.updatePassword') }}
          </button>
        </div>
      </form>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { authState } from '../../stores/auth';
import { changePassword } from '../../api/auth';
import { ApiError } from '../../api/client';

type MessageState = { key: string; params?: Record<string, unknown> } | { raw: string } | null;

const { t, te } = useI18n();

const tabs = computed(() => [
  { key: 'users', label: t('admin.tabs.users'), href: '#/admin/users' },
  { key: 'roles', label: t('admin.tabs.roles'), href: '#/admin/roles' },
  { key: 'profile', label: t('admin.tabs.profile'), href: '#/profile' }
]);

function tabStyle(active: boolean): Record<string, string> {
  return {
    padding: '8px 2px',
    borderBottom: active ? '2px solid var(--primary)' : '2px solid transparent',
    color: active ? 'var(--text)' : 'var(--text-muted)',
    fontWeight: active ? '700' : '550',
    textDecoration: 'none',
    fontSize: '0.92rem',
    marginBottom: '-1px'
  };
}

const errorBannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--danger-soft); color:var(--danger-text); border:1px solid var(--danger-border); margin-bottom:12px';
const successBannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--success-soft); color:var(--success-text); border:1px solid var(--success-border); margin-bottom:12px';

const user = computed(() => authState.user);

const form = reactive<{ current: string; next: string; confirm: string }>({
  current: '',
  next: '',
  confirm: ''
});

const busy = ref(false);
const successMessage = ref<MessageState>(null);
const errorMessage = ref<MessageState>(null);
const displaySuccessMessage = computed(() => renderMessage(successMessage.value));
const displayErrorMessage = computed(() => renderMessage(errorMessage.value));

async function onSubmit(): Promise<void> {
  successMessage.value = null;
  errorMessage.value = null;
  if (form.next.length < 6) {
    errorMessage.value = { key: 'admin.profile.errors.passwordTooShort' };
    return;
  }
  if (form.next !== form.confirm) {
    errorMessage.value = { key: 'admin.profile.errors.passwordMismatch' };
    return;
  }
  busy.value = true;
  try {
    await changePassword(form.current, form.next);
    successMessage.value = { key: 'admin.profile.passwordUpdated' };
    form.current = '';
    form.next = '';
    form.confirm = '';
  } catch (e) {
    errorMessage.value = messageFromError(e, 'admin.profile.errors.failedToUpdate');
  } finally {
    busy.value = false;
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
