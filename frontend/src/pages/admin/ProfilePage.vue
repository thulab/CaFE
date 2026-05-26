<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Account</p>
        <h1>My profile</h1>
      </div>
    </header>

    <nav class="admin-tabs" aria-label="Administration sections" style="display:flex; gap:18px; margin: -6px 0 14px; border-bottom:1px solid var(--border)">
      <a
        v-for="tab in tabs"
        :key="tab.href"
        :href="tab.href"
        :style="tabStyle(tab.key === 'profile')"
      >{{ tab.label }}</a>
    </nav>

    <section v-if="user" class="card pad" style="margin-bottom:16px">
      <h2 style="margin:0 0 12px; font-size:1.05rem">Account details</h2>
      <dl style="display:grid; grid-template-columns: 140px 1fr; row-gap:10px; column-gap:16px; margin:0">
        <dt class="muted">Username</dt>
        <dd style="margin:0; font-weight:600">{{ user.username }}</dd>

        <dt class="muted">Email</dt>
        <dd style="margin:0">{{ user.email || '—' }}</dd>

        <dt class="muted">Roles</dt>
        <dd style="margin:0; display:flex; flex-wrap:wrap; gap:6px">
          <span v-if="user.roles.length === 0" class="faint">No role</span>
          <span
            v-for="r in user.roles"
            :key="r"
            class="badge sm"
            :class="r === 'admin' ? 'accent' : 'neutral'"
          >{{ r }}</span>
        </dd>

        <template v-if="user.is_superuser">
          <dt class="muted">Privileges</dt>
          <dd style="margin:0">
            <span class="badge sm primary">Superuser</span>
          </dd>
        </template>
      </dl>
    </section>

    <section class="card pad">
      <h2 style="margin:0 0 12px; font-size:1.05rem">Change password</h2>

      <div v-if="successMessage" role="status" :style="successBannerStyle">{{ successMessage }}</div>
      <div v-if="errorMessage" role="alert" :style="errorBannerStyle">{{ errorMessage }}</div>

      <form class="stack" style="display:grid; gap:12px; max-width:420px" @submit.prevent="onSubmit">
        <div class="field">
          <label class="label" for="pw-current">Current password</label>
          <input id="pw-current" v-model="form.current" type="password" autocomplete="current-password" required />
        </div>
        <div class="field">
          <label class="label" for="pw-new">New password</label>
          <input id="pw-new" v-model="form.next" type="password" autocomplete="new-password" required />
          <p class="field-help">Minimum 6 characters.</p>
        </div>
        <div class="field">
          <label class="label" for="pw-confirm">Confirm new password</label>
          <input id="pw-confirm" v-model="form.confirm" type="password" autocomplete="new-password" required />
        </div>
        <div class="head-actions" style="justify-content:flex-end">
          <button class="btn accent sm" type="submit" :disabled="busy">
            {{ busy ? 'Updating…' : 'Update password' }}
          </button>
        </div>
      </form>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from 'vue';
import { authState } from '../../stores/auth';
import { changePassword } from '../../api/auth';
import { ApiError } from '../../api/client';

const tabs = [
  { key: 'users', label: 'Users', href: '#/admin/users' },
  { key: 'roles', label: 'Roles', href: '#/admin/roles' },
  { key: 'profile', label: 'My profile', href: '#/profile' }
];

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
const successMessage = ref<string | null>(null);
const errorMessage = ref<string | null>(null);

async function onSubmit(): Promise<void> {
  successMessage.value = null;
  errorMessage.value = null;
  if (form.next.length < 6) {
    errorMessage.value = 'New password must be at least 6 characters.';
    return;
  }
  if (form.next !== form.confirm) {
    errorMessage.value = 'New password and confirmation do not match.';
    return;
  }
  busy.value = true;
  try {
    await changePassword(form.current, form.next);
    successMessage.value = 'Password updated.';
    form.current = '';
    form.next = '';
    form.confirm = '';
  } catch (e) {
    if (e instanceof ApiError) errorMessage.value = e.message || 'Failed to update password.';
    else if (e instanceof Error) errorMessage.value = e.message || 'Failed to update password.';
    else errorMessage.value = 'Failed to update password.';
  } finally {
    busy.value = false;
  }
}
</script>
