<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('admin.users.eyebrow') }}</p>
        <h1>{{ t('admin.users.title') }}</h1>
      </div>
      <div class="head-actions">
        <button class="btn accent sm" type="button" @click="openCreate">
          <Icon name="plus" :size="15" /> {{ t('admin.users.newUser') }}
        </button>
      </div>
    </header>

    <nav class="admin-tabs" :aria-label="t('admin.sectionsLabel')" style="display:flex; gap:18px; margin: -6px 0 14px; border-bottom:1px solid var(--border)">
      <a
        v-for="tab in tabs"
        :key="tab.href"
        :href="tab.href"
        :style="tabStyle(tab.key === 'users')"
      >{{ tab.label }}</a>
    </nav>

    <section v-if="createOpen" class="card pad" style="margin-bottom:14px">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
        <h2 style="margin:0; font-size:1.05rem">{{ t('admin.users.createUser') }}</h2>
        <button class="btn ghost sm" type="button" @click="closeCreate">
          <Icon name="x" :size="14" /> {{ t('admin.users.close') }}
        </button>
      </div>

      <div v-if="displayFormError" role="alert" :style="bannerStyle">{{ displayFormError }}</div>

      <form class="stack" style="display:grid; gap:12px" @submit.prevent="onCreate">
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:12px">
          <div class="field">
            <label class="label" for="new-username">{{ t('admin.users.username') }}</label>
            <input id="new-username" v-model="createForm.username" type="text" required autocomplete="off" />
          </div>
          <div class="field">
            <label class="label" for="new-email">{{ t('admin.users.email') }} <span class="faint">({{ t('admin.users.optional') }})</span></label>
            <input id="new-email" v-model="createForm.email" type="text" autocomplete="off" />
          </div>
          <div class="field">
            <label class="label" for="new-password">{{ t('admin.users.password') }}</label>
            <input id="new-password" v-model="createForm.password" type="password" required autocomplete="new-password" />
            <p class="field-help">{{ t('admin.users.minPassword') }}</p>
          </div>
          <div class="field">
            <label class="label" for="new-role">{{ t('admin.users.role') }}</label>
            <select id="new-role" v-model="createForm.role_id">
              <option v-for="r in roles" :key="r.role_id" :value="r.role_id">{{ r.name }}</option>
            </select>
          </div>
        </div>
        <label style="display:flex; align-items:center; gap:8px">
          <input v-model="createForm.is_active" type="checkbox" style="width:auto; min-height:0" />
          <span>{{ t('admin.users.active') }}</span>
        </label>
        <div class="head-actions" style="justify-content:flex-end">
          <button class="btn ghost sm" type="button" @click="closeCreate">{{ t('admin.users.cancel') }}</button>
          <button class="btn accent sm" type="submit" :disabled="busy">{{ busy ? t('admin.users.creating') : t('admin.users.createUser') }}</button>
        </div>
      </form>
    </section>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="displayLoadError"
        :empty="!loading && !displayLoadError && users.length === 0"
        empty-icon="users"
        :empty-title="t('admin.users.noUsers')"
        :empty-desc="t('admin.users.noUsersDesc')"
        @retry="refresh"
      >
        <div class="table-wrap">
          <table class="data">
            <thead>
              <tr>
                <th>{{ t('admin.users.username') }}</th>
                <th>{{ t('admin.users.email') }}</th>
                <th>{{ t('admin.users.role') }}</th>
                <th>{{ t('admin.users.status') }}</th>
                <th style="width:48px" :aria-label="t('admin.users.actions')"></th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="u in users"
                :key="u.user_id"
                :class="{ 'is-selected': selectedId === u.user_id }"
                style="cursor:pointer"
                @click="selectUser(u)"
              >
                <td>
                  <span style="font-weight:600">{{ u.username }}</span>
                  <span v-if="u.is_superuser" class="badge sm primary" style="margin-left:6px">{{ t('admin.users.superuser') }}</span>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(u.user_id) }}</div>
                </td>
                <td class="muted">{{ u.email || t('common.notAvailable') }}</td>
                <td class="muted">{{ u.role_names.join(', ') || t('common.notAvailable') }}</td>
                <td>
                  <span class="badge sm" :class="u.is_active ? 'success' : 'neutral'">
                    {{ u.is_active ? t('admin.users.active') : t('admin.users.disabled') }}
                  </span>
                </td>
                <td>
                  <button
                    class="icon-btn"
                    type="button"
                    :aria-label="t('admin.users.editUser')"
                    @click.stop="selectUser(u)"
                  >
                    <Icon name="settings" :size="16" />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </StateBlock>
    </section>

    <section v-if="selected" class="card pad" style="margin-top:16px">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px">
        <div>
          <h2 style="margin:0; font-size:1.05rem">{{ t('admin.users.editUserTitle', { username: selected.username }) }}</h2>
          <p class="faint" style="margin:4px 0 0; font-size:0.78rem">
            {{ t('admin.users.createdAndId', { created: formatDateTime(selected.created_at), id: shortId(selected.user_id) }) }}
          </p>
        </div>
        <button class="btn ghost sm" type="button" @click="closeEditor">
          <Icon name="x" :size="14" /> {{ t('admin.users.close') }}
        </button>
      </div>

      <div v-if="displayEditError" role="alert" :style="bannerStyle">{{ displayEditError }}</div>
      <div v-if="displayEditInfo" role="status" :style="successBannerStyle">{{ displayEditInfo }}</div>

      <div style="display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; margin-top:6px">
        <div class="field">
          <span class="label">{{ t('admin.users.username') }}</span>
          <input type="text" :value="selected.username" disabled />
          <p class="field-help">{{ t('admin.users.usernameLocked') }}</p>
        </div>
        <div class="field">
          <label class="label" for="edit-email">{{ t('admin.users.email') }}</label>
          <input id="edit-email" v-model="editForm.email" type="text" />
        </div>
        <div class="field">
          <label class="label" for="edit-role">{{ t('admin.users.role') }}</label>
          <select id="edit-role" v-model="editForm.role_id">
            <option v-for="r in roles" :key="r.role_id" :value="r.role_id">{{ r.name }}</option>
          </select>
          <p v-if="selected.role_ids.length > 1" class="field-help">
            {{ t('admin.users.multipleRoles') }}
          </p>
        </div>
        <div class="field">
          <span class="label">{{ t('admin.users.active') }}</span>
          <label style="display:flex; align-items:center; gap:8px; min-height:40px">
            <input v-model="editForm.is_active" type="checkbox" style="width:auto; min-height:0" />
            <span class="muted">{{ t('admin.users.userCanSignIn') }}</span>
          </label>
        </div>
      </div>

      <div class="head-actions" style="margin-top:16px; justify-content:space-between">
        <div style="display:flex; gap:8px; flex-wrap:wrap">
          <button class="btn sm" type="button" :disabled="busy" @click="onResetPassword">
            <Icon name="key" :size="14" /> {{ t('admin.users.resetPassword') }}
          </button>
          <button
            class="btn danger sm"
            type="button"
            :disabled="busy || isSelf"
            :title="isSelf ? t('admin.users.cannotDeleteSelf') : undefined"
            @click="onDelete"
          >
            <Icon name="ban" :size="14" /> {{ t('admin.users.deleteUser') }}
          </button>
        </div>
        <button class="btn accent sm" type="button" :disabled="busy" @click="onSave">
          {{ busy ? t('admin.users.saving') : t('admin.users.saveChanges') }}
        </button>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import Icon from '../../components/ui/Icon.vue';
import StateBlock from '../../components/ui/StateBlock.vue';
import {
  listUsers,
  listRoles,
  createUser,
  updateUser,
  setUserRoles,
  deleteUser,
  resetUserPassword,
  type UserDTO,
  type RoleDTO
} from '../../api/auth';
import { authState } from '../../stores/auth';
import { useFormat } from '../../composables/useFormat';
import { ApiError } from '../../api/client';
import { shortId } from '../../lib/format';

type MessageState = { key: string; params?: Record<string, unknown> } | { raw: string } | null;

const { t, te } = useI18n();
const { formatDateTime } = useFormat();

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

const bannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--danger-soft); color:var(--danger-text); border:1px solid var(--danger-border); margin-bottom:12px';
const successBannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--success-soft); color:var(--success-text); border:1px solid var(--success-border); margin-bottom:12px';

const users = ref<UserDTO[]>([]);
const roles = ref<RoleDTO[]>([]);
const loading = ref(true);
const loadError = ref<MessageState>(null);
const displayLoadError = computed(() => renderMessage(loadError.value));

const selectedId = ref<string | null>(null);
const selected = computed<UserDTO | null>(() =>
  selectedId.value ? users.value.find((u) => u.user_id === selectedId.value) ?? null : null
);
const isSelf = computed(() => !!selected.value && authState.user?.user_id === selected.value.user_id);

const editForm = reactive<{ email: string; is_active: boolean; role_id: string }>({
  email: '',
  is_active: true,
  role_id: ''
});
const editError = ref<MessageState>(null);
const editInfo = ref<MessageState>(null);
const displayEditError = computed(() => renderMessage(editError.value));
const displayEditInfo = computed(() => renderMessage(editInfo.value));

const createOpen = ref(false);
const createForm = reactive<{ username: string; email: string; password: string; role_id: string; is_active: boolean }>({
  username: '',
  email: '',
  password: '',
  role_id: '',
  is_active: true
});
const formError = ref<MessageState>(null);
const displayFormError = computed(() => renderMessage(formError.value));

const busy = ref(false);

onMounted(refresh);

async function refresh(): Promise<void> {
  loading.value = true;
  loadError.value = null;
  try {
    const [u, r] = await Promise.all([listUsers(), listRoles()]);
    users.value = u.items;
    roles.value = r.items;
    if (!createForm.role_id) {
      const viewer = roles.value.find((x) => x.name === 'viewer');
      createForm.role_id = (viewer ?? roles.value[0])?.role_id ?? '';
    }
    if (selectedId.value) {
      const still = users.value.find((u) => u.user_id === selectedId.value);
      if (still) hydrateEditForm(still);
      else selectedId.value = null;
    }
  } catch (e) {
    loadError.value = errorMessage(e, 'admin.users.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}

function hydrateEditForm(u: UserDTO): void {
  editForm.email = u.email ?? '';
  editForm.is_active = u.is_active;
  editForm.role_id = u.role_ids[0] ?? (roles.value.find((r) => r.name === 'viewer')?.role_id ?? roles.value[0]?.role_id ?? '');
}

function selectUser(u: UserDTO): void {
  selectedId.value = u.user_id;
  hydrateEditForm(u);
  editError.value = null;
  editInfo.value = null;
}

function closeEditor(): void {
  selectedId.value = null;
  editError.value = null;
  editInfo.value = null;
}

function openCreate(): void {
  createOpen.value = true;
  formError.value = null;
  createForm.username = '';
  createForm.email = '';
  createForm.password = '';
  createForm.is_active = true;
  const viewer = roles.value.find((r) => r.name === 'viewer');
  createForm.role_id = (viewer ?? roles.value[0])?.role_id ?? '';
}

function closeCreate(): void {
  createOpen.value = false;
  formError.value = null;
}

async function onCreate(): Promise<void> {
  if (!createForm.username.trim() || createForm.password.length < 6) {
    formError.value = { key: 'admin.users.errors.invalidCreate' };
    return;
  }
  busy.value = true;
  formError.value = null;
  try {
    await createUser({
      username: createForm.username.trim(),
      password: createForm.password,
      email: createForm.email.trim() || null,
      role_ids: createForm.role_id ? [createForm.role_id] : [],
      is_active: createForm.is_active
    });
    closeCreate();
    await refresh();
  } catch (e) {
    formError.value = errorMessage(e, 'admin.users.errors.failedToCreate');
  } finally {
    busy.value = false;
  }
}

async function onSave(): Promise<void> {
  if (!selected.value) return;
  busy.value = true;
  editError.value = null;
  editInfo.value = null;
  const cur = selected.value;
  try {
    const emailChanged = (cur.email ?? '') !== editForm.email;
    const activeChanged = cur.is_active !== editForm.is_active;
    if (emailChanged || activeChanged) {
      await updateUser(cur.user_id, {
        email: emailChanged ? (editForm.email.trim() || null) : undefined,
        is_active: activeChanged ? editForm.is_active : undefined
      });
    }
    const desiredRoles = editForm.role_id ? [editForm.role_id] : [];
    const currentRoles = [...cur.role_ids].sort().join(',');
    const nextRoles = [...desiredRoles].sort().join(',');
    if (currentRoles !== nextRoles) {
      await setUserRoles(cur.user_id, desiredRoles);
    }
    editInfo.value = { key: 'admin.users.saved' };
    await refresh();
  } catch (e) {
    editError.value = errorMessage(e, 'admin.users.errors.failedToSave');
  } finally {
    busy.value = false;
  }
}

async function onResetPassword(): Promise<void> {
  if (!selected.value) return;
  const next = window.prompt(t('admin.users.resetPrompt', { username: selected.value.username }));
  if (next === null) return;
  if (next.length < 6) {
    editError.value = { key: 'admin.users.errors.passwordTooShort' };
    return;
  }
  busy.value = true;
  editError.value = null;
  editInfo.value = null;
  try {
    await resetUserPassword(selected.value.user_id, next);
    editInfo.value = { key: 'admin.users.passwordReset' };
  } catch (e) {
    editError.value = errorMessage(e, 'admin.users.errors.failedToReset');
  } finally {
    busy.value = false;
  }
}

async function onDelete(): Promise<void> {
  if (!selected.value || isSelf.value) return;
  const ok = window.confirm(t('admin.users.deleteConfirm', { username: selected.value.username }));
  if (!ok) return;
  busy.value = true;
  editError.value = null;
  try {
    await deleteUser(selected.value.user_id);
    closeEditor();
    await refresh();
  } catch (e) {
    editError.value = errorMessage(e, 'admin.users.errors.failedToDelete');
    busy.value = false;
  }
}

function renderMessage(message: MessageState): string {
  if (!message) return '';
  if ('raw' in message) return message.raw;
  return t(message.key, message.params ?? {});
}

function errorMessage(error: unknown, fallbackKey: string): MessageState {
  if (error instanceof ApiError) {
    const key = `errors.${error.error_code}`;
    if (te(key)) return { key };
    return error.message ? { raw: error.message } : { key: fallbackKey };
  }
  if (error instanceof Error && error.message) return { raw: error.message };
  return { key: fallbackKey };
}
</script>

<style scoped>
tr.is-selected {
  background: var(--surface-hover);
}
</style>
