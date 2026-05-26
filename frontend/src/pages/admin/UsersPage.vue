<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Administration</p>
        <h1>Users</h1>
      </div>
      <div class="head-actions">
        <button class="btn accent sm" type="button" @click="openCreate">
          <Icon name="plus" :size="15" /> New user
        </button>
      </div>
    </header>

    <nav class="admin-tabs" aria-label="Administration sections" style="display:flex; gap:18px; margin: -6px 0 14px; border-bottom:1px solid var(--border)">
      <a
        v-for="tab in tabs"
        :key="tab.href"
        :href="tab.href"
        :style="tabStyle(tab.key === 'users')"
      >{{ tab.label }}</a>
    </nav>

    <section v-if="createOpen" class="card pad" style="margin-bottom:14px">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
        <h2 style="margin:0; font-size:1.05rem">Create user</h2>
        <button class="btn ghost sm" type="button" @click="closeCreate">
          <Icon name="x" :size="14" /> Close
        </button>
      </div>

      <div v-if="formError" role="alert" :style="bannerStyle">{{ formError }}</div>

      <form class="stack" style="display:grid; gap:12px" @submit.prevent="onCreate">
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:12px">
          <div class="field">
            <label class="label" for="new-username">Username</label>
            <input id="new-username" v-model="createForm.username" type="text" required autocomplete="off" />
          </div>
          <div class="field">
            <label class="label" for="new-email">Email <span class="faint">(optional)</span></label>
            <input id="new-email" v-model="createForm.email" type="text" autocomplete="off" />
          </div>
          <div class="field">
            <label class="label" for="new-password">Password</label>
            <input id="new-password" v-model="createForm.password" type="password" required autocomplete="new-password" />
            <p class="field-help">Minimum 6 characters.</p>
          </div>
          <div class="field">
            <label class="label" for="new-role">Role</label>
            <select id="new-role" v-model="createForm.role_id">
              <option v-for="r in roles" :key="r.role_id" :value="r.role_id">{{ r.name }}</option>
            </select>
          </div>
        </div>
        <label style="display:flex; align-items:center; gap:8px">
          <input v-model="createForm.is_active" type="checkbox" style="width:auto; min-height:0" />
          <span>Active</span>
        </label>
        <div class="head-actions" style="justify-content:flex-end">
          <button class="btn ghost sm" type="button" @click="closeCreate">Cancel</button>
          <button class="btn accent sm" type="submit" :disabled="busy">{{ busy ? 'Creating…' : 'Create user' }}</button>
        </div>
      </form>
    </section>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="loadError"
        :empty="!loading && !loadError && users.length === 0"
        empty-icon="users"
        empty-title="No users"
        empty-desc="Use “New user” to add the first account."
        @retry="refresh"
      >
        <div class="table-wrap">
          <table class="data">
            <thead>
              <tr>
                <th>Username</th>
                <th>Email</th>
                <th>Role</th>
                <th>Status</th>
                <th style="width:48px" aria-label="Actions"></th>
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
                  <span v-if="u.is_superuser" class="badge sm primary" style="margin-left:6px">superuser</span>
                  <div class="faint mono" style="font-size:0.74rem">{{ shortId(u.user_id) }}</div>
                </td>
                <td class="muted">{{ u.email || '—' }}</td>
                <td class="muted">{{ u.role_names.join(', ') || '—' }}</td>
                <td>
                  <span class="badge sm" :class="u.is_active ? 'success' : 'neutral'">
                    {{ u.is_active ? 'Active' : 'Disabled' }}
                  </span>
                </td>
                <td>
                  <button
                    class="icon-btn"
                    type="button"
                    aria-label="Edit user"
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
          <h2 style="margin:0; font-size:1.05rem">Edit user — {{ selected.username }}</h2>
          <p class="faint" style="margin:4px 0 0; font-size:0.78rem">
            Created {{ formatDateTime(selected.created_at) }} · ID {{ shortId(selected.user_id) }}
          </p>
        </div>
        <button class="btn ghost sm" type="button" @click="closeEditor">
          <Icon name="x" :size="14" /> Close
        </button>
      </div>

      <div v-if="editError" role="alert" :style="bannerStyle">{{ editError }}</div>
      <div v-if="editInfo" role="status" :style="successBannerStyle">{{ editInfo }}</div>

      <div style="display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; margin-top:6px">
        <div class="field">
          <span class="label">Username</span>
          <input type="text" :value="selected.username" disabled />
          <p class="field-help">Username cannot be changed.</p>
        </div>
        <div class="field">
          <label class="label" for="edit-email">Email</label>
          <input id="edit-email" v-model="editForm.email" type="text" />
        </div>
        <div class="field">
          <label class="label" for="edit-role">Role</label>
          <select id="edit-role" v-model="editForm.role_id">
            <option v-for="r in roles" :key="r.role_id" :value="r.role_id">{{ r.name }}</option>
          </select>
          <p v-if="selected.role_ids.length > 1" class="field-help">
            This user has multiple roles in the backend; the UI assigns a single role. Saving will replace the role set.
          </p>
        </div>
        <div class="field">
          <span class="label">Active</span>
          <label style="display:flex; align-items:center; gap:8px; min-height:40px">
            <input v-model="editForm.is_active" type="checkbox" style="width:auto; min-height:0" />
            <span class="muted">User can sign in</span>
          </label>
        </div>
      </div>

      <div class="head-actions" style="margin-top:16px; justify-content:space-between">
        <div style="display:flex; gap:8px; flex-wrap:wrap">
          <button class="btn sm" type="button" :disabled="busy" @click="onResetPassword">
            <Icon name="key" :size="14" /> Reset password
          </button>
          <button
            class="btn danger sm"
            type="button"
            :disabled="busy || isSelf"
            :title="isSelf ? 'You cannot delete your own account' : undefined"
            @click="onDelete"
          >
            <Icon name="ban" :size="14" /> Delete user
          </button>
        </div>
        <button class="btn accent sm" type="button" :disabled="busy" @click="onSave">
          {{ busy ? 'Saving…' : 'Save changes' }}
        </button>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue';
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
import { ApiError } from '../../api/client';
import { authState } from '../../stores/auth';
import { formatDateTime, shortId } from '../../lib/format';

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

const bannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--danger-soft); color:var(--danger-text); border:1px solid var(--danger-border); margin-bottom:12px';
const successBannerStyle =
  'padding:10px 12px; border-radius:8px; background:var(--success-soft); color:var(--success-text); border:1px solid var(--success-border); margin-bottom:12px';

const users = ref<UserDTO[]>([]);
const roles = ref<RoleDTO[]>([]);
const loading = ref(true);
const loadError = ref<string | null>(null);

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
const editError = ref<string | null>(null);
const editInfo = ref<string | null>(null);

const createOpen = ref(false);
const createForm = reactive<{ username: string; email: string; password: string; role_id: string; is_active: boolean }>({
  username: '',
  email: '',
  password: '',
  role_id: '',
  is_active: true
});
const formError = ref<string | null>(null);

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
    loadError.value = errorMessage(e, 'Failed to load users.');
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
    formError.value = 'Username is required and password must be at least 6 characters.';
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
    formError.value = errorMessage(e, 'Failed to create user.');
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
    editInfo.value = 'Saved.';
    await refresh();
  } catch (e) {
    editError.value = errorMessage(e, 'Failed to save changes.');
  } finally {
    busy.value = false;
  }
}

async function onResetPassword(): Promise<void> {
  if (!selected.value) return;
  const next = window.prompt(`Set a new password for ${selected.value.username} (≥6 characters):`);
  if (next === null) return;
  if (next.length < 6) {
    editError.value = 'Password must be at least 6 characters.';
    return;
  }
  busy.value = true;
  editError.value = null;
  editInfo.value = null;
  try {
    await resetUserPassword(selected.value.user_id, next);
    editInfo.value = 'Password reset.';
  } catch (e) {
    editError.value = errorMessage(e, 'Failed to reset password.');
  } finally {
    busy.value = false;
  }
}

async function onDelete(): Promise<void> {
  if (!selected.value || isSelf.value) return;
  const ok = window.confirm(`Delete user "${selected.value.username}"? This cannot be undone.`);
  if (!ok) return;
  busy.value = true;
  editError.value = null;
  try {
    await deleteUser(selected.value.user_id);
    closeEditor();
    await refresh();
  } catch (e) {
    editError.value = errorMessage(e, 'Failed to delete user.');
    busy.value = false;
  }
}

function errorMessage(e: unknown, fallback: string): string {
  if (e instanceof ApiError) return e.message || fallback;
  if (e instanceof Error) return e.message || fallback;
  return fallback;
}
</script>

<style scoped>
tr.is-selected {
  background: var(--surface-hover);
}
</style>
