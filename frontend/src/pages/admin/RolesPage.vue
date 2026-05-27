<template>
  <main class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">{{ t('admin.roles.eyebrow') }}</p>
        <h1>{{ t('admin.roles.title') }}</h1>
      </div>
    </header>

    <nav class="admin-tabs" :aria-label="t('admin.sectionsLabel')" style="display:flex; gap:18px; margin: -6px 0 14px; border-bottom:1px solid var(--border)">
      <a
        v-for="tab in tabs"
        :key="tab.href"
        :href="tab.href"
        :style="tabStyle(tab.key === 'roles')"
      >{{ tab.label }}</a>
    </nav>

    <section class="card pad">
      <StateBlock
        :loading="loading"
        :error="loadError"
        :empty="!loading && !loadError && roles.length === 0"
        empty-icon="shield"
        :empty-title="t('admin.roles.noRoles')"
        @retry="refresh"
      >
        <div style="display:grid; grid-template-columns: 240px 1fr; gap:24px">
          <aside :aria-label="t('admin.roles.rolesLabel')">
            <ul style="list-style:none; padding:0; margin:0; display:grid; gap:4px">
              <li v-for="r in roles" :key="r.role_id">
                <button
                  type="button"
                  :style="roleItemStyle(r.role_id === selectedId)"
                  @click="selectedId = r.role_id"
                >
                  <span>{{ r.name }}</span>
                  <span v-if="r.is_system" class="badge sm neutral" style="margin-left:6px">{{ t('admin.roles.systemShort') }}</span>
                </button>
              </li>
            </ul>
          </aside>

          <section v-if="selected">
            <header style="display:flex; align-items:baseline; gap:10px; margin-bottom:6px">
              <h2 style="margin:0; font-size:1.05rem">{{ t('admin.roles.permissionsFor', { role: selected.name }) }}</h2>
              <span v-if="selected.is_system" class="badge sm neutral">{{ t('admin.roles.system') }}</span>
            </header>
            <p v-if="selected.description" class="faint" style="margin:0 0 12px">{{ selected.description }}</p>
            <p v-if="selected.is_system" class="faint" style="margin: 0 0 14px; font-size:0.86rem">
              {{ t('admin.roles.systemManaged') }}
            </p>

            <div v-if="selected.permission_codes.length === 0" class="faint">{{ t('admin.roles.noPermissions') }}</div>

            <div v-for="group in groupedPerms" :key="group.title" style="margin-bottom:14px">
              <h3 style="margin:0 0 6px; font-size:0.86rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted)">
                {{ group.title }}
              </h3>
              <ul style="list-style:none; padding:0; margin:0; display:grid; gap:4px">
                <li v-for="code in group.codes" :key="code" style="display:flex; gap:8px; align-items:baseline">
                  <span aria-hidden="true" style="color:var(--text-faint)">•</span>
                  <span class="mono" style="font-size:0.86rem">{{ code }}</span>
                  <span v-if="permDesc(code)" class="faint" style="font-size:0.84rem">— {{ permDesc(code) }}</span>
                </li>
              </ul>
            </div>
          </section>
        </div>
      </StateBlock>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import StateBlock from '../../components/ui/StateBlock.vue';
import { listRoles, listPermissions, type RoleDTO, type PermissionDTO } from '../../api/auth';
import { displayError } from '../../lib/errors';

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

function roleItemStyle(active: boolean): Record<string, string> {
  return {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    border: '1px solid ' + (active ? 'var(--primary-border)' : 'var(--border)'),
    background: active ? 'var(--primary-soft)' : 'var(--surface)',
    color: active ? 'var(--primary-text)' : 'var(--text)',
    fontWeight: active ? '650' : '550',
    cursor: 'pointer',
    textAlign: 'left'
  };
}

const roles = ref<RoleDTO[]>([]);
const permissions = ref<PermissionDTO[]>([]);
const loading = ref(true);
const loadError = ref<string | null>(null);
const selectedId = ref<string | null>(null);

const selected = computed<RoleDTO | null>(() =>
  selectedId.value ? roles.value.find((r) => r.role_id === selectedId.value) ?? null : null
);

const permMap = computed<Map<string, PermissionDTO>>(() => {
  const m = new Map<string, PermissionDTO>();
  for (const p of permissions.value) m.set(p.code, p);
  return m;
});

interface PermGroup {
  title: string;
  codes: string[];
}

const groupedPerms = computed<PermGroup[]>(() => {
  if (!selected.value) return [];
  const bins = new Map<string, string[]>();
  const order: string[] = [];
  for (const code of [...selected.value.permission_codes].sort()) {
    const seg = code.split('.')[0] || 'other';
    const title = groupTitle(seg);
    if (!bins.has(title)) {
      bins.set(title, []);
      order.push(title);
    }
    bins.get(title)!.push(code);
  }
  return order.map((title) => ({ title, codes: bins.get(title)! }));
});

function permDesc(code: string): string {
  return permMap.value.get(code)?.description ?? '';
}

function groupTitle(segment: string): string {
  if (segment === 'dataset') return t('admin.roles.groups.dataset');
  if (segment === 'run') return t('admin.roles.groups.run');
  if (segment === 'report') return t('admin.roles.groups.report');
  if (['user', 'role', 'permission', 'audit'].includes(segment)) return t('admin.roles.groups.administration');
  return cap(segment);
}

function cap(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

onMounted(refresh);

async function refresh(): Promise<void> {
  loading.value = true;
  loadError.value = null;
  try {
    const [r, p] = await Promise.all([listRoles(), listPermissions()]);
    roles.value = r.items;
    permissions.value = p.items;
    if (!selectedId.value && roles.value.length > 0) {
      selectedId.value = roles.value[0].role_id;
    }
  } catch (e) {
    loadError.value = displayError(e, t, te, 'admin.roles.errors.failedToLoad');
  } finally {
    loading.value = false;
  }
}
</script>
