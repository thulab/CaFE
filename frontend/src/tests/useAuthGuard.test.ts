import { afterEach, describe, expect, it } from 'vitest';
import { evaluateGuard } from '../composables/useAuthGuard';
import { authState, type MeDTO } from '../stores/auth';

const ADMIN: MeDTO = {
  user_id: 'u-1',
  username: 'admin',
  email: null,
  is_active: true,
  is_superuser: true,
  roles: ['admin'],
  permissions: []
};

const VIEWER: MeDTO = {
  user_id: 'u-2',
  username: 'viewer',
  email: null,
  is_active: true,
  is_superuser: false,
  roles: ['viewer'],
  permissions: ['dataset.read', 'run.read']
};

function login(user: MeDTO) {
  authState.user = user;
}
function logout() {
  authState.user = null;
}

afterEach(() => {
  logout();
});

describe('useAuthGuard.evaluateGuard', () => {
  it('public tier always allows', () => {
    logout();
    expect(evaluateGuard('public', null, '/leaderboards').redirectTo).toBeNull();
    login(VIEWER);
    expect(evaluateGuard('public', null, '/leaderboards').redirectTo).toBeNull();
  });

  it('authed tier redirects anonymous to /login with next', () => {
    logout();
    const r = evaluateGuard('authed', null, '/datasets');
    expect(r.redirectTo).toMatch(/^\/login\?next=/);
    expect(r.redirectTo).toContain(encodeURIComponent('#/datasets'));
  });

  it('authed tier allows any logged-in user', () => {
    login(VIEWER);
    expect(evaluateGuard('authed', null, '/datasets').redirectTo).toBeNull();
  });

  it('perm tier redirects to /forbidden when user lacks the code', () => {
    login(VIEWER);
    const r = evaluateGuard('perm', 'run.execute', '/new');
    expect(r.redirectTo).toBe(`/forbidden?need=${encodeURIComponent('run.execute')}`);
  });

  it('perm tier allows superuser regardless of permissions list', () => {
    login(ADMIN);
    expect(evaluateGuard('perm', 'role.manage', '/admin/roles').redirectTo).toBeNull();
  });

  it('perm tier allows when user has the exact code', () => {
    login({ ...VIEWER, permissions: ['run.execute'] });
    expect(evaluateGuard('perm', 'run.execute', '/new').redirectTo).toBeNull();
  });
});
