import { fireEvent, render, screen } from '@testing-library/vue';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { i18n, setLocale } from '../i18n';
import LoginPage from '../pages/LoginPage.vue';
import ProfilePage from '../pages/admin/ProfilePage.vue';
import UsersPage from '../pages/admin/UsersPage.vue';
import { authState } from '../stores/auth';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function installAdminFetch() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input);
    if (url === '/api/users') {
      return jsonResponse({
        items: [{
          user_id: 'user-1',
          username: 'alice',
          email: null,
          is_active: true,
          is_superuser: false,
          role_ids: ['role-viewer'],
          role_names: ['viewer'],
          created_at: '2026-05-01T00:00:00Z',
          updated_at: '2026-05-01T00:00:00Z',
        }],
      });
    }
    if (url === '/api/roles') {
      return jsonResponse({
        items: [{
          role_id: 'role-viewer',
          name: 'viewer',
          description: null,
          is_system: true,
          permission_codes: [],
          created_at: '2026-05-01T00:00:00Z',
        }],
      });
    }
    return jsonResponse({});
  });
}

describe('auth and admin transient messages', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setLocale('en-US');
    authState.user = {
      user_id: 'admin-test',
      username: 'admin',
      email: null,
      is_active: true,
      is_superuser: true,
      roles: ['admin'],
      permissions: [],
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
    authState.user = null;
  });

  it('updates a visible login invalid-credentials banner after locale changes', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      error_code: 'invalid_credentials',
      message: 'Invalid username or password.',
    }, 401));

    render(LoginPage, { global: { plugins: [i18n] } });
    await fireEvent.update(screen.getByLabelText('Username'), 'alice');
    await fireEvent.update(screen.getByLabelText('Password'), 'wrong-password');
    await fireEvent.submit(screen.getByRole('button', { name: /Sign in/ }).closest('form')!);

    expect((await screen.findByRole('alert')).textContent).toContain('Invalid username or password.');

    setLocale('zh-CN');

    expect((await screen.findByRole('alert')).textContent).toContain('用户名或密码无效。');
  });

  it('updates visible profile validation banners after locale changes', async () => {
    render(ProfilePage, { global: { plugins: [i18n] } });
    await fireEvent.update(screen.getByLabelText('New password'), 'abcdef');
    await fireEvent.update(screen.getByLabelText('Confirm new password'), 'abcdefg');
    await fireEvent.submit(screen.getByRole('button', { name: 'Update password' }).closest('form')!);

    expect((await screen.findByRole('alert')).textContent).toContain('New password and confirmation do not match.');

    setLocale('zh-CN');

    expect((await screen.findByRole('alert')).textContent).toContain('新密码和确认密码不匹配。');
  });

  it('updates visible user-admin validation banners after locale changes', async () => {
    installAdminFetch();
    vi.spyOn(window, 'prompt').mockReturnValue('123');

    render(UsersPage, { global: { plugins: [i18n] } });
    await fireEvent.click(await screen.findByText('alice'));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset password' }));

    expect((await screen.findByRole('alert')).textContent).toContain('Password must be at least 6 characters.');

    setLocale('zh-CN');

    expect((await screen.findByRole('alert')).textContent).toContain('密码必须至少 6 个字符。');
  });
});
