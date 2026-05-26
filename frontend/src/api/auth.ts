import { apiRequest } from './client';

export interface UserDTO {
  user_id: string;
  username: string;
  email: string | null;
  is_active: boolean;
  is_superuser: boolean;
  role_ids: string[];
  role_names: string[];
  created_at: string;
  updated_at: string;
}

export interface RoleDTO {
  role_id: string;
  name: string;
  description: string | null;
  is_system: boolean;
  permission_codes: string[];
  created_at: string;
}

export interface PermissionDTO {
  code: string;
  description: string;
  group: string;
}

export function changePassword(old_password: string, new_password: string): Promise<{ ok: true }> {
  return apiRequest<{ ok: true }>('/auth/password', {
    method: 'POST',
    body: JSON.stringify({ old_password, new_password })
  });
}

// ---------------- users (requires user.manage) ----------------

export function listUsers(): Promise<{ items: UserDTO[] }> {
  return apiRequest<{ items: UserDTO[] }>('/users');
}

export function createUser(payload: {
  username: string;
  password: string;
  email?: string | null;
  role_ids?: string[];
  is_superuser?: boolean;
  is_active?: boolean;
}): Promise<UserDTO> {
  return apiRequest<UserDTO>('/users', { method: 'POST', body: JSON.stringify(payload) });
}

export function updateUser(userId: string, payload: { email?: string | null; is_active?: boolean }): Promise<UserDTO> {
  return apiRequest<UserDTO>(`/users/${userId}`, { method: 'PATCH', body: JSON.stringify(payload) });
}

export function deleteUser(userId: string): Promise<{ ok: true }> {
  return apiRequest<{ ok: true }>(`/users/${userId}`, { method: 'DELETE' });
}

export function setUserRoles(userId: string, role_ids: string[]): Promise<UserDTO> {
  return apiRequest<UserDTO>(`/users/${userId}/roles`, { method: 'PUT', body: JSON.stringify({ role_ids }) });
}

export function resetUserPassword(userId: string, new_password: string): Promise<{ ok: true }> {
  return apiRequest<{ ok: true }>(`/users/${userId}/reset-password`, {
    method: 'POST',
    body: JSON.stringify({ new_password })
  });
}

// ---------------- roles & permissions (requires role.manage) ----------------

export function listRoles(): Promise<{ items: RoleDTO[] }> {
  return apiRequest<{ items: RoleDTO[] }>('/roles');
}

export function listPermissions(): Promise<{ items: PermissionDTO[] }> {
  return apiRequest<{ items: PermissionDTO[] }>('/permissions');
}
