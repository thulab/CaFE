import { reactive } from 'vue';
import { apiRequest } from '../api/client';

// Module-level reactive auth state. Token + current user + permissions.
// Persisted under tsb.auth.token; user is rehydrated from /auth/me on bootstrap.

export interface MeDTO {
  user_id: string;
  username: string;
  email: string | null;
  is_active: boolean;
  is_superuser: boolean;
  roles: string[];
  permissions: string[];
}

const TOKEN_KEY = 'tsb.auth.token';

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage may be unavailable */
  }
}

const state = reactive<{
  token: string | null;
  user: MeDTO | null;
}>({
  token: readToken(),
  user: null
});

export const authState = state;

export function getToken(): string | null {
  return state.token;
}

export function has(code: string): boolean {
  if (!state.user) return false;
  if (state.user.is_superuser) return true;
  return state.user.permissions.includes(code);
}

interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export async function login(username: string, password: string): Promise<void> {
  const resp = await apiRequest<LoginResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password })
  });
  state.token = resp.access_token;
  writeToken(state.token);
  state.user = await apiRequest<MeDTO>('/auth/me');
}

export function logout() {
  state.token = null;
  state.user = null;
  writeToken(null);
}

/** Hydrate user state from an existing token at app boot. Invalid token => silent logout. */
export async function bootstrap(): Promise<void> {
  if (!state.token) return;
  try {
    state.user = await apiRequest<MeDTO>('/auth/me');
  } catch {
    logout();
  }
}
