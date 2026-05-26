export class ApiError extends Error {
  constructor(
    public error_code: string,
    message: string,
    public details: Record<string, unknown> = {},
    public status: number = 0
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Auth integration points — assigned lazily by stores/auth.ts to avoid a circular import.
// The auth store imports apiRequest; this hook lets the client call back into the store
// for the current token and the redirect-on-401 behavior.

let getAuthToken: () => string | null = () => null;
let onUnauthorized: () => void = () => {};

export function configureAuth(opts: {
  getToken: () => string | null;
  onUnauthorized: () => void;
}) {
  getAuthToken = opts.getToken;
  onUnauthorized = opts.onUnauthorized;
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.body instanceof FormData ? {} : { 'content-type': 'application/json' }),
    ...((init.headers as Record<string, string>) || {})
  };
  const token = getAuthToken();
  if (token && !('Authorization' in headers) && !('authorization' in headers)) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(`/api${path}`, { ...init, headers });
  const body = await parseResponseBody(response);
  if (!response.ok) {
    const errorBody = body && typeof body === 'object' && !Array.isArray(body) ? body as Record<string, unknown> : {};
    const details = errorBody.details && typeof errorBody.details === 'object' && !Array.isArray(errorBody.details)
      ? errorBody.details as Record<string, unknown>
      : { status: response.status };
    const code = typeof errorBody.error_code === 'string' ? errorBody.error_code : 'api_error';
    // 401 on a non-login route means the stored token is invalid or expired.
    // Wipe state and bounce to login (caller of the page also gets the rejection).
    if (response.status === 401 && path !== '/auth/login') {
      onUnauthorized();
    }
    throw new ApiError(
      code,
      typeof errorBody.message === 'string' ? errorBody.message : `Request failed with status ${response.status}`,
      details,
      response.status
    );
  }
  return body as T;
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_error) {
    if (!response.ok) return text;
    throw new ApiError('invalid_json_response', 'API returned an invalid JSON response', { status: response.status }, response.status);
  }
}
