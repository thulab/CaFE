export class ApiError extends Error {
  constructor(
    public error_code: string,
    message: string,
    public details: Record<string, unknown> = {}
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: init.body instanceof FormData ? init.headers : { 'content-type': 'application/json', ...(init.headers || {}) },
    ...init
  });
  const body = await response.json();
  if (!response.ok) {
    throw new ApiError(body.error_code || 'api_error', body.message || 'Request failed', body.details || {});
  }
  return body as T;
}
