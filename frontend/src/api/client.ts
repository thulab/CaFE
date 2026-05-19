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
  const body = await parseResponseBody(response);
  if (!response.ok) {
    const errorBody = body && typeof body === 'object' && !Array.isArray(body) ? body as Record<string, unknown> : {};
    const details = errorBody.details && typeof errorBody.details === 'object' && !Array.isArray(errorBody.details)
      ? errorBody.details as Record<string, unknown>
      : { status: response.status };
    throw new ApiError(
      typeof errorBody.error_code === 'string' ? errorBody.error_code : 'api_error',
      typeof errorBody.message === 'string' ? errorBody.message : `Request failed with status ${response.status}`,
      details
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
    throw new ApiError('invalid_json_response', 'API returned an invalid JSON response', { status: response.status });
  }
}
