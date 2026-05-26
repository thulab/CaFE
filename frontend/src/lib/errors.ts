import { ApiError } from '../api/client';

export type TranslateFn = (key: string, params?: Record<string, unknown>) => string;
export type TranslationExistsFn = (key: string) => boolean;

export function displayError(
  error: unknown,
  t: TranslateFn,
  te: TranslationExistsFn,
  fallbackKey = 'errors.apiError'
): string {
  if (error instanceof ApiError) {
    const key = `errors.${error.error_code}`;
    if (te(key)) return t(key);
    return error.message || t(fallbackKey);
  }
  if (error instanceof Error && error.message) return error.message;
  return t(fallbackKey);
}
