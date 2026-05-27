import { ApiError } from '../api/client';

export type TranslateFn = (key: string, params?: Record<string, unknown>) => string;
export type TranslationExistsFn = (key: string) => boolean;
export type MessageState = { key: string; params?: Record<string, unknown> } | { raw: string } | null;

export function messageFromError(
  error: unknown,
  te: TranslationExistsFn,
  fallbackKey = 'errors.apiError'
): MessageState {
  if (error instanceof ApiError) {
    const key = `errors.${error.error_code}`;
    if (te(key)) return { key };
    return error.message ? { raw: error.message } : { key: fallbackKey };
  }
  if (error instanceof Error && error.message) return { raw: error.message };
  return { key: fallbackKey };
}

export function renderMessage(message: MessageState, t: TranslateFn): string {
  if (!message) return '';
  if ('raw' in message) return message.raw;
  return t(message.key, message.params);
}

export function displayError(
  error: unknown,
  t: TranslateFn,
  te: TranslationExistsFn,
  fallbackKey = 'errors.apiError'
): string {
  return renderMessage(messageFromError(error, te, fallbackKey), t);
}
