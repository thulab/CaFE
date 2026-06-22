export const LOCALES = ['en-US', 'zh-CN'] as const;
export type LocaleCode = typeof LOCALES[number];

export const DEFAULT_LOCALE: LocaleCode = 'en-US';
export const STORAGE_KEY = 'tsbenchmark.locale';

export function isLocaleCode(value: unknown): value is LocaleCode {
  return typeof value === 'string' && (LOCALES as readonly string[]).includes(value);
}

export function normalizeLocale(value: unknown): LocaleCode | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  const normalized = value.trim();
  if (isLocaleCode(normalized)) return normalized;
  const lower = normalized.toLowerCase();
  if (lower === 'zh' || lower.startsWith('zh-')) return 'zh-CN';
  if (lower === 'en' || lower.startsWith('en-')) return 'en-US';
  return null;
}

export function readLocaleFromUrl(search = window.location.search, hash = window.location.hash): LocaleCode | null {
  const direct = new URLSearchParams(search).get('lang');
  const fromDirect = normalizeLocale(direct);
  if (fromDirect) return fromDirect;

  const queryIndex = hash.indexOf('?');
  if (queryIndex < 0) return null;
  const hashQuery = hash.slice(queryIndex + 1);
  return normalizeLocale(new URLSearchParams(hashQuery).get('lang'));
}

export function readStoredLocale(storage: Pick<Storage, 'getItem'> = window.localStorage): LocaleCode | null {
  try {
    return normalizeLocale(storage.getItem(STORAGE_KEY));
  } catch (_error) {
    return null;
  }
}

export function resolveInitialLocale(): LocaleCode {
  return readLocaleFromUrl()
    ?? readStoredLocale()
    ?? normalizeLocale(window.navigator.language)
    ?? DEFAULT_LOCALE;
}
