import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { normalizeLocale, readLocaleFromUrl, STORAGE_KEY } from '../i18n/keys';
import enUS from '../i18n/locales/en-US';
import zhCN from '../i18n/locales/zh-CN';

function flattenKeys(value: unknown, prefix = ''): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [prefix];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
    flattenKeys(child, prefix ? `${prefix}.${key}` : key)
  );
}

function valueAtKey(catalog: unknown, key: string): unknown {
  return key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)[part], catalog);
}

function extractPlaceholders(value: string): string[] {
  return Array.from(value.matchAll(/\{([^{}]+)\}/g), ([, placeholder]) => placeholder).sort();
}

function installStorage(initialValues: Record<string, string> = {}) {
  const values = { ...initialValues };
  const storage = {
    getItem: vi.fn((key: string) => values[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      values[key] = String(value);
    }),
  } satisfies Pick<Storage, 'getItem' | 'setItem'>;

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: storage,
  });

  return storage;
}

describe('i18n locale catalogs', () => {
  it('keeps English and Chinese key sets aligned', () => {
    expect(flattenKeys(zhCN).sort()).toEqual(flattenKeys(enUS).sort());
  });

  it('does not ship empty translation values', () => {
    for (const catalog of [enUS, zhCN]) {
      const keys = flattenKeys(catalog);
      for (const key of keys) {
        const value = valueAtKey(catalog, key);
        expect(typeof value === 'string' && value.length > 0, key).toBe(true);
      }
    }
  });

  it('keeps interpolation placeholders aligned for every shared message', () => {
    for (const key of flattenKeys(enUS)) {
      const english = valueAtKey(enUS, key);
      const chinese = valueAtKey(zhCN, key);

      expect(extractPlaceholders(english as string), key).toEqual(extractPlaceholders(chinese as string));
    }
  });
});

describe('locale resolution', () => {
  it('normalizes supported browser locale variants', () => {
    expect(normalizeLocale('zh')).toBe('zh-CN');
    expect(normalizeLocale('zh-Hans')).toBe('zh-CN');
    expect(normalizeLocale('en')).toBe('en-US');
    expect(normalizeLocale('fr-FR')).toBeNull();
  });

  it('reads lang from page query and hash query', () => {
    expect(readLocaleFromUrl('?lang=zh-CN', '#/runs')).toBe('zh-CN');
    expect(readLocaleFromUrl('', '#/runs?lang=en-US')).toBe('en-US');
    expect(readLocaleFromUrl('', '#/runs')).toBeNull();
  });
});

describe('i18n runtime', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    installStorage();
    window.history.pushState({}, '', '/');
    document.documentElement.removeAttribute('lang');
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.setItem(STORAGE_KEY, '');
    window.history.pushState({}, '', '/');
    document.documentElement.removeAttribute('lang');
  });

  it('prefers URL locale over stored locale on initialization', async () => {
    installStorage({ [STORAGE_KEY]: 'zh-CN' });
    window.history.pushState({}, '', '/?lang=en-US#/runs');

    const { i18n } = await import('../i18n');

    expect(i18n.global.locale.value).toBe('en-US');
    expect(document.documentElement.lang).toBe('en-US');
  });

  it('uses stored locale when URL has no locale', async () => {
    installStorage({ [STORAGE_KEY]: 'zh-CN' });
    window.history.pushState({}, '', '/#/runs');

    const { i18n } = await import('../i18n');

    expect(i18n.global.locale.value).toBe('zh-CN');
    expect(document.documentElement.lang).toBe('zh-CN');
  });

  it('updates runtime locale, document lang, and storage', async () => {
    const { i18n, setLocale } = await import('../i18n');

    setLocale('zh-CN');

    expect(i18n.global.locale.value).toBe('zh-CN');
    expect(document.documentElement.lang).toBe('zh-CN');
    expect(window.localStorage.getItem('tsbenchmark.locale')).toBe('zh-CN');
  });

  it('falls back for unknown locale input', async () => {
    const { i18n, setLocaleFromUnknown } = await import('../i18n');

    expect(setLocaleFromUnknown('fr-FR')).toBe('en-US');
    expect(i18n.global.locale.value).toBe('en-US');
    expect(document.documentElement.lang).toBe('en-US');
    expect(window.localStorage.getItem('tsbenchmark.locale')).toBe('en-US');
  });

  it('still updates runtime locale when storage write fails', async () => {
    const { i18n, setLocale } = await import('../i18n');
    vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new Error('storage unavailable');
    });

    expect(() => setLocale('zh-CN')).not.toThrow();
    expect(i18n.global.locale.value).toBe('zh-CN');
    expect(document.documentElement.lang).toBe('zh-CN');
  });
});
