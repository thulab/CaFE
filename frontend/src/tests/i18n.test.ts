import { describe, expect, it } from 'vitest';
import { normalizeLocale, readLocaleFromUrl } from '../i18n/keys';
import enUS from '../i18n/locales/en-US';
import zhCN from '../i18n/locales/zh-CN';

function flattenKeys(value: unknown, prefix = ''): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [prefix];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
    flattenKeys(child, prefix ? `${prefix}.${key}` : key)
  );
}

describe('i18n locale catalogs', () => {
  it('keeps English and Chinese key sets aligned', () => {
    expect(flattenKeys(zhCN).sort()).toEqual(flattenKeys(enUS).sort());
  });

  it('does not ship empty translation values', () => {
    for (const catalog of [enUS, zhCN]) {
      const keys = flattenKeys(catalog);
      for (const key of keys) {
        const value = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)[part], catalog);
        expect(typeof value === 'string' && value.length > 0, key).toBe(true);
      }
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
