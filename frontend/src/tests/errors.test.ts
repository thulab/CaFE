import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import enUS from '../i18n/locales/en-US';
import zhCN from '../i18n/locales/zh-CN';
import { displayError } from '../lib/errors';

function lookup(catalog: Record<string, unknown>, key: string): string | undefined {
  const value = key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], catalog);
  return typeof value === 'string' ? value : undefined;
}

const t = (catalog: Record<string, unknown>) => (key: string) => lookup(catalog, key) ?? key;
const te = (catalog: Record<string, unknown>) => (key: string) => lookup(catalog, key) !== undefined;

describe('displayError', () => {
  it('maps known ApiError codes to the active locale', () => {
    const error = new ApiError('auth_required', 'login required', {}, 401);
    expect(displayError(error, t(enUS), te(enUS))).toBe('Please sign in to continue.');
    expect(displayError(error, t(zhCN), te(zhCN))).toBe('请先登录。');
  });

  it('falls back to backend message for unknown ApiError codes', () => {
    const error = new ApiError('new_backend_code', 'Backend fallback', {}, 400);
    expect(displayError(error, t(enUS), te(enUS))).toBe('Backend fallback');
  });

  it('uses translated fallback for non-error values', () => {
    expect(displayError(null, t(enUS), te(enUS))).toBe('Request failed');
    expect(displayError(null, t(zhCN), te(zhCN))).toBe('请求失败');
  });
});
