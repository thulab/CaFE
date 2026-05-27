import { describe, expect, it } from 'vitest';
import { useAsyncData } from '../composables/useAsync';
import { setLocale } from '../i18n';

describe('useAsyncData', () => {
  it('uses the active locale for non-error fallback failures', async () => {
    setLocale('zh-CN');

    const state = useAsyncData(() => Promise.reject('offline'));

    await state.run();

    expect(state.error.value).toBe('请求失败');
    setLocale('en-US');
  });
});
