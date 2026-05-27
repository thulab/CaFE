import { afterEach, describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import { useAsyncData } from '../composables/useAsync';
import { setLocale } from '../i18n';

describe('useAsyncData', () => {
  afterEach(() => setLocale('en-US'));

  it('uses the active locale for non-error fallback failures', async () => {
    setLocale('zh-CN');

    const state = useAsyncData(() => Promise.reject('offline'));

    await state.run();

    expect(state.error.value).toBe('请求失败');
  });

  it('maps ApiError codes through the active locale and updates after locale changes', async () => {
    const state = useAsyncData(() => Promise.reject(new ApiError('forbidden', 'raw forbidden', {}, 403)));

    await state.run();

    expect(state.error.value).toBe('You do not have permission to perform this action.');

    setLocale('zh-CN');

    expect(state.error.value).toBe('你没有执行此操作的权限。');
  });
});
