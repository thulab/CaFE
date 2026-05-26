import { fireEvent, render, screen } from '@testing-library/vue';
import { afterEach, describe, expect, it } from 'vitest';
import StateBlock from '../components/ui/StateBlock.vue';
import { i18n, setLocale } from '../i18n';

function renderStateBlock(props: Record<string, unknown>) {
  return render(StateBlock, {
    props,
    global: { plugins: [i18n] },
  });
}

describe('StateBlock', () => {
  afterEach(() => setLocale('en-US'));

  it('renders default loading text in English and Chinese', () => {
    const loading = renderStateBlock({ loading: true });
    expect(screen.getByText('Loading...')).toBeTruthy();

    loading.unmount();
    setLocale('zh-CN');
    renderStateBlock({ loading: true });
    expect(screen.getByText('加载中...')).toBeTruthy();
  });

  it('renders default error text and retry label in English and Chinese', async () => {
    const error = renderStateBlock({ error: 'Network unavailable' });
    expect(screen.getByText('Something went wrong')).toBeTruthy();
    expect(screen.getByRole('button', { name: /Try again/ })).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: /Try again/ }));
    expect(error.emitted('retry')).toHaveLength(1);

    error.unmount();
    setLocale('zh-CN');
    renderStateBlock({ error: 'Network unavailable' });
    expect(screen.getByText('出了点问题')).toBeTruthy();
    expect(screen.getByRole('button', { name: /重试/ })).toBeTruthy();
  });

  it('renders default empty text in English and Chinese', () => {
    const empty = renderStateBlock({ empty: true });
    expect(screen.getByText('Nothing here yet')).toBeTruthy();

    empty.unmount();
    setLocale('zh-CN');
    renderStateBlock({ empty: true });
    expect(screen.getByText('暂无内容')).toBeTruthy();
  });
});
