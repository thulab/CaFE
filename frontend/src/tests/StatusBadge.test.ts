import { render, screen } from '@testing-library/vue';
import { afterEach, describe, expect, it } from 'vitest';
import StatusBadge from '../components/ui/StatusBadge.vue';
import { i18n, setLocale } from '../i18n';

function renderStatusBadge(props: Record<string, unknown>) {
  return render(StatusBadge, {
    props,
    global: { plugins: [i18n] },
  });
}

describe('StatusBadge', () => {
  afterEach(() => setLocale('en-US'));

  it('translates a known status in English and Chinese', () => {
    const badge = renderStatusBadge({ status: 'running' });
    expect(screen.getByText('Running')).toBeTruthy();

    badge.unmount();
    setLocale('zh-CN');
    renderStatusBadge({ status: 'running' });
    expect(screen.getByText('运行中')).toBeTruthy();
  });

  it('honors an explicit label prop', () => {
    renderStatusBadge({ status: 'running', label: 'Custom label' });
    expect(screen.getByText('Custom label')).toBeTruthy();
  });
});
