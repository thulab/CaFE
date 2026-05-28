import { render, screen } from '@testing-library/vue';
import { nextTick } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.vue';
import RunStep from '../components/wizard/RunStep.vue';
import { i18n, setLocale } from '../i18n';
import { authState } from '../stores/auth';
import { resetWizard, wizardState } from '../stores/wizard';

describe('frontend smoke flow', () => {
  beforeEach(() => {
    resetWizard();
    setLocale('en-US');
    authState.user = {
      user_id: 'admin-test',
      username: 'admin',
      email: null,
      is_active: true,
      is_superuser: true,
      roles: ['admin'],
      permissions: []
    };
    vi.restoreAllMocks();
    window.location.hash = '#/new';
  });

  it('renders the workbench shell, the guided wizard, and model-backed run controls', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/models') {
        return Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 }));
      }
      if (url === '/api/tracks') {
        return Promise.resolve(new Response(JSON.stringify({ items: [] }), { status: 200 }));
      }
      return Promise.resolve(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    });
    wizardState.preview = { upload_id: 'upload-1', source_uri: '/tmp/hourly.csv', columns: [{ name: 'time' }, { name: 'target' }], preview_rows: [] };
    wizardState.shardId = 'shard-1';
    wizardState.trackId = 'track-1';
    wizardState.step = 4;

    const rendered = render(App, { global: { plugins: [i18n] } });

    // App shell + wizard sub-page
    expect(screen.getByText('TSBenchmark')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'New evaluation' })).toBeTruthy();
    rendered.unmount();

    render(RunStep, { global: { plugins: [i18n] } });
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();
  });

  it('updates guided wizard step labels when switching to Chinese', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    const rendered = render(App, { global: { plugins: [i18n] } });

    expect(screen.getAllByText('Upload data').length).toBeGreaterThan(0);

    setLocale('zh-CN');
    await nextTick();

    expect(screen.getAllByText('上传数据').length).toBeGreaterThan(0);
    rendered.unmount();
  });
});
