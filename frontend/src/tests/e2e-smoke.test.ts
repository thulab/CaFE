import { render, screen } from '@testing-library/vue';
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
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ items: [{ model_id: 'm1', name: 'Timer 3.5', adapter_type: 'timer_service' }] }), { status: 200 })));
    wizardState.preview = { upload_id: 'upload-1', source_uri: '/tmp/hourly.csv', columns: [{ name: 'time' }, { name: 'target' }], preview_rows: [] };
    wizardState.shardId = 'shard-1';
    wizardState.trackId = 'track-1';
    wizardState.step = 4;

    const rendered = render(App, { global: { plugins: [i18n] } });

    // App shell + wizard sub-page
    expect(screen.getByText('TSBenchmark')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'New evaluation' })).toBeTruthy();
    rendered.unmount();

    render(RunStep);
    expect(await screen.findByText('Timer 3.5')).toBeTruthy();
  });
});
