import { render, screen } from '@testing-library/vue';
import { beforeEach, describe, expect, it } from 'vitest';
import HomePage from '../pages/HomePage.vue';
import DatasetsPage from '../pages/DatasetsPage.vue';
import RunsPage from '../pages/RunsPage.vue';
import { clearRecents, recordRecent } from '../stores/recents';

describe('workspace pages', () => {
  beforeEach(() => clearRecents());

  it('renders the overview with an empty activity state', () => {
    render(HomePage);
    expect(screen.getByRole('heading', { name: 'Workbench overview' })).toBeTruthy();
    expect(screen.getByText('No activity yet')).toBeTruthy();
  });

  it('lists recorded datasets and shards', () => {
    recordRecent({ kind: 'dataset', id: 'manifest-9', title: 'Uploaded dataset', href: '#/datasets/manifest-9' });
    recordRecent({ kind: 'shard', id: 'shard-9', title: 'Shard · target', href: '#/shards/shard-9' });

    render(DatasetsPage);
    expect(screen.getByRole('heading', { name: 'Datasets' })).toBeTruthy();
    expect(screen.getByText('Uploaded dataset')).toBeTruthy();
    expect(screen.getByText('Shard · target')).toBeTruthy();
  });

  it('shows an empty runs state until a run is recorded', async () => {
    render(RunsPage);
    expect(screen.getByRole('heading', { name: 'Runs' })).toBeTruthy();
    expect(screen.getByText('No runs yet')).toBeTruthy();

    clearRecents();
    recordRecent({ kind: 'run', id: 'run-9', title: 'Run · 2 models', subtitle: 'running', href: '#/runs/run-9' });
    render(RunsPage);
    expect((await screen.findAllByText('Run · 2 models')).length).toBeGreaterThan(0);
  });
});
