import { render, screen, waitFor } from '@testing-library/vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import LoadShardStep from '../components/wizard/LoadShardStep.vue';
import { resetWizard, wizardState } from '../stores/wizard';

describe('LoadShardStep', () => {
  beforeEach(() => {
    resetWizard();
    wizardState.shardId = 'shard-1';
    vi.restoreAllMocks();
  });

  it('displays shard sample summary', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ sample_id: 's1' }, { sample_id: 's2' }] }), { status: 200 }));

    render(LoadShardStep);

    await waitFor(() => expect(screen.getByText('2 samples')).toBeTruthy());
  });
});
