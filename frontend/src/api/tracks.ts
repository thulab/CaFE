import { apiRequest } from './client';

export function createRealDatasetTrack(payload: { name: string; shard_ids: string[]; primary_metric_id: string }) {
  return apiRequest<{ track_id: string; capability_block_id: string; ranking_list_id: string }>('/wizard/real-dataset-track', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
