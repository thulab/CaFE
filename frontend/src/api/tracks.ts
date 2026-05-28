import { apiRequest } from './client';
import type { TrackDTO } from './types';

export function listTracks() {
  return apiRequest<{ items: TrackDTO[] }>('/tracks');
}

export function getTrack(trackId: string) {
  return apiRequest<TrackDTO>(`/tracks/${encodeURIComponent(trackId)}`);
}

export function createRealDatasetTrack(payload: { name: string; shard_ids: string[]; primary_metric_id: string }) {
  return apiRequest<{ track_id: string; capability_block_id: string; ranking_list_id: string }>('/wizard/real-dataset-track', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
