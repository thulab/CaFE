import { apiRequest } from './client';
import type { DatasetLoadJobCreateDTO, DatasetLoadJobDTO, DatasetManifestCreateDTO, UploadPreviewDTO } from './types';

export function uploadDataset(file: File): Promise<UploadPreviewDTO> {
  const body = new FormData();
  body.append('file', file);
  return apiRequest<UploadPreviewDTO>('/dataset-manifests/upload', { method: 'POST', body });
}

export function createDatasetManifest(payload: DatasetManifestCreateDTO) {
  return apiRequest<{ dataset_manifest_id: string }>('/dataset-manifests', { method: 'POST', body: JSON.stringify(payload) });
}

export function createLoadJob(payload: DatasetLoadJobCreateDTO): Promise<DatasetLoadJobDTO> {
  return apiRequest<DatasetLoadJobDTO>('/dataset-load-jobs', { method: 'POST', body: JSON.stringify(payload) });
}

export function getShardSamples(shardId: string) {
  return apiRequest<{ items: Array<{ sample_id: string }> }>(`/shards/${shardId}/samples`);
}
