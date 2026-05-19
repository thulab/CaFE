import { apiRequest } from './client';
import type {
  DatasetLoadJobCreateDTO,
  DatasetLoadJobDetailDTO,
  DatasetLoadJobDTO,
  DatasetManifestCreateDTO,
  DatasetManifestDTO,
  ShardDTO,
  ShardSamplesDTO,
  UploadPreviewDTO
} from './types';

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

export function getDatasetManifest(datasetManifestId: string): Promise<DatasetManifestDTO> {
  return apiRequest<DatasetManifestDTO>(`/dataset-manifests/${datasetManifestId}`);
}

export function getLoadJob(loadJobId: string): Promise<DatasetLoadJobDetailDTO> {
  return apiRequest<DatasetLoadJobDetailDTO>(`/dataset-load-jobs/${loadJobId}`);
}

export function getShard(shardId: string): Promise<ShardDTO> {
  return apiRequest<ShardDTO>(`/shards/${shardId}`);
}

export function getShardSamples(shardId: string): Promise<ShardSamplesDTO> {
  return apiRequest<ShardSamplesDTO>(`/shards/${shardId}/samples`);
}
