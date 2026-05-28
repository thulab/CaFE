import { apiRequest } from './client';
import type { DeletionImpactDTO, ResourceType } from './types';

export type LifecycleAction = 'archive' | 'restore' | 'purge';

const PATHS: Record<ResourceType, string> = {
  dataset_manifest: '/dataset-manifests',
  shard: '/shards',
  track: '/tracks',
  benchmarking_run: '/benchmarking-runs',
};

function resourcePath(resourceType: ResourceType, resourceId: string): string {
  return `${PATHS[resourceType]}/${encodeURIComponent(resourceId)}`;
}

export function getDeletionImpact(resourceType: ResourceType, resourceId: string): Promise<DeletionImpactDTO> {
  return apiRequest<DeletionImpactDTO>(`${resourcePath(resourceType, resourceId)}/deletion-impact`);
}

export function archiveResource(resourceType: ResourceType, resourceId: string) {
  return apiRequest<{ resource_type: ResourceType; resource_id: string; archived_at?: string | null }>(
    `${resourcePath(resourceType, resourceId)}/archive`,
    { method: 'POST' }
  );
}

export function restoreResource(resourceType: ResourceType, resourceId: string) {
  return apiRequest<{ resource_type: ResourceType; resource_id: string; archived_at?: string | null }>(
    `${resourcePath(resourceType, resourceId)}/restore`,
    { method: 'POST' }
  );
}

export function purgeResource(resourceType: ResourceType, resourceId: string, cascade = true) {
  return apiRequest<{ ok: true; purged?: DeletionImpactDTO }>(
    `${resourcePath(resourceType, resourceId)}?cascade=${cascade ? 'true' : 'false'}`,
    { method: 'DELETE' }
  );
}
